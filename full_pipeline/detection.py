from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from shapely.geometry import Polygon

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

class ResNetBackbone(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=False):
        super().__init__()
        backbones = {
            "resnet18": (tv_models.resnet18, [64, 128, 256, 512]),
            "resnet34": (tv_models.resnet34, [64, 128, 256, 512]),
            "resnet50": (tv_models.resnet50, [256, 512, 1024, 2048]),
        }
        constructor, self.out_channels = backbones[model_name]
        resnet = constructor(weights=None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]

class ASF(nn.Module):
    def __init__(self, in_channels, inter_channels=64):
        super().__init__()
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_attn = nn.Sequential(nn.Conv2d(inter_channels, 1, 3, padding=1, bias=False), nn.Sigmoid())
        self.scale_attn = nn.Sequential(nn.Conv2d(inter_channels, 4, 1, bias=False), nn.Sigmoid())

    def forward(self, features):
        val = self.conv_init(torch.cat(features, dim=1))
        val = val * self.spatial_attn(val) + val
        w = self.scale_attn(val)
        return torch.cat([f * w[:, i : i + 1] for i, f in enumerate(features)], dim=1)

class DBNetFPN(nn.Module):
    def __init__(self, in_channels, inner_channels=256, dropout=0.20):
        super().__init__()
        out_channels = inner_channels // 4
        self.in_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, inner_channels, 1, bias=False),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])
        self.out_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inner_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels
        ])
        self.dropout = nn.Dropout2d(p=dropout)
        self.asf = ASF(inner_channels, inter_channels=out_channels)

    def forward(self, features):
        c2, c3, c4, c5 = features
        in5 = self.in_convs[3](c5)
        in4 = self.in_convs[2](c4) + F.interpolate(in5, size=c4.shape[2:], mode="nearest")
        in3 = self.in_convs[1](c3) + F.interpolate(in4, size=c3.shape[2:], mode="nearest")
        in2 = self.in_convs[0](c2) + F.interpolate(in3, size=c2.shape[2:], mode="nearest")

        p5 = F.interpolate(self.out_convs[3](in5), size=c2.shape[2:], mode="nearest")
        p4 = F.interpolate(self.out_convs[2](in4), size=c2.shape[2:], mode="nearest")
        p3 = F.interpolate(self.out_convs[1](in3), size=c2.shape[2:], mode="nearest")
        p2 = self.out_convs[0](in2)
        return self.dropout(self.asf([p5, p4, p3, p2]))

class DBHead(nn.Module):
    def __init__(self, in_channels=256, k=50):
        super().__init__()
        self.k = k
        intermediate = in_channels // 4
        self.binarize = self._make_up_head(in_channels, intermediate)
        self.thresh = self._make_up_head(in_channels, intermediate)

    @staticmethod
    def _make_up_head(in_channels, intermediate):
        return nn.Sequential(
            nn.Conv2d(in_channels, intermediate, 3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(intermediate, intermediate, 2, 2),
            nn.BatchNorm2d(intermediate),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(intermediate, 1, 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, fused, return_loss=False):
        probability = self.binarize(fused)
        if not self.training and not return_loss:
            return probability
        threshold = self.thresh(fused)
        binary = torch.reciprocal(1.0 + torch.exp(-self.k * (probability - threshold)))
        return torch.cat([probability, threshold, binary], dim=1)

class DBNetPP(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=False, k=50):
        super().__init__()
        self.backbone = ResNetBackbone(model_name, pretrained)
        self.fpn = DBNetFPN(self.backbone.out_channels, inner_channels=256, dropout=0.20)
        self.head = DBHead(in_channels=256, k=k)

    def forward(self, x, return_loss=False):
        return self.head(self.fpn(self.backbone(x)), return_loss=return_loss)

def seg_to_polygons(prob_map, thresh=0.3, box_thresh=0.5, unclip_ratio=1.7, min_size=3.0):
    bitmap = (prob_map > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons, scores = [], []
    for contour in contours[:1000]:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if points.shape[0] < 4:
            continue
        x_min = max(int(points[:, 0].min()), 0)
        x_max = min(int(points[:, 0].max()), prob_map.shape[1] - 1)
        y_min = max(int(points[:, 1].min()), 0)
        y_max = min(int(points[:, 1].max()), prob_map.shape[0] - 1)
        if x_max <= x_min or y_max <= y_min:
            continue

        local_mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        cv2.fillPoly(local_mask, [points.astype(np.int32) - np.array([x_min, y_min])], 1)
        score = float(cv2.mean(prob_map[y_min : y_max + 1, x_min : x_max + 1], local_mask)[0])
        if score < box_thresh:
            continue

        shape = Polygon(points).buffer(0)
        if shape.is_empty or shape.area < 5 or shape.length <= 0:
            continue
        expanded = shape.buffer(shape.area * unclip_ratio / shape.length, join_style="round")
        if expanded.is_empty:
            continue
        if expanded.geom_type == "MultiPolygon":
            expanded = max(expanded.geoms, key=lambda p: p.area)
        exp_pts = np.asarray(expanded.exterior.coords[:-1], dtype=np.float32)
        if len(exp_pts) >= 4:
            _, (w, h), _ = cv2.minAreaRect(exp_pts)
            if min(w, h) >= min_size:
                polygons.append(exp_pts)
                scores.append(score)
    return polygons, scores

class TextDetector:
    def __init__(self, weights_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = DBNetPP(model_name="resnet50", pretrained=False).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def detect(self, rgb_image: np.ndarray, target_size: int = 800, thresh: float = 0.3, box_thresh: float = 0.5, unclip_ratio: float = 1.7):
        h, w = rgb_image.shape[:2]
        scale = min(target_size / max(1, h), target_size / max(1, w))
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))

        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(rgb_image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        norm = (canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        inp = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        prob_map = self.model(inp, return_loss=False)[0, 0].cpu().numpy()[:nh, :nw]
        polys, scores = seg_to_polygons(prob_map, thresh=thresh, box_thresh=box_thresh, unclip_ratio=unclip_ratio)

        scaled_polys = []
        for poly in polys:
            p = poly / scale
            p[:, 0] = np.clip(p[:, 0], 0, w - 1)
            p[:, 1] = np.clip(p[:, 1], 0, h - 1)
            scaled_polys.append(p)

        return scaled_polys, scores

