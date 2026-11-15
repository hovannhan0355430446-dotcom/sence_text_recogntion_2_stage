import json
import unicodedata
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T

def normalize_nfc(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return " ".join(text.split()).strip()

class TextRecognizer:
    def __init__(self, weights_path: str, vocab_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        chars = vocab_data["characters"]
        charset = "".join(chars)

        torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True)
        from strhub.models.utils import create_model

        self.model = create_model(
            "parseq",
            pretrained=False,
            charset_train=charset,
            charset_test=charset,
            max_label_length=25,
            img_size=(32, 128),
        ).to(self.device)

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((32, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.inference_mode()
    def predict_batch(self, crops_rgb: list, batch_size: int = 32):
        if not crops_rgb:
            return []

        results = []
        for i in range(0, len(crops_rgb), batch_size):
            chunk = crops_rgb[i : i + batch_size]
            tensors = [self.transform(Image.fromarray(c)) for c in chunk]
            batch = torch.stack(tensors).to(self.device)

            logits = self.model(batch)
            probs = logits.softmax(-1)
            preds, confs = self.model.tokenizer.decode(probs)

            for pred, conf in zip(preds, confs):
                clean_text = normalize_nfc(pred)
                mean_conf = float(conf.mean().item()) if hasattr(conf, "mean") else 1.0
                results.append((clean_text, mean_conf))

        return results

