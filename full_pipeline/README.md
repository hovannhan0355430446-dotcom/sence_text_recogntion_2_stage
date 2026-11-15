# End-to-End Scene Text Recognition (STR) Pipeline

Hệ thống nhận diện chữ tiếng Việt trong ảnh tự nhiên gồm 2 giai đoạn:
1. **Text Detection**: DBNet++ (Backbone ResNet-50 + Adaptive Scale Fusion - ASF + DBHead) phát hiện vị trí polygon và bounding box của từng vùng chữ.
2. **Text Recognition**: PARSeq (Permutation Autoregressive Sequence) nhận dạng chuỗi ký tự tiếng Việt với bộ từ vựng 233 lớp ký tự.

## Cấu trúc thư mục

```
full_pipeline/
├── weights/
│   ├── dbnet_best.pth               # Trọng số huấn luyện tốt nhất của DBNet++
│   ├── parseq_vietnamese_best.pth   # Trọng số huấn luyện tốt nhất của PARSeq
│   └── vocab.json                   # Bảng mã 233 ký tự tiếng Việt và ký tự đặc biệt
├── detection.py                     # Định nghĩa kiến trúc DBNet++ và bộ tách polygon
├── recognition.py                   # Định nghĩa bộ tiền xử lý và suy luận mô hình PARSeq
├── pipeline.py                      # Tích hợp phát hiện, cắt nắn đa giác, sắp xếp thứ tự đọc và trực quan hóa
├── app.py                           # Máy chủ FastAPI cung cấp giao diện Web và REST API
├── run.py                           # Script khởi động máy chủ
├── start.bat                        # File batch chạy nhanh trên Windows
├── templates/
│   └── index.html                   # Giao diện HTML
└── static/
    ├── css/style.css                # Định kiểu giao diện
    └── js/script.js                 # Xử lý kéo thả ảnh, gọi API và tương tác canvas
```

## Cách chạy

### Cách 1: Sử dụng file batch
Nhấp đúp chuột vào file `start.bat` trong thư mục `full_pipeline`.

### Cách 2: Chạy lệnh từ terminal
```bash
cd full_pipeline
..\.venv\Scripts\python.exe run.py
```

Truy cập giao diện tại: `http://127.0.0.1:8000`

## Tính năng giao diện
- Tải ảnh lên bằng cách kéo thả hoặc chọn từ máy tính.
- Nút chọn nhanh các ảnh chụp thực tế có sẵn (`img1.jpg`, `img2.webp`).
- Tùy chỉnh tham số suy luận: `thresh`, `box_thresh`, `unclip_ratio`.
- Chuyển đổi xem ảnh trực quan (có khung viền và số thứ tự) hoặc ảnh gốc.
- Tự động sắp xếp văn bản theo thứ tự đọc tự nhiên từ trên xuống dưới, từ trái sang phải.
- Xem bảng chi tiết từng từ: số thứ tự, ảnh cắt, nội dung nhận diện, độ tin cậy phát hiện và nhận dạng.
- Di chuột qua bảng để làm nổi bật vị trí vùng chữ tương ứng trên ảnh.
- Nút sao chép nhanh toàn bộ văn bản vào clipboard.

