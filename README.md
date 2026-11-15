# Vietnamese Scene Text Recognition

Ứng dụng nhận diện chữ tiếng Việt trong ảnh tự nhiên bằng DBNet++ để phát hiện vùng chữ và PARSeq để nhận dạng nội dung.

## Yêu cầu

- Python 3.10 trở lên
- Git
- Trọng số mô hình tải riêng từ [Google Drive](https://drive.google.com/drive/folders/1A8N7d7QgelwojPgUGh6FFU0w-vfQPP6M?usp=sharing)

## Cài đặt trên Windows

Tạo môi trường ảo và cài dependencies từ thư mục gốc dự án:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tải hai trọng số sau từ Google Drive và đặt vào `full_pipeline\weights\`:

- `dbnet_best.pth`
- `parseq_vietnamese_best.pth`

Giữ nguyên file `full_pipeline\weights\vocab.json` có sẵn trong repository. Nếu Google Drive yêu cầu quyền truy cập, hãy liên hệ chủ dự án.

## Chạy ứng dụng web

Từ thư mục gốc dự án:

```powershell
python full_pipeline\run.py
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000). Nhấn `Ctrl+C` trong terminal để dừng máy chủ.

## Thư mục chính

- `full_pipeline\`: mã nguồn web, pipeline, giao diện và từ vựng của bộ nhận dạng.
- `full_pipeline\benchmark\`: mã nguồn đánh giá benchmark.
- `img_real_life\`: ảnh mẫu dùng trong giao diện.
- `notebook\`: notebook huấn luyện.

Dataset, trọng số mô hình, từ điển benchmark và các kết quả benchmark sinh ra không được lưu trong Git. Benchmark cần dataset VinText và một tệp từ điển được cung cấp riêng.

## Chạy benchmark

Từ thư mục gốc dự án, sau khi đặt dataset tại `dataset\vintext_official\vietnamese` và từ điển tại `full_pipeline\vn_dictionary.txt`, chạy:

```powershell
python full_pipeline\benchmark\benchmark.py
```

Benchmark lưu kết quả đầy đủ vào thư mục output và cập nhật `full_pipeline\benchmark\benchmark_summary.json`, dữ liệu được hiển thị trong trang web. Có thể chỉ định đường dẫn khác bằng `--dataset`, `--lexicon` và `--output`.

## Chia sẻ và giấy phép

Mã nguồn được công khai trên GitHub; các trọng số được chia sẻ riêng qua Google Drive. Hãy xác nhận quyền chia sẻ ảnh mẫu, trọng số và các tài nguyên bên ngoài trước khi sử dụng hoặc phân phối lại. Repository chưa kèm giấy phép sử dụng mã nguồn.
