# 📦 Phân Tích Video Sản Phẩm Với Claude 3.7 Sonnet (AWS Bedrock)

Ứng dụng Streamlit cho phép bạn **tải lên video sản phẩm** và sử dụng **Claude 3.7 Sonnet** trên **AWS Bedrock** để phân tích hình ảnh trích xuất từ video, xác định và đếm các loại sản phẩm khác nhau xuất hiện.

---

## 🧠 Chức năng chính

- Tải lên video (MP4, MOV, AVI, v.v.)
- Trích xuất frames bằng 3 phương pháp:
  - Số lượng đều đặn
  - Khoảng thời gian (giây)
  - Tự động phát hiện keyframes
- Gửi frames + prompt đến mô hình Claude 3.7 Sonnet (qua AWS Bedrock)
- Hiển thị kết quả phân tích dưới dạng văn bản

---

## 🚀 Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống

- Python 3.8+
- AWS Account có quyền sử dụng Bedrock và Claude 3.7 Sonnet
- Các thư viện Python sau:

```bash
pip install streamlit boto3 opencv-python
```

---

## ☁️ Cấu hình AWS

Ứng dụng yêu cầu quyền truy cập vào dịch vụ Amazon Bedrock và model `claude-3-sonnet`.

Bạn có thể cấu hình AWS theo 2 cách:

### 1. Sử dụng biến môi trường:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

---

## ✅ Yêu cầu hệ thống

- Các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## ▶️ Chạy ứng dụng

```bash
streamlit run main.py
```

```bash
http://localhost:8501
```
