# Hướng dẫn cấu hình Gemini AI

## 🚀 Thiết lập Gemini API

### 1. Lấy API Key từ Google AI Studio

1. Truy cập [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Đăng nhập bằng tài khoản Google
3. Tạo API key mới
4. Copy API key

### 2. Cấu hình trong dự án

#### Cách 1: Sử dụng file .env (Khuyến nghị)

```bash
# Tạo file .env trong thư mục backend
cd backend
touch .env
```

Thêm vào file `.env`:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

#### Cách 2: Sử dụng biến môi trường

```bash
export GEMINI_API_KEY="your_actual_api_key_here"
```

### 3. Cài đặt dependencies

```bash
# Kích hoạt virtual environment
source env_fake_new/bin/activate

# Cài đặt dependencies mới
pip install -r backend/requirements.txt
```

### 4. Test cấu hình

```bash
# Chạy backend
cd backend
python app.py
```

Kiểm tra log để thấy:
- ✅ "Gemini service initialized successfully" - Cấu hình thành công
- ⚠️ "Warning: GEMINI_API_KEY not found" - Cần cấu hình API key

## 🎯 Tính năng Gemini AI

Khi cấu hình thành công, ứng dụng sẽ có thêm:

### 📊 Phân tích chi tiết
- **Tóm tắt**: Tóm tắt ngắn gọn về văn bản
- **Dấu hiệu**: Các dấu hiệu tin giả/tin thật
- **Phân tích**: Giải thích chi tiết tại sao model đưa ra kết quả
- **Khuyến nghị**: Lời khuyên cho người dùng
- **Giải thích độ tin cậy**: Ý nghĩa của con số phần trăm

### 🔄 Fallback System
- Nếu Gemini không khả dụng, hệ thống sẽ sử dụng phân tích cơ bản
- Vẫn hoạt động bình thường mà không cần Gemini

## 🛠️ Troubleshooting

### Lỗi thường gặp:

1. **"GEMINI_API_KEY not found"**
   - Kiểm tra file .env có đúng vị trí không
   - Kiểm tra tên biến có đúng không

2. **"Failed to initialize Gemini service"**
   - Kiểm tra API key có hợp lệ không
   - Kiểm tra kết nối internet

3. **"Gemini analysis failed"**
   - API key có thể hết hạn
   - Quota API có thể đã hết

### Debug:

```python
# Thêm vào backend/app.py để debug
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 💰 Chi phí

- Gemini API có quota miễn phí hàng tháng
- Xem chi tiết tại [Google AI Pricing](https://ai.google.dev/pricing)

## 🔒 Bảo mật

- **KHÔNG** commit file .env vào git
- Sử dụng biến môi trường trong production
- Rotate API key định kỳ

## 📝 Ví dụ sử dụng

```bash
# Chạy với Gemini
./start_app.sh

# Test API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Tin tức cần phân tích", "include_analysis": true}'
```

Kết quả sẽ bao gồm:
```json
{
  "prediction": 0,
  "label": "Giả mạo",
  "confidence": 85.2,
  "text": "Tin tức đã phân tích",
  "analysis": {
    "success": true,
    "analysis": {
      "summary": "Tóm tắt từ Gemini...",
      "indicators": {...},
      "analysis": "Phân tích chi tiết...",
      "recommendations": "Khuyến nghị...",
      "confidence_explanation": "Giải thích độ tin cậy..."
    },
    "source": "gemini"
  }
}
```
