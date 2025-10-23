# Ứng Dụng Phát Hiện Tin Giả

Ứng dụng AI sử dụng Deep Learning để phát hiện tin tức giả mạo bằng tiếng Việt.

## 🚀 Tính năng

- **Chat Interface**: Giao diện chat hiện đại giống ChatGPT
- **AI Analysis**: Tích hợp Gemini AI để phân tích chi tiết và giải thích kết quả
- **Dark Theme**: Giao diện tối chuyên nghiệp, dễ sử dụng
- **Real-time Chat**: Tương tác tự nhiên với AI
- **Responsive Design**: Hoạt động tốt trên mọi thiết bị
- **API RESTful**: Backend Flask với API endpoints rõ ràng

## 🏗️ Kiến trúc

```
Project_predict_Fake_New/
├── backend/                 # Flask API Server
│   ├── app.py             # Main Flask application
│   ├── requirements.txt   # Python dependencies
│   └── model/            # Trained model files
│       ├── model_state.pth
│       └── vocab.pkl
├── frontend/              # React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx
│   │   │   └── WelcomeScreen.tsx
│   │   └── App.tsx
│   └── package.json
├── dataset/              # Training data
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── train/               # Training notebook
    └── train.ipynb
```

## 🛠️ Cài đặt và Chạy

### Yêu cầu hệ thống
- Python 3.8+
- Node.js 16+
- npm hoặc yarn

### 1. Cài đặt Backend

```bash
# Kích hoạt virtual environment
source env_fake_new/bin/activate

# Cài đặt dependencies
pip install -r backend/requirements.txt

# Cấu hình Gemini API (tùy chọn)
# Xem GEMINI_SETUP.md để biết chi tiết
```

### 2. Cài đặt Frontend

```bash
cd frontend
npm install
```

### 3. Chạy ứng dụng

**Cách 1: Chạy riêng biệt**

```bash
# Terminal 1 - Backend
./start_backend.sh

# Terminal 2 - Frontend  
./start_frontend.sh
```

**Cách 2: Chạy thủ công**

```bash
# Backend (Terminal 1)
source env_fake_new/bin/activate
cd backend
python app.py

# Frontend (Terminal 2)
cd frontend
npm start
```

### 4. Truy cập ứng dụng

- **Chat Interface**: http://localhost:3000
- **Backend API**: http://localhost:5000

## 📡 API Endpoints

### POST /predict
Phân tích văn bản và trả về kết quả dự đoán.

**Request:**
```json
{
  "text": "Văn bản cần phân tích"
}
```

**Response:**
```json
{
  "prediction": 0,
  "label": "Giả mạo",
  "confidence": 85.2,
  "text": "Văn bản đã phân tích",
  "analysis": {
    "success": true,
    "analysis": {
      "summary": "Tóm tắt từ Gemini AI...",
      "indicators": {
        "fake_indicators": ["Dấu hiệu 1", "Dấu hiệu 2"],
        "real_indicators": []
      },
      "analysis": "Phân tích chi tiết...",
      "recommendations": "Khuyến nghị...",
      "confidence_explanation": "Giải thích độ tin cậy..."
    },
    "source": "gemini"
  }
}
```

### GET /health
Kiểm tra trạng thái API và model.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🤖 Mô hình AI

- **Kiến trúc**: TextClassificationModel với EmbeddingBag
- **Preprocessing**: Loại bỏ URL, HTML, emoji, chuẩn hóa văn bản
- **Tokenization**: Basic English tokenizer
- **Vocabulary**: 10,000 từ phổ biến nhất
- **Classes**: 
  - 0: Giả mạo (Fake)
  - 1: Bình thường (Real)

## 🎨 Giao diện

- **Material-UI**: Component library hiện đại
- **Responsive Design**: Tương thích mobile và desktop
- **Dark/Light Theme**: Tự động điều chỉnh theo hệ thống
- **Real-time Feedback**: Loading states và error handling

## 🔧 Phát triển

### Thêm tính năng mới

1. **Backend**: Thêm endpoints trong `backend/app.py`
2. **Frontend**: Tạo components trong `frontend/src/components/`

### Cải thiện mô hình

1. Chỉnh sửa `train/train.ipynb`
2. Retrain model
3. Cập nhật `backend/model/`

## 📊 Kết quả đánh giá

- **Accuracy**: ~85-90%
- **Precision**: Cao cho cả 2 class
- **Recall**: Cân bằng giữa fake và real news

## 🚨 Lưu ý

- Model được train trên dữ liệu tiếng Việt
- Kết quả chỉ mang tính tham khảo
- Luôn kiểm tra thông tin từ nhiều nguồn đáng tin cậy

## 📝 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

---

**Phát triển bởi**: [Tên của bạn]  
**Email**: [email@example.com]  
**GitHub**: [github.com/username]