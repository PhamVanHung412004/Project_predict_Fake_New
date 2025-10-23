import google.generativeai as genai
from config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        """Initialize Gemini service"""
        if not Config.validate_gemini_config():
            self.gemini_available = False
            logger.warning("Gemini API not configured. Analysis will be limited.")
            return
            
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.gemini_available = True
            logger.info("Gemini service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini service: {e}")
            self.gemini_available = False

    def analyze_prediction(self, text: str, prediction: int, confidence: float, label: str) -> dict:
        """
        Analyze the prediction result using Gemini AI
        
        Args:
            text: Original text that was analyzed
            prediction: Model prediction (0 or 1)
            confidence: Confidence score (0-100)
            label: Human-readable label ("Giả mạo" or "Bình thường")
        
        Returns:
            dict: Analysis result with explanation
        """
        if not self.gemini_available:
            return self._get_fallback_analysis(prediction, confidence, label)
        
        try:
            # Create prompt for Gemini
            prompt = self._create_analysis_prompt(text, prediction, confidence, label)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Parse response
            analysis = self._parse_gemini_response(response.text)
            
            return {
                'success': True,
                'analysis': analysis,
                'source': 'gemini'
            }
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return self._get_fallback_analysis(prediction, confidence, label)

    def _create_analysis_prompt(self, text: str, prediction: int, confidence: float, label: str) -> str:
        """Create analysis prompt for Gemini"""
        
        prediction_context = "tin giả" if prediction == 0 else "tin thật"
        confidence_level = "cao" if confidence >= 80 else "trung bình" if confidence >= 60 else "thấp"
        
        prompt = f"""
Bạn là một chuyên gia phân tích tin tức và phát hiện tin giả. Hãy phân tích văn bản sau dựa trên kết quả từ mô hình AI:

VĂN BẢN CẦN PHÂN TÍCH:
"{text}"

KẾT QUẢ MÔ HÌNH AI:
- Dự đoán: {label} ({prediction_context})
- Độ tin cậy: {confidence}% ({confidence_level})

Hãy cung cấp phân tích chi tiết theo định dạng JSON sau:

{{
    "summary": "Tóm tắt ngắn gọn về văn bản và kết quả phân tích",
    "indicators": {{
        "fake_indicators": ["danh sách các dấu hiệu cho thấy đây có thể là tin giả (nếu có)"],
        "real_indicators": ["danh sách các dấu hiệu cho thấy đây có thể là tin thật (nếu có)"]
    }},
    "analysis": "Phân tích chi tiết tại sao mô hình đưa ra kết quả này",
    "recommendations": "Khuyến nghị cho người đọc về cách xử lý thông tin này",
    "confidence_explanation": "Giải thích về độ tin cậy {confidence}% có nghĩa là gì"
}}

Lưu ý:
- Phân tích phải khách quan và dựa trên bằng chứng
- Nếu là tin giả, hãy chỉ ra các dấu hiệu cụ thể
- Nếu là tin thật, hãy giải thích các yếu tố đáng tin cậy
- Luôn khuyến nghị kiểm tra thông tin từ nhiều nguồn
- Trả lời bằng tiếng Việt
"""
        return prompt

    def _parse_gemini_response(self, response_text: str) -> dict:
        """Parse Gemini response and extract JSON"""
        try:
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                return {
                    "summary": "Phân tích từ Gemini AI",
                    "analysis": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                    "recommendations": "Hãy kiểm tra thông tin từ nhiều nguồn đáng tin cậy"
                }
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return {
                "summary": "Phân tích từ Gemini AI",
                "analysis": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "recommendations": "Hãy kiểm tra thông tin từ nhiều nguồn đáng tin cậy"
            }

    def _get_fallback_analysis(self, prediction: int, confidence: float, label: str) -> dict:
        """Get fallback analysis when Gemini is not available"""
        
        if prediction == 0:  # Fake news
            return {
                'success': False,
                'analysis': {
                    'summary': f'Mô hình AI đã phân tích và xác định đây có thể là {label} với độ tin cậy {confidence}%',
                    'indicators': {
                        'fake_indicators': [
                            'Ngôn ngữ cảm xúc mạnh mẽ',
                            'Thiếu nguồn thông tin cụ thể',
                            'Từ ngữ gây sốc hoặc kích động'
                        ],
                        'real_indicators': []
                    },
                    'analysis': f'Dựa trên phân tích của mô hình AI, văn bản này có {confidence}% khả năng là tin giả. Hãy cẩn thận và kiểm tra thông tin từ các nguồn đáng tin cậy.',
                    'recommendations': 'Khuyến nghị kiểm tra thông tin từ các nguồn chính thức và đáng tin cậy trước khi chia sẻ.',
                    'confidence_explanation': f'Độ tin cậy {confidence}% có nghĩa là mô hình AI khá chắc chắn về kết quả này.'
                },
                'source': 'fallback'
            }
        else:  # Real news
            return {
                'success': False,
                'analysis': {
                    'summary': f'Mô hình AI đã phân tích và xác định đây có thể là {label} với độ tin cậy {confidence}%',
                    'indicators': {
                        'fake_indicators': [],
                        'real_indicators': [
                            'Ngôn ngữ trung tính và khách quan',
                            'Có thể có nguồn thông tin rõ ràng',
                            'Từ ngữ chuyên nghiệp'
                        ]
                    },
                    'analysis': f'Dựa trên phân tích của mô hình AI, văn bản này có {confidence}% khả năng là tin thật. Tuy nhiên, hãy luôn kiểm tra thông tin từ nhiều nguồn.',
                    'recommendations': 'Mặc dù có vẻ đáng tin cậy, hãy luôn kiểm tra thông tin từ các nguồn chính thức.',
                    'confidence_explanation': f'Độ tin cậy {confidence}% có nghĩa là mô hình AI khá chắc chắn về kết quả này.'
                },
                'source': 'fallback'
            }

# Global instance
gemini_service = GeminiService()
