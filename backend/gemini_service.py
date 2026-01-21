import google.generativeai as genai
from config import Config
import logging
from googletrans import Translator

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
            self.translator = Translator()
            self.gemini_available = True
            logger.info("Gemini service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini service: {e}")
            self.gemini_available = False

    def _translate_text(self, text: str, target_lang: str = 'en') -> tuple[str, str]:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_lang: Target language code ('en' for English, 'vi' for Vietnamese)
            
        Returns:
            tuple: (translated_text, source_language)
        """
        try:
            translation = self.translator.translate(text, dest=target_lang)
            return translation.text, translation.src
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text, 'unknown'

    def analyze_prediction(self, text: str, prediction: int, confidence: float, label: str) -> dict:
        """
        Analyze the prediction result using Gemini AI
        
        Args:
            text: Original text that was analyzed
            prediction: Model prediction (0 or 1)
            confidence: Confidence score (0-100)
            label: Human-readable label ("Fake" or "Real")
        
        Returns:
            dict: Analysis result with explanation
        """
        if not self.gemini_available:
            return self._get_fallback_analysis(prediction, confidence, label)
        
        try:
            # Translate text to English if it's in Vietnamese
            translated_text, source_lang = self._translate_text(text)
            
            # Create prompt for Gemini
            prompt = self._create_analysis_prompt(translated_text, prediction, confidence, label)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Parse response
            analysis = self._parse_gemini_response(response.text)
            
            # If original text was in Vietnamese, translate the analysis back
            if source_lang == 'vi':
                analysis = self._translate_analysis_to_vietnamese(analysis)
            
            return {
                'success': True,
                'analysis': analysis,
                'source': 'gemini',
                'translation_info': {
                    'was_translated': source_lang == 'vi',
                    'original_language': source_lang
                }
            }
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return self._get_fallback_analysis(prediction, confidence, label)

    def _translate_analysis_to_vietnamese(self, analysis: dict) -> dict:
        """
        Translate analysis results back to Vietnamese
        """
        try:
            # Translate main fields
            for field in ['summary', 'analysis', 'recommendations', 'confidence_explanation']:
                if field in analysis:
                    analysis[field], _ = self._translate_text(analysis[field], 'vi')
            
            # Translate indicators
            if 'indicators' in analysis:
                if 'fake_indicators' in analysis['indicators']:
                    analysis['indicators']['fake_indicators'] = [
                        self._translate_text(indicator, 'vi')[0]
                        for indicator in analysis['indicators']['fake_indicators']
                    ]
                if 'real_indicators' in analysis['indicators']:
                    analysis['indicators']['real_indicators'] = [
                        self._translate_text(indicator, 'vi')[0]
                        for indicator in analysis['indicators']['real_indicators']
                    ]
            
            return analysis
        except Exception as e:
            logger.error(f"Translation back to Vietnamese failed: {e}")
            return analysis

    def _create_analysis_prompt(self, text: str, prediction: int, confidence: float, label: str) -> str:
        """Create analysis prompt for Gemini"""
        
        prediction_context = "fake news" if prediction == 0 else "real news"
        confidence_level = "high" if confidence >= 80 else "medium" if confidence >= 60 else "low"
        
        prompt = f"""
You are a news analysis expert specializing in fake news detection. Please analyze the following text based on AI model results:

TEXT TO ANALYZE:
"{text}"

AI MODEL RESULTS:
- Prediction: {label} ({prediction_context})
- Confidence: {confidence}% ({confidence_level})

Please provide a detailed analysis in the following JSON format:

{{
    "summary": "Brief summary of the text and analysis results",
    "indicators": {{
        "fake_indicators": ["list of indicators suggesting this might be fake news (if any)"],
        "real_indicators": ["list of indicators suggesting this might be real news (if any)"]
    }},
    "analysis": "Detailed analysis of why the model made this prediction",
    "recommendations": "Recommendations for readers on how to handle this information",
    "confidence_explanation": "Explanation of what the {confidence}% confidence means"
}}

Notes:
- Analysis must be objective and evidence-based
- If fake news, point out specific indicators
- If real news, explain credibility factors
- Always recommend checking information from multiple sources
- Respond in English
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
                    "summary": "Analysis from Gemini AI",
                    "analysis": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                    "recommendations": "Please verify information from multiple reliable sources"
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
                    'summary': f'AI model has analyzed and determined this might be {label} with {confidence}% confidence',
                    'indicators': {
                        'fake_indicators': [
                            'Strong emotional language',
                            'Lack of specific sources',
                            'Shocking or provocative wording'
                        ],
                        'real_indicators': []
                    },
                    'analysis': f'Based on AI model analysis, this text has a {confidence}% chance of being fake news. Please be cautious and verify information from reliable sources.',
                    'recommendations': 'Recommend checking information from official and reliable sources before sharing.',
                    'confidence_explanation': f'A confidence of {confidence}% means the AI model is quite certain about this result.'
                },
                'source': 'fallback'
            }
        else:  # Real news
            return {
                'success': False,
                'analysis': {
                    'summary': f'AI model has analyzed and determined this might be {label} with {confidence}% confidence',
                    'indicators': {
                        'fake_indicators': [],
                        'real_indicators': [
                            'Neutral and objective language',
                            'Clear information sources',
                            'Professional wording'
                        ]
                    },
                    'analysis': f'Based on AI model analysis, this text has a {confidence}% chance of being real news. However, always verify information from multiple sources.',
                    'recommendations': 'Although it appears reliable, always verify information from official sources.',
                    'confidence_explanation': f'A confidence of {confidence}% means the AI model is quite certain about this result.'
                },
                'source': 'fallback'
            }

# Global instance
gemini_service = GeminiService()
