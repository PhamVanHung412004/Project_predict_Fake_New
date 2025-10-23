import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    
    # Flask Configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    @staticmethod
    def validate_gemini_config():
        """Validate Gemini API configuration"""
        if not Config.GEMINI_API_KEY:
            print("⚠️  Warning: GEMINI_API_KEY not found in environment variables")
            print("   Please set GEMINI_API_KEY in your .env file or environment")
            return False
        return True
