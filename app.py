from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rag.chain import ask_gale_bot
from dotenv import load_dotenv
import os
import requests
import json
import base64
from io import BytesIO
import logging

load_dotenv()

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Microsoft Azure Translator Text API configuration (Free tier available)
AZURE_TRANSLATOR_KEY = os.getenv('AZURE_TRANSLATOR_KEY', '')
AZURE_TRANSLATOR_ENDPOINT = os.getenv('AZURE_TRANSLATOR_ENDPOINT', 'https://api.cognitive.microsofttranslator.com')
AZURE_TRANSLATOR_REGION = os.getenv('AZURE_TRANSLATOR_REGION', 'eastus')

# Supported languages with proper language codes
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'voice': 'en-US-JennyNeural'},
    'es': {'name': 'Spanish', 'voice': 'es-ES-ElviraNeural'},
    'fr': {'name': 'French', 'voice': 'fr-FR-DeniseNeural'},
    'de': {'name': 'German', 'voice': 'de-DE-KatjaNeural'},
    'hi': {'name': 'Hindi', 'voice': 'hi-IN-SwaraNeural'},
    'zh': {'name': 'Chinese', 'voice': 'zh-CN-XiaoxiaoNeural'},
    'ja': {'name': 'Japanese', 'voice': 'ja-JP-NanamiNeural'},
    'ko': {'name': 'Korean', 'voice': 'ko-KR-SunHiNeural'},
    'ar': {'name': 'Arabic', 'voice': 'ar-SA-ZariyahNeural'},
    'ru': {'name': 'Russian', 'voice': 'ru-RU-SvetlanaNeural'},
    'pt': {'name': 'Portuguese', 'voice': 'pt-BR-FranciscaNeural'},
    'it': {'name': 'Italian', 'voice': 'it-IT-ElsaNeural'},
    'bn': {'name': 'Bengali', 'voice': 'bn-IN-TanishaaNeural'},
    'te': {'name': 'Telugu', 'voice': 'te-IN-MohanNeural'},
    'ta': {'name': 'Tamil', 'voice': 'ta-IN-PallaviNeural'},
    'gu': {'name': 'Gujarati', 'voice': 'gu-IN-DhwaniNeural'},
    'mr': {'name': 'Marathi', 'voice': 'mr-IN-AarohiNeural'},
    'kn': {'name': 'Kannada', 'voice': 'kn-IN-SapnaNeural'},
    'ml': {'name': 'Malayalam', 'voice': 'ml-IN-SobhanaNeural'}
}

# Fallback to Google Translate if Azure isn't configured
def translate_text_fallback(text, target_lang='en'):
    """Fallback translation using a simple API"""
    try:
        # Simple translation using MyMemory API (free)
        if target_lang == 'en':
            return text
        
        url = f"https://api.mymemory.translated.net/get"
        params = {
            'q': text,
            'langpair': f'en|{target_lang}',
            'de': 'your-email@example.com'  # Replace with your email for higher limits
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            translated = data.get('responseData', {}).get('translatedText', text)
            return translated
        return text
    except Exception as e:
        logger.error(f"Fallback translation error: {str(e)}")
        return text

def translate_text_azure(text, target_lang='en'):
    """Translate text using Azure Translator"""
    try:
        if not AZURE_TRANSLATOR_KEY or target_lang == 'en':
            return text
        
        # Construct the request
        url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate"
        
        params = {
            'api-version': '3.0',
            'to': target_lang
        }
        
        headers = {
            'Ocp-Apim-Subscription-Key': AZURE_TRANSLATOR_KEY,
            'Ocp-Apim-Subscription-Region': AZURE_TRANSLATOR_REGION,
            'Content-Type': 'application/json'
        }
        
        body = [{'text': text}]
        
        response = requests.post(url, params=params, headers=headers, json=body)
        
        if response.status_code == 200:
            translations = response.json()
            if translations and len(translations) > 0:
                translated_text = translations[0]['translations'][0]['text']
                return translated_text
        
        # If Azure fails, use fallback
        return translate_text_fallback(text, target_lang)
        
    except Exception as e:
        logger.error(f"Azure translation error: {str(e)}")
        return translate_text_fallback(text, target_lang)

def text_to_speech_azure(text, language_code='en'):
    """Convert text to speech using Azure Cognitive Services"""
    try:
        if not AZURE_TRANSLATOR_KEY:
            # Fallback to gTTS
            from gtts import gTTS
            import tempfile
            
            tts = gTTS(text=text, lang=language_code[:2], slow=False)
            
            # Save to bytes buffer
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            
            return base64.b64encode(audio_bytes.read()).decode('utf-8')
        
        # Azure TTS endpoint
        url = f"https://{AZURE_TRANSLATOR_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
        
        headers = {
            'Ocp-Apim-Subscription-Key': AZURE_TRANSLATOR_KEY,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3'
        }
        
        # Get voice for the language
        voice = SUPPORTED_LANGUAGES.get(language_code, {}).get('voice', 'en-US-JennyNeural')
        
        # SSML with neural voice
        ssml = f"""
        <speak version='1.0' xml:lang='{language_code}'>
            <voice name='{voice}'>
                {text}
            </voice>
        </speak>
        """
        
        response = requests.post(url, headers=headers, data=ssml.encode('utf-8'))
        
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
        else:
            # Fallback to gTTS
            from gtts import gTTS
            
            tts = gTTS(text=text, lang=language_code[:2], slow=False)
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            
            return base64.b64encode(audio_bytes.read()).decode('utf-8')
            
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        
        # Ultimate fallback - return empty audio
        try:
            from gtts import gTTS
            tts = gTTS(text=text[:500] if len(text) > 500 else text, 
                      lang=language_code[:2] if language_code[:2] in ['en', 'es', 'fr', 'de', 'hi'] else 'en', 
                      slow=False)
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return base64.b64encode(audio_bytes.read()).decode('utf-8')
        except:
            # Return a silent MP3 as last resort
            silent_mp3 = base64.b64encode(b'').decode('utf-8')
            return silent_mp3

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"answer": "Please enter a medical question."})

    try:
        answer = ask_gale_bot(question)
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        return jsonify({"answer": f"Error: {str(e)}"})

@app.route("/translate", methods=["POST"])
def translate_text():
    """Translate text to target language"""
    try:
        data = request.get_json()
        text = data.get("text", "")
        target_lang = data.get("lang", "en")
        
        logger.debug(f"Translation request: lang={target_lang}, text_length={len(text)}")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Get language code (ensure it's in our supported list)
        lang_code = target_lang if target_lang in SUPPORTED_LANGUAGES else 'en'
        
        # Translate the text
        translated_text = translate_text_azure(text, lang_code)
        
        logger.debug(f"Translation successful: {lang_code}")
        
        return jsonify({
            "original": text,
            "translated": translated_text,
            "language": lang_code,
            "language_name": SUPPORTED_LANGUAGES.get(lang_code, {}).get('name', 'English')
        })
        
    except Exception as e:
        logger.error(f"Translation endpoint error: {str(e)}")
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

@app.route("/text-to-speech", methods=["POST"])
def text_to_speech():
    """Convert text to speech in specified language"""
    try:
        data = request.get_json()
        text = data.get("text", "")
        lang = data.get("lang", "en")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Get language code
        lang_code = lang if lang in SUPPORTED_LANGUAGES else 'en'
        
        logger.debug(f"TTS request: lang={lang_code}, text_length={len(text)}")
        
        # Create TTS audio
        audio_base64 = text_to_speech_azure(text, lang_code)
        
        return jsonify({
            "audio": audio_base64,
            "lang": lang_code,
            "lang_name": SUPPORTED_LANGUAGES.get(lang_code, {}).get('name', 'English'),
            "success": True
        })
        
    except Exception as e:
        logger.error(f"TTS endpoint error: {str(e)}")
        return jsonify({"error": f"Speech synthesis failed: {str(e)}", "success": False}), 500

@app.route("/translate-and-speak", methods=["POST"])
def translate_and_speak():
    """Translate and convert to speech in one go"""
    try:
        data = request.get_json()
        text = data.get("text", "")
        target_lang = data.get("lang", "en")
        
        logger.debug(f"Translate and speak: lang={target_lang}, text_length={len(text)}")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Get language code
        lang_code = target_lang if target_lang in SUPPORTED_LANGUAGES else 'en'
        
        # Translate the text
        translated_text = translate_text_azure(text, lang_code)
        
        # Create TTS audio
        audio_base64 = text_to_speech_azure(translated_text, lang_code)
        
        return jsonify({
            "original": text,
            "translated": translated_text,
            "audio": audio_base64,
            "lang": lang_code,
            "lang_name": SUPPORTED_LANGUAGES.get(lang_code, {}).get('name', 'English'),
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Translate and speak error: {str(e)}")
        return jsonify({"error": f"Translation and speech failed: {str(e)}", "success": False}), 500

@app.route("/get-languages", methods=["GET"])
def get_languages():
    """Get list of supported languages"""
    languages = []
    for code, info in SUPPORTED_LANGUAGES.items():
        languages.append({
            "code": code,
            "name": info['name'],
            "voice": info['voice']
        })
    
    return jsonify({"languages": languages, "success": True})

@app.route("/get-supported-languages", methods=["GET"])
def get_supported_languages():
    """Simple endpoint to check available languages"""
    return jsonify({
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "language_details": SUPPORTED_LANGUAGES,
        "success": True
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "translation_service": "Azure Translator" if AZURE_TRANSLATOR_KEY else "Fallback (MyMemory)",
        "tts_service": "Azure TTS" if AZURE_TRANSLATOR_KEY else "gTTS",
        "supported_languages": len(SUPPORTED_LANGUAGES)
    })

if __name__ == "__main__":
    # Install required packages if not available
    required_packages = ['gtts', 'requests']
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)