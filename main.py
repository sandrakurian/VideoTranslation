import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os

def audio_to_text(file_path, language="en-US"):
    """
    Converts audio to text using the SpeechRecognition library.
    
    Args:
        file_path (str): Path to the audio file.
        language (str): Language of the audio file. Default is "en-US".
    
    Returns:
        str: Transcribed text from the audio.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            print("Processing audio...")
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language=language)
            return text
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return None

def translate_text(text, src_lang="auto", target_lang="en"):
    """
    Translates text using Google Translate.
    
    Args:
        text (str): Text to be translated.
        src_lang (str): Source language (default: 'auto').
        target_lang (str): Target language.
    
    Returns:
        str: Translated text.
    """
    translator = Translator()
    try:
        translated = translator.translate(text, src=src_lang, dest=target_lang)
        return translated.text
    except Exception as e:
        print(f"Error in translation: {e}")
        return None

def text_to_audio(text, output_file, language="en"):
    """
    Converts text to an audio file using gTTS.
    
    Args:
        text (str): Text to convert to speech.
        output_file (str): Path to save the output audio file.
        language (str): Language of the speech.
    """
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_file)
        print(f"Audio file saved as {output_file}")
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")

if __name__ == "__main__":
    # Input audio file path
    audio_file = "your_audio_file.wav"  # Replace with your file path
    source_language = "en-US"  # Language of the audio
    target_language = "es"  # Target translation language (e.g., Spanish)
    target_tts_language = "es"  # Language code for text-to-speech

    # Step 1: Convert audio to text
    transcribed_text = audio_to_text(audio_file, language=source_language)
    if transcribed_text:
        print(f"Transcribed Text: {transcribed_text}")
        
        # Step 2: Translate text
        translated_text = translate_text(transcribed_text, target_lang=target_language)
        if translated_text:
            print(f"Translated Text: {translated_text}")
            
            # Step 3: Convert translated text to audio
            output_audio_file = "translated_audio.mp3"  # Output file name
            text_to_audio(translated_text, output_file=output_audio_file, language=target_tts_language)
