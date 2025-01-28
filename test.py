import os
import subprocess
import whisper
import re
import torch
from TTS.api import TTS
from pydub import AudioSegment
from typing import Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from language import language_mapper  # Import the LanguageMapper

# --------- HELPER FUNCTIONS --------- #

def format_timestamp(seconds):
    """Format seconds into SRT-compatible timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millisecs:03}"

def translate_srt(input_file, output_file, target_lang):
    try:
        # Initialize the model and tokenizer
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Check language mapping
        nllb_target_lang = language_mapper.get_nllb_language(target_lang)
        if not nllb_target_lang:
            print(f"Warning: The language {target_lang} is not supported by NLLB.")
            return

        print(f"Using NLLB model with target language code: {nllb_target_lang}")
        
        with open(input_file, "r", encoding="utf-8") as src, open(output_file, "w", encoding="utf-8") as tgt:
            for line in src:
                if "-->" in line or line.strip().isdigit():
                    tgt.write(line)  # Write timestamps and indexes unchanged
                elif line.strip():
                    inputs = tokenizer(line.strip(), return_tensors="pt", max_length=512, truncation=True)

                    # Log the tokenized input
                    print(f"Input tokens: {inputs}")

                    # Set the forced BOS token for the target language
                    inputs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(f"<<{nllb_target_lang}>>")
                    print(f"Using forced BOS token for {nllb_target_lang}: {inputs['forced_bos_token_id']}")

                    outputs = model.generate(**inputs, max_length=512, num_beams=5, no_repeat_ngram_size=2)
                    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Log the translation to check if it's working
                    print(f"Translated text: {translated_text}")

                    tgt.write(translated_text + "\n\n")

        print(f"Translated SRT saved to {output_file}")

    except Exception as e:
        print(f"Error during translation: {e}")

def timestamp_to_ms(timestamp):
    """Convert SRT timestamp to milliseconds."""
    hours, minutes, seconds_ms = timestamp.replace(',', '.').split(':')
    seconds, ms = seconds_ms.split('.')
    return (
        int(hours) * 3600 * 1000
        + int(minutes) * 60 * 1000
        + int(seconds) * 1000
        + int(ms)
    )

# --------- MAIN FUNCTIONS --------- #

def transcribe_audio_with_sentence_timestamps(audio_path: str, output_srt_path: str):
    """Transcribe audio using Whisper and save it as an SRT file with sentence-level timestamps."""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)

        sentence_segments = []
        current_sentence = []
        current_start = None

        for segment in result["segments"]:
            text = segment["text"].strip()
            if not current_start:
                current_start = segment["start"]

            current_sentence.append(text)
            if text.rstrip()[-1] in ".!?":
                sentence_segments.append({"start": current_start, "end": segment["end"], "text": " ".join(current_sentence).strip()})
                current_sentence = []
                current_start = None

        with open(output_srt_path, "w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(sentence_segments, start=1):
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"]
                srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

        print(f"Transcription saved to {output_srt_path}")

    except Exception as e:
        print(f"Error during transcription: {e}"

def format_timestamp(seconds: float) -> str:
    """Format seconds into SRT-compatible timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millisecs:03}"

def transcribe_translate(audio, srt_orig, srt_trans, lang_output):
    """Transcribe audio and translate the transcription."""
    transcribe_audio_with_sentence_timestamps(audio, srt_orig)
    translate_srt(srt_orig, srt_trans, 

# --------- MAIN SCRIPT --------- #

if __name__ == "__main__":
    video_input = "results/video1/video_eng.mp4"
    results_dir = "results/video1/to_spanish"

    lang_input = "en"
    lang_output = "es"

    if not language_mapper.validate_language(lang_input) or not language_mapper.validate_language(lang_output):
        print(f"Error: Unsupported language(s). Please check {lang_input} and {lang_output}.")
        exit(1)
    tts_lang_output = language_mapper.get_tts_language(lang_output)
    nllb_lang_output = language_mapper.get_nllb_language(lang_output)
    nllb_lang_input = language_mapper.get_nllb_language(lang_input)

    os.makedirs(results_dir, exist_ok=True)

    video_no_audio = os.path.join(results_dir, "../video_no_audio.mp4")
    video_output = os.path.join(results_dir, f"video_{lang_output}.mp4")
    audio_input = os.path.join(results_dir, f"../audio_{lang_input}.mp3")
    audio_output = os.path.join(results_dir, f"audio_{lang_output}.wav")
    reference_audio = os.path.join(results_dir, "../reference_audio.wav")
    srt_input = os.path.join(results_dir, f"../transcript_{lang_input}.srt")
    srt_output = os.path.join(results_dir, f"transcript_{lang_output}.srt")

    translate_srt(srt_input, srt_output, target_lang=nllb_lang_output)