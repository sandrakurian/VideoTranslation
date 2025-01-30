# Video Dubbing and Translation Tool

## Overview
This Python application automates the process of dubbing videos into a different language by extracting audio, transcribing, translating, and regenerating synchronized audio.

## Workflow
1. Extract original audio from video
2. Create a muted version of the original video
3. Transcribe audio to original language subtitles
4. Translate subtitles to target language
5. Generate new audio using TTS for translated text
6. Combine muted video with new audio track

## Main Functions
`split_audio(video, video_blank, audio)`
Extracts audio and creates a muted video.

`transcribe_translate(audio, srt_orig, srt_trans, lang_input, lang_output)`
Transcribes the audio and translates the subtitles.

`generate_audio(srt_output, audio_output, lang_output)`
Generates text-to-speech (TTS) audio from the translated subtitles.

`dub_video(video_output, video_no_audio, audio_output)`
Merges the generated audio with the muted video to create a final dubbed video.

## Models Used
This script leverages several machine learning models for transcription, translation, and text-to-speech synthesis:

Whisper (Speech-to-Text): Transcribes speech from audio files and generates word timestamps.

MarianMT (Translation): Translates subtitles from the source language to the target language.
- `Helsinki-NLP/opus-mt-{source_lang}-{target_lang}`

TTS (Text-to-Speech): Converts translated text into speech and aligns it with timestamps.
- `tts_models/en/ljspeech/tacotron2-DDC` (for English)
- or `tts_models/{lang_output}/css10/vits` (for other languages)

## Current Limitations
- Single speaker support
- Translation accuracy dependent on model
- Voice cloning not yet implemented

## Future Improvements
- Multi-speaker voice differentiation
- Enhanced translation accuracy
- Voice cloning capabilities

## Usage
```python
# Example configuration
video_input = "video.mp4"
lang_input = "en"   # Original language
lang_output = "es"  # Target language

# Main processing steps
split_audio(video_input, video_no_audio, audio_input)
transcribe_translate(audio_input, srt_input, srt_output, lang_input, lang_output)
generate_audio(srt_output, audio_output, lang_output)
dub_video(video_output, video_no_audio, audio_output)
```
