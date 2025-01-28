# language.py

class LanguageMapper:
    """map language codes for different models to ensure consistency across models"""

    def __init__(self):
        # Define mappings for Facebook NLLB model
        self.nllb_languages = {
            "en": "eng_Latn",  # English
            "es": "spa",  # Spanish
            "fr": "fra_Latn",  # French
            "de": "deu_Latn",  # German
            "zh": "zho_Hans",  # Simplified Chinese
            "hi": "hin_Deva",  # Hindi
            "it": "ita_Latn",  # Italian
            "pt": "por_Latn",  # Portuguese
            "nl": "nld_Latn",  # Dutch
            "pl": "pol_Latn",  # Polish
            "ru": "rus_Cyrl",  # Russian
            "uk": "ukr_Cyrl",  # Ukrainian
            "ar": "arb_Arab",  # Arabic
            "bn": "ben_Beng",  # Bengali
            "ja": "jpn_Jpan",  # Japanese
            "ko": "kor_Hang",  # Korean
            "vi": "vie_Latn",  # Vietnamese
            "tr": "tur_Latn",  # Turkish
            "th": "tha_Thai"   # Thai
        }

        # Define mappings for the XTTS-v2 model
        self.tts_languages = {
            "en": "en",  # English
            "es": "es",  # Spanish
            "fr": "fr",  # French
            "de": "de",  # German
            "zh": "zh",  # Simplified Chinese
            "hi": "hi",  # Hindi
            "it": "it",  # Italian
            "pt": "pt",  # Portuguese
            "nl": "nl",  # Dutch
            "pl": "pl",  # Polish
            "ru": "ru",  # Russian
            "uk": "uk",  # Ukrainian
            "ar": "ar",  # Arabic
            "bn": "bn",  # Bengali
            "ja": "ja",  # Japanese
            "ko": "ko",  # Korean
            "vi": "vi",  # Vietnamese
            "tr": "tr",  # Turkish
            "th": "th"   # Thai
        }

    def get_nllb_language(self, lang_code):
        return self.nllb_languages.get(lang_code)

    def get_tts_language(self, lang_code):
        return self.tts_languages.get(lang_code)

    def validate_language(self, lang_code):
        return lang_code in self.nllb_languages

language_mapper = LanguageMapper()
