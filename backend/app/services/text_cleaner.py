import re
import unicodedata

class TextCleaner:
    """Normalize extracted PDF text while preserving paragraphs."""

    def clean(self, text: str) -> str:
        """Return normalized text suitable for chunking"""
        cleaned_text = unicodedata.normalize("NFKC", text)

        cleaned_text = cleaned_text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        cleaned_text = re.sub(r" *\n *", "\n", cleaned_text)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        return cleaned_text.strip()