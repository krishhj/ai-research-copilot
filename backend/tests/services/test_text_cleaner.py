from app.services.text_cleaner import TextCleaner

def test_clean_normalizes_whitespace():
    cleaner = TextCleaner()

    cleaned_text = cleaner.clean(
        "  Transformer\tmodels  \n\n\n\n  scale effectively.  "
    )

    assert cleaned_text == "Transformer models\n\nscale effectively."

def test_clean_normalizes_unicode_characters():
    cleaner = TextCleaner()

    cleaned_text = cleaner.clean("ef\ufb01cient attention")

    assert cleaned_text == "efficient attention"

def test_clean_empty_text():
    cleaner = TextCleaner()
    assert cleaner.clean("   \n\t  ") == ""