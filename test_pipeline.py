"""
Quick sanity tests for the pipeline stages. Run with:  python -m pytest test_pipeline.py -v
(or just: python test_pipeline.py)
"""

from pipeline import summarize, load_image, extract_text_from_image, run_pipeline

LONG_TEXT = (
    "The library opens at nine in the morning on weekdays. "
    "Students often arrive early to claim quiet study rooms. "
    "The second floor houses the largest collection of history books. "
    "Group study spaces are available on the third floor with advance booking. "
    "Printing costs ten cents per page for black and white documents. "
    "Color printing is available at the main desk for a higher fee. "
    "The library closes at eight in the evening except on Fridays."
)


def test_summarize_shrinks_long_text():
    result = summarize(LONG_TEXT, max_sentences=3)
    assert result.num_source_sentences == 7
    assert len(result.sentences) == 3
    assert result.summary  # non-empty
    print("OK: summarize shrinks a 7-sentence text to 3")


def test_summarize_short_text_passthrough():
    short = "Only one sentence here."
    result = summarize(short, max_sentences=3)
    assert result.summary == short
    assert len(result.sentences) == 1
    print("OK: short text passes through unchanged")


def test_summarize_empty_text():
    result = summarize("", max_sentences=3)
    assert result.summary == ""
    assert result.sentences == []
    print("OK: empty text handled without error")


def test_ocr_on_sample_image():
    img = load_image("sample_data/sample_notice.png")
    text = extract_text_from_image(img)
    assert "intelligence" in text.lower()
    print("OK: OCR extracts recognizable words from sample image")


def test_full_pipeline_without_translation():
    img = load_image("sample_data/sample_notice.png")
    result = run_pipeline(img, max_sentences=2, target_lang=None)
    assert result.raw_text
    assert result.summary.summary
    assert result.translated_summary is None
    assert result.translation_error is None
    print("OK: full pipeline runs end-to-end (vision -> NLP)")


if __name__ == "__main__":
    test_summarize_shrinks_long_text()
    test_summarize_short_text_passthrough()
    test_summarize_empty_text()
    test_ocr_on_sample_image()
    test_full_pipeline_without_translation()
    print("\nAll tests passed.")
