# SnapBrief

Upload a photo of text — a page, a sign, a whiteboard, a screenshot — and get
a short summary, translated into Nepali. Three AI capabilities chained into
one pipeline:

```
   Image                Raw text              Summary            Summary
     │      OCR             │      extractive      │   translate     │
     ├─── (vision) ────────>├──── summarize ───────>├──── (NLP) ────>│
     │                      │      (NLP)            │   → Nepali     │
```

1. **Vision** — Tesseract OCR reads the text out of the image (with a
   grayscale + autocontrast preprocessing pass for better accuracy on real
   photos).
2. **NLP** — a frequency-weighted extractive summarizer scores each sentence
   by how central its words are to the document, then keeps the top N
   sentences in their original order. No external model download required.
3. **Translation** — the summary is translated into Nepali (नेपाली) via
   Google Translate (through `deep-translator`). Can be toggled off in the
   sidebar if you just want the English summary.

## Project layout

```
snapbrief/
├── pipeline.py        # the three pipeline stages, as pure functions
├── app.py              # Streamlit UI
├── test_pipeline.py     # sanity tests for each stage
├── sample_data/
│   └── sample_notice.png
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

You'll also need the Tesseract OCR binary on your system:

```bash
# Debian/Ubuntu
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

## Run the web app

```bash
streamlit run app.py
```

Upload an image (or click the sample-image button), pick a summary length
in the sidebar, and the app runs all three stages live — OCR, summarize,
then translate the summary into Nepali (toggle-able).

## Run the tests

```bash
python test_pipeline.py
```

## Design notes

- **Graceful degradation**: if translation is requested but the network/API
  is unavailable, the pipeline catches that specific failure
  (`TranslationUnavailable`) and reports it cleanly instead of crashing —
  both the CLI and the Streamlit app surface it as a readable message.
- **No heavyweight model downloads**: the summarizer is a from-scratch
  frequency-scoring extractive method (TextRank-lite), so there's nothing
  to download and no GPU dependency — it runs anywhere Python runs.
- **Shared core**: `pipeline.py` has zero UI code in it, so the pipeline
  logic stays fully testable independent of Streamlit.
