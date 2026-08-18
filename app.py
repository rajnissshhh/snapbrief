"""
SnapBrief — Streamlit app
==========================
Upload a photo of text (a page, sign, whiteboard, screenshot...) and get:
  1. The text read out of the image (OCR / vision)
  2. A short extractive summary (NLP)
  3. That summary translated into a language of your choice (translation)

Run with:  streamlit run app.py
"""

import streamlit as st
from PIL import Image

from pipeline import run_pipeline, TranslationUnavailable

TARGET_LANG = "ne"  # Nepali

st.set_page_config(page_title="SnapBrief", page_icon="📸", layout="centered")

st.title("📸 SnapBrief")
st.caption(
    "Vision → NLP → Translation, chained into one pipeline. "
    "Upload an image with text; get a short summary translated into Nepali."
)

with st.sidebar:
    st.header("Settings")
    max_sentences = st.slider("Summary length (sentences)", 1, 6, 3)
    translate_on = st.checkbox("Translate summary to Nepali (नेपाली)", value=True)
    target_lang = TARGET_LANG if translate_on else None
    st.markdown("---")
    st.markdown(
        "**Pipeline stages**\n\n"
        "1. 👁️ **Vision** — Tesseract OCR reads text from the image\n"
        "2. 🧠 **NLP** — frequency-based extractive summarization\n"
        "3. 🌐 **Translation** — English → Nepali (via `deep-translator`)"
    )

uploaded = st.file_uploader(
    "Upload an image (photo of a page, sign, whiteboard, screenshot...)",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
)

use_sample = st.button("Or try the bundled sample image")

image_to_process = None
if uploaded is not None:
    image_to_process = Image.open(uploaded).convert("RGB")
elif use_sample:
    image_to_process = Image.open("sample_data/sample_notice.png").convert("RGB")

if image_to_process is not None:
    st.image(image_to_process, caption="Input image", use_container_width=True)

    with st.spinner("Running OCR → summarization → translation..."):
        result = run_pipeline(
            image_to_process,
            max_sentences=max_sentences,
            target_lang=target_lang,
        )

    st.subheader("1️⃣ Extracted text (vision)")
    if result.raw_text.strip():
        st.text_area("Raw OCR output", result.raw_text, height=150)
    else:
        st.warning("No text was detected in this image. Try a clearer, higher-contrast photo.")

    st.subheader("2️⃣ Summary (NLP)")
    if result.summary.summary:
        st.info(result.summary.summary)
        st.caption(
            f"Condensed {result.summary.num_source_sentences} sentence(s) "
            f"down to {len(result.summary.sentences)}."
        )
    else:
        st.write("Nothing to summarize.")

    if target_lang:
        st.subheader("3️⃣ Translation (Nepali)")
        if result.translated_summary:
            st.success(result.translated_summary)
        elif result.translation_error:
            st.error(
                "Translation service unavailable right now "
                f"(network issue): {result.translation_error}"
            )
else:
    st.info("Upload an image above, or click the sample button, to run the pipeline.")
