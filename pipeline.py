
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------------------------------------------------------
# Stage 1: Vision — OCR
# --------------------------------------------------------------------------

def extract_text_from_image(image: Image.Image) -> str:
    """Run OCR on a PIL image and return the raw extracted text.

    A light preprocessing pass (grayscale + autocontrast) is applied first;
    Tesseract is noticeably more accurate on normalized images than on raw
    photos with uneven lighting.
    """
    prepped = ImageOps.autocontrast(ImageOps.grayscale(image))
    raw = pytesseract.image_to_string(prepped)
    # Collapse stray whitespace/line-wrap artifacts from OCR into clean text.
    text = re.sub(r"[ \t]+", " ", raw)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    return text


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# --------------------------------------------------------------------------
# Stage 2: NLP — extractive summarization (frequency-weighted TextRank-lite)
# --------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so",
    "of", "in", "on", "at", "by", "for", "to", "with", "about", "as",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "it", "its", "this", "that", "these", "those", "there", "here",
    "he", "she", "they", "we", "you", "i", "his", "her", "their", "our",
    "your", "my", "them", "us", "him",
    "not", "no", "yes", "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "will", "would", "can", "could", "should", "may", "might", "must",
    "from", "into", "over", "under", "again", "further", "once",
    "up", "down", "out", "off", "just", "very", "also", "too",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class SummaryResult:
    sentences: list[str] = field(default_factory=list)
    summary: str = ""
    num_source_sentences: int = 0

    def __str__(self) -> str:
        return self.summary


def _split_sentences(text: str) -> list[str]:
    # Normalize newlines from OCR into spaces before splitting on sentence
    # punctuation, since OCR line-wraps don't correspond to sentence breaks.
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return []
    parts = _SENTENCE_SPLIT_RE.split(flat)
    return [p.strip() for p in parts if p.strip()]


def _tokenize(sentence: str) -> list[str]:
    return [
        w.strip(string.punctuation).lower()
        for w in sentence.split()
        if w.strip(string.punctuation)
    ]


def summarize(text: str, max_sentences: int = 3) -> SummaryResult:
    """Extractive summary using word-frequency sentence scoring.

    1. Split into sentences.
    2. Score each word by frequency (stopwords excluded).
    3. Score each sentence by its mean word score, with a small boost for
       sentences near the start (leading sentences tend to carry the topic).
    4. Take the top-N sentences, but re-emit them in original order so the
       summary still reads coherently.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return SummaryResult(sentences=[], summary="", num_source_sentences=0)

    if len(sentences) <= max_sentences:
        return SummaryResult(
            sentences=sentences,
            summary=" ".join(sentences),
            num_source_sentences=len(sentences),
        )

    word_freq: dict[str, int] = {}
    tokenized = [_tokenize(s) for s in sentences]
    for words in tokenized:
        for w in words:
            if w and w not in _STOPWORDS:
                word_freq[w] = word_freq.get(w, 0) + 1

    max_freq = max(word_freq.values()) if word_freq else 1

    scores = []
    for i, words in enumerate(tokenized):
        content_words = [w for w in words if w not in _STOPWORDS]
        if not content_words:
            scores.append(0.0)
            continue
        score = sum(word_freq.get(w, 0) / max_freq for w in content_words)
        score /= len(content_words)
        if i == 0:
            score *= 1.15  # slight lead bias
        scores.append(score)

    ranked_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    top_idx = sorted(ranked_idx[:max_sentences])
    chosen = [sentences[i] for i in top_idx]

    return SummaryResult(
        sentences=chosen,
        summary=" ".join(chosen),
        num_source_sentences=len(sentences),
    )


# --------------------------------------------------------------------------
# Stage 3: Translation
# --------------------------------------------------------------------------

class TranslationUnavailable(RuntimeError):
    """Raised when the translation backend can't be reached."""


def translate(text: str, target_lang: str = "es") -> str:
    """Translate text into `target_lang` (ISO 639-1 code, e.g. 'es', 'fr', 'ja').

    Uses deep-translator's free Google Translate backend. This requires
    outbound internet access at runtime; if it's unavailable, a clear
    exception is raised so callers can degrade gracefully instead of
    crashing on a stack trace.
    """
    if not text.strip():
        return ""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as exc:  # network error, bad lang code, etc.
        raise TranslationUnavailable(str(exc)) from exc


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------

@dataclass
class PipelineResult:
    raw_text: str
    summary: SummaryResult
    translated_summary: str | None
    target_lang: str | None
    translation_error: str | None = None


def run_pipeline(
    image: Image.Image,
    max_sentences: int = 3,
    target_lang: str | None = None,
) -> PipelineResult:
    raw_text = extract_text_from_image(image)
    summary = summarize(raw_text, max_sentences=max_sentences)

    translated_summary = None
    translation_error = None
    if target_lang:
        try:
            translated_summary = translate(summary.summary, target_lang)
        except TranslationUnavailable as exc:
            translation_error = str(exc)

    return PipelineResult(
        raw_text=raw_text,
        summary=summary,
        translated_summary=translated_summary,
        target_lang=target_lang,
        translation_error=translation_error,
    )
