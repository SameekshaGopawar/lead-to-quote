"""
Text preprocessing for NLP input.
Cleans and normalises raw requirement text before encoding.
"""

import re


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "we", "our", "i", "my", "you", "your", "they", "their", "it", "its",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "need", "want", "looking", "also", "very", "that", "this",
    "which", "some", "all", "any", "each", "there", "than", "so", "as",
}


def clean_text(text: str) -> str:
    """
    Lowercase, remove punctuation, collapse whitespace, strip stopwords.
    Preserves domain terms (numbers, acronyms) that carry meaning.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [t for t in text.split() if t not in _STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def build_input(requirements: str, industry: str = "") -> str:
    """
    Combines requirements + industry into a single cleaned text for the model.
    Industry is appended so the model has sector context.
    """
    combined = f"{requirements} {industry}".strip()
    return clean_text(combined)
