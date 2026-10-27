"""Caption handling: split the `description` field into captions, and tokenize them.

`description` holds 10 captions in one string, separated by "\\n", with a trailing
"\\n" (so a naive split gives 10 captions + 1 empty string).
"""

import re

import pandas as pd

# A word = lowercase letters, optionally with one apostrophe part ("it's", "bird's").
# Everything else (spaces, punctuation, digits, broken characters) separates words.
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def split_description(description: str) -> list[str]:
    """Split one description into its captions: split on "\\n", strip each line, drop empty ones."""
    return [line.strip() for line in description.split("\n") if line.strip()]


def tokenize(caption: str) -> list[str]:
    """Words of a caption (lowercased first), using WORD_RE."""
    return WORD_RE.findall(caption.lower())


def caption_table(descriptions: list[str]) -> pd.DataFrame:
    """One row per caption.

    Returns:
        Columns image (row index of the image), caption, n_words, n_chars.
    """
    rows = [
        {"image": i, "caption": c, "n_words": len(tokenize(c)), "n_chars": len(c)}
        for i, d in enumerate(descriptions)
        for c in split_description(d)
    ]
    return pd.DataFrame(rows)
