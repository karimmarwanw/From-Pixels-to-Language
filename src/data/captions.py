"""Caption handling: split the `description` field into captions, tokenize, and clean them.

`description` holds 10 captions in one string, separated by "\\n", with a trailing
"\\n" (so a naive split gives 10 captions + 1 empty string).

Cleaning (clean_description) applies these steps, in this order; each one fixes a
problem measured in EDA step 6:
    1. split on "\\n", strip, drop empty strings     (captions are glued together)
    2. repair_encoding: U+FFFD runs -> one space      (27 captions had broken spaces)
    3. lowercase                                      (consistency; data is already lowercase)
    4. remove_punctuation: keep only words and numbers (inconsistent commas/periods)
    5. fix_typos: hand-checked word map              (frequent, clear misspellings only)
    6. drop captions with fewer than min_words words  (one cut-off caption: "this")
    7. drop exact repeats within the same image       (583 images repeated a caption)
The same function is later applied unchanged to val and test captions.
"""

import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import yaml

# A word = lowercase letters, optionally with one apostrophe part ("it's", "bird's").
# Everything else (spaces, punctuation, digits, broken characters) separates words.
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

# A token kept by cleaning = a number (decimals like 2.5 stay whole) or a word as above.
TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?")


def split_description(description: str) -> list[str]:
    """Split one description into its captions: split on "\\n", strip each line, drop empty ones."""
    return [line.strip() for line in description.split("\n") if line.strip()]


def tokenize(caption: str) -> list[str]:
    """Words of a caption (lowercased first), using WORD_RE. Used by the EDA analysis."""
    return WORD_RE.findall(caption.lower())


def caption_table(descriptions: list[str]) -> pd.DataFrame:
    """One row per (raw) caption.

    Returns:
        Columns image (row index of the image), caption, n_words, n_chars.
    """
    rows = [
        {"image": i, "caption": c, "n_words": len(tokenize(c)), "n_chars": len(c)}
        for i, d in enumerate(descriptions)
        for c in split_description(d)
    ]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- cleaning steps
def repair_encoding(caption: str) -> str:
    """Replace each run of U+FFFD replacement characters with one space ("this��bird" -> "this bird")."""
    return re.sub("�+", " ", caption)


def remove_punctuation(caption: str) -> str:
    """Keep only words and numbers, joined by single spaces.

    "&" becomes the word "and" and a backtick used as apostrophe becomes "'" first.
    Hyphens, slashes, commas, periods, brackets, quotes... all act as separators, so
    "spotted-white" -> "spotted white" and "grey,beek" -> "grey beek". Apostrophes inside
    words ("it's") and decimal numbers ("2.5") are kept. This also collapses extra spaces.
    """
    caption = caption.replace("&", " and ").replace("`", "'")
    return " ".join(TOKEN_RE.findall(caption))


def fix_typos(caption: str, typo_map: Mapping[str, str]) -> str:
    """Replace whole words found in typo_map (word-level, so "bil" is fixed but "bill" is untouched)."""
    return " ".join(typo_map.get(w, w) for w in caption.split())


def clean_caption(caption: str, typo_map: Mapping[str, str]) -> str:
    """Steps 2-5 for one caption: repair encoding, lowercase, remove punctuation, fix typos."""
    caption = repair_encoding(caption)
    caption = caption.lower()
    caption = remove_punctuation(caption)
    return fix_typos(caption, typo_map)


def clean_description(description: str, typo_map: Mapping[str, str], min_words: int) -> list[str]:
    """All cleaning steps for one image: split, clean each caption, drop too-short ones and repeats.

    Repeats are detected AFTER cleaning, so captions differing only in punctuation count as
    the same. The first occurrence is kept, the original order is preserved.
    """
    cleaned = [clean_caption(c, typo_map) for c in split_description(description)]
    seen: set[str] = set()
    result = []
    for c in cleaned:
        if len(c.split()) >= min_words and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def load_typo_map(path: str | Path) -> dict[str, str]:
    """Read the hand-checked typo map (the `typos:` mapping of a YAML file)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["typos"]
