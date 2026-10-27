"""Quick tests for caption splitting and cleaning (src/data/captions.py)."""

from src.data.captions import (
    clean_caption,
    clean_description,
    fix_typos,
    remove_punctuation,
    repair_encoding,
    split_description,
)

TYPOS = {"bil": "bill", "whtie": "white", "witha": "with a"}


def test_split_drops_trailing_empty_line():
    assert split_description("a red bird.\n  a small bird  \n") == ["a red bird.", "a small bird"]


def test_repair_encoding_turns_broken_spaces_into_one_space():
    assert repair_encoding("this��bird��has") == "this bird has"


def test_remove_punctuation_keeps_words_numbers_and_apostrophes():
    assert remove_punctuation("a black body, white belly.") == "a black body white belly"
    assert remove_punctuation("grey,beek is beige") == "grey beek is beige"           # missing space after comma
    assert remove_punctuation("spotted-white eyering") == "spotted white eyering"     # hyphen splits
    assert remove_punctuation("black & white") == "black and white"                  # & becomes a word
    assert remove_punctuation("it's eyes") == "it's eyes"                            # apostrophe kept
    assert remove_punctuation("it`s eyes") == "it's eyes"                            # backtick -> apostrophe
    assert remove_punctuation("about 2.5 times, 2 wing bars") == "about 2.5 times 2 wing bars"  # numbers kept


def test_fix_typos_is_word_level():
    assert fix_typos("a short bil", TYPOS) == "a short bill"
    assert fix_typos("a short bill", TYPOS) == "a short bill"                         # correct word untouched
    assert fix_typos("grey witha large beak", TYPOS) == "grey with a large beak"      # one word -> two words


def test_clean_caption_runs_all_steps():
    raw = "Stark��whtie eye, short bil."
    assert clean_caption(raw, TYPOS) == "stark white eye short bill"


def test_clean_description_drops_short_captions_and_repeats():
    description = (
        "this bird has a white belly.\n"
        "this\n"                                   # too short -> dropped
        "this bird has a white belly\n"            # same as the first after removing "." -> dropped
        "a small bird with a short bil.\n"
    )
    assert clean_description(description, TYPOS, min_words=2) == [
        "this bird has a white belly",
        "a small bird with a short bill",
    ]
