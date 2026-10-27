"""Task 1.1 EDA, one function per step: python -m src.cli eda --config configs/eda.yaml --step N

These functions are the CLI's reporting layer: they call the reusable logic in
src/data/ and src/visualization/ and print what they find, so printing is fine here.
All steps use the official TRAIN set only; the test set is never loaded.
"""

import random
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.captions import caption_table, tokenize
from src.data.class_stats import class_counts, imbalance_summary
from src.data.duplicates import (
    DUPLICATE_TYPES,
    duplicate_type,
    exact_duplicate_groups,
    find_near_duplicates,
    load_or_build_hashes,
    nearest_distances,
)
from src.data.image_stats import load_or_build_image_stats
from src.data.loading import get_label_names, load_cub_split
from src.utils.config import EDAConfig
from src.visualization.eda import (
    plot_caption_lengths,
    plot_class_distribution,
    plot_color_modes,
    plot_image_grid,
    plot_image_pairs,
    plot_image_rows,
    plot_image_sizes,
    plot_nearest_distance,
    plot_records,
    plot_text_rows,
    plot_word_bars,
    plot_word_frequency,
)


def _header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def step1_inspect(cfg: EDAConfig) -> None:
    """Load the official train set, print its schema and a few full records, and show their images."""
    train = load_cub_split(cfg.train_name, "train", cfg.cache_dir)
    names = get_label_names(train)

    _header("STEP 1 - Load & inspect (official train set)")
    print(f"Rows:    {len(train)}")
    print(f"Columns: {train.column_names}")
    print("\nSchema:")
    for col, feature in train.features.items():
        kind = type(feature).__name__
        detail = f"{feature.num_classes} classes" if kind == "ClassLabel" else str(feature)
        print(f"  {col:<12} {kind:<11} {detail}")
    print(f"\nClass names: {len(names)}  (first 3: {names[:3]}, last: {names[-1]!r})")

    idx = sorted(random.Random(cfg.seed).sample(range(len(train)), cfg.n_inspect))
    images, titles = [], []
    for i in idx:
        rec = train[i]
        img = rec["image"]
        _header(f"Record {i}")
        print(f"id:          {rec['id']}")
        print(f"file_name:   {rec['file_name']}")
        print(f"label:       {rec['label']} -> {names[rec['label']]}")
        print(f"image:       {img.width}x{img.height} px (W x H), mode {img.mode}, format {img.format}")
        print(f"description: {len(rec['description'])} characters, {rec['description'].count(chr(10))} newline(s)")
        print("raw (repr, so the \\n separators are visible):")
        print(f"  {rec['description']!r}")
        images.append(img)
        titles.append(f"{names[rec['label']]} (label {rec['label']})\n{img.width}x{img.height}, {img.mode}\nrow {i}")

    out = plot_records(images, titles, "Step 1 - three training records",
                       Path(cfg.figures_dir) / "step01_records.png")
    _header("Figure")
    print(f"  {out}")


def step2_class_distribution(cfg: EDAConfig) -> None:
    """Images per class in the official train set: summary, extremes, and two plots."""
    train = load_cub_split(cfg.train_name, "train", cfg.cache_dir)
    counts = class_counts(train["label"], get_label_names(train))  # label column only, no images decoded
    s = imbalance_summary(counts)

    _header("STEP 2 - Class distribution (official train set)")
    print(f"Classes: {s['classes']}   Images: {s['images']}")
    print(f"Images per class -> min {s['min']} | max {s['max']} | mean {s['mean']:.2f} | "
          f"median {s['median']:.0f}")
    print(f"Imbalance ratio (max / min): {s['imbalance_ratio']:.3f}")

    _header("How many classes have N images")
    for n, k in counts["count"].value_counts().sort_index().items():
        print(f"  {n} images: {k:>3} classes")

    smallest = counts[counts["count"] == s["min"]]
    smallest = smallest if len(smallest) > 5 else counts.tail(5)  # show every class tied at the minimum
    _header(f"{len(smallest)} smallest classes")
    print(smallest.sort_values("label").to_string(index=False))
    _header("5 largest classes")
    n_max = (counts["count"] == s["max"]).sum()
    print(f"({n_max} classes are tied at the maximum of {s['max']}; showing the first 5 by label)")
    print(counts.head(5).to_string(index=False))

    out = plot_class_distribution(counts, Path(cfg.figures_dir) / "step02_class_distribution.png")
    _header("Figure")
    print(f"  {out}")


def step3_image_properties(cfg: EDAConfig) -> None:
    """Sizes, aspect ratios, color modes, corrupted / small / gray / dark / bright images."""
    train = load_cub_split(cfg.train_name, "train", cfg.cache_dir)
    names = get_label_names(train)
    df = load_or_build_image_stats(
        train, names, Path(cfg.stats_dir) / "image_stats.csv", cfg.gray_tolerance, cfg.rebuild_stats
    )
    ok = df[df["ok"]]
    cols = ["index", "class_name", "width", "height", "mode"]

    _header("STEP 3 - Image properties & quality (official train set)")
    print(f"Images scanned: {len(df)}   decoded OK: {len(ok)}   formats: {df['format'].value_counts().to_dict()}")
    print("\nSummary (px, bytes, brightness 0-255):")
    summary = ok[["width", "height", "aspect", "short_side", "n_bytes", "mean_brightness"]].describe()
    print(summary.loc[["min", "25%", "50%", "mean", "75%", "max"]].round(2).to_string())
    longest_500 = ((ok["width"] == 500) | (ok["height"] == 500)).mean()
    print(f"\nImages whose longest side is 500 px: {longest_500:.1%}")

    _header("Corrupted files (fail to decode)")
    bad = df[~df["ok"]]
    print(f"Count: {len(bad)}")
    if len(bad):
        print(bad[["index", "file_name", "error"]].to_string(index=False))

    _header("Color modes")
    print(df["mode"].value_counts().to_string())
    gray = ok[ok["gray_content"].astype(bool)]
    print(f"\nGrayscale images ({len(gray)}): stored as 'L', or stored as RGB but with no color "
          f"(every pixel has |R-G|,|G-B|,|R-B| <= {cfg.gray_tolerance}):")
    print(gray[cols + ["max_channel_diff"]].to_string(index=False))
    near = ok[(~ok["gray_content"].astype(bool)) & (ok["max_channel_diff"] <= 20)]
    print(f"\nNearly gray (channel difference {cfg.gray_tolerance + 1}-20), kept as color: {len(near)}")
    if len(near):
        print(near[cols + ["max_channel_diff"]].to_string(index=False))

    _header(f"Small images (shorter side < {cfg.small_side_px} px, will be upscaled)")
    small = ok[ok["short_side"] < cfg.small_side_px].sort_values("short_side")
    print(f"Count: {len(small)} ({len(small) / len(ok):.1%})")
    print(small[cols + ["short_side"]].head(10).to_string(index=False))

    wide = ok[ok["aspect"] > cfg.wide_aspect].sort_values("aspect", ascending=False)
    tall = ok[ok["aspect"] < cfg.tall_aspect].sort_values("aspect")
    dark = ok[ok["mean_brightness"] < cfg.dark_brightness].sort_values("mean_brightness")
    bright = ok[ok["mean_brightness"] > cfg.bright_brightness].sort_values("mean_brightness", ascending=False)

    _header(f"Extreme aspect ratios (wide: > {cfg.wide_aspect}, tall: < {cfg.tall_aspect})")
    print(f"Very wide: {len(wide)}");  print(wide[cols + ["aspect"]].head(5).round(2).to_string(index=False))
    print(f"\nVery tall: {len(tall)}"); print(tall[cols + ["aspect"]].head(5).round(2).to_string(index=False))

    _header(f"Very dark (< {cfg.dark_brightness}) and very bright (> {cfg.bright_brightness}); "
            "mean pixel value, 0 = black, 255 = white")
    print(f"Very dark: {len(dark)}");     print(dark[cols + ["mean_brightness"]].head(5).round(1).to_string(index=False))
    print(f"\nVery bright: {len(bright)}"); print(bright[cols + ["mean_brightness"]].head(5).round(1).to_string(index=False))

    categories = [
        ("Corrupted (cannot be opened)", bad, "fails to decode"),
        ("Grayscale", gray, f"mode L, or RGB with channel difference <= {cfg.gray_tolerance}"),
        ("Small", small, f"shorter side < {cfg.small_side_px} px"),
        ("Very wide", wide, f"aspect > {cfg.wide_aspect}"),
        ("Very tall", tall, f"aspect < {cfg.tall_aspect}"),
        ("Very dark", dark, f"mean brightness < {cfg.dark_brightness}"),
        ("Very bright", bright, f"mean brightness > {cfg.bright_brightness}"),
    ]
    _header("Summary: unusual images (official train set)")
    print(f"{'category':<30}{'count':>6}{'share':>8}   rule")
    for name, sub, rule in categories:
        print(f"{name:<30}{len(sub):>6}{len(sub) / len(df):>8.1%}   {rule}")
    flagged = set().union(*(set(sub["index"]) for _, sub, _ in categories))
    print(f"{'Any of the above (unique)':<30}{len(flagged):>6}{len(flagged) / len(df):>8.1%}"
          "   an image can be in several categories")

    # ------------------------------------------------------------------ figures
    figs = Path(cfg.figures_dir)
    out_sizes = plot_image_sizes(ok, cfg.small_side_px, figs / "step03_image_sizes.png")
    n_rgb_gray = int((gray["mode"] == "RGB").sum())
    out_modes = plot_color_modes(
        ["RGB (color)", "RGB, but no color", "L (grayscale)"],
        [int((ok["mode"] == "RGB").sum()) - n_rgb_gray, n_rgb_gray, int((ok["mode"] == "L").sum())],
        figs / "step03_color_modes.png",
    )
    n = cfg.n_unusual
    groups = [
        (f"Grayscale ({len(gray)})", gray.head(n), lambda r: f"{r['mode']}, max diff {r['max_channel_diff']}"),
        (f"Small ({len(small)})", small.head(n), lambda r: f"{r['width']}x{r['height']}"),
        (f"Wide {len(wide)} / tall {len(tall)}", pd.concat([wide.head(n - min(len(tall), 2)), tall.head(2)]),
         lambda r: f"aspect {r['aspect']:.2f}"),
        (f"Very dark ({len(dark)})", dark.head(n), lambda r: f"brightness {r['mean_brightness']:.0f}"),
        (f"Very bright ({len(bright)})", bright.head(n), lambda r: f"brightness {r['mean_brightness']:.0f}"),
    ]
    rows = []
    for title, sub, describe in groups:
        images = [train[int(i)]["image"] for i in sub["index"]]
        captions = [f"{r['class_name']}\nrow {r['index']} | {describe(r)}" for _, r in sub.iterrows()]
        rows.append((title, images, captions))
    out_grid = plot_image_rows(rows, "Step 3 - unusual training images", figs / "step03_unusual_images.png")

    _header("Files")
    for p in [Path(cfg.stats_dir) / "image_stats.csv", out_sizes, out_modes, out_grid]:
        print(f"  {p}")


def step4_duplicates(cfg: EDAConfig) -> None:
    """Exact (MD5) and near (pHash) duplicates within the official train set."""
    train = load_cub_split(cfg.train_name, "train", cfg.cache_dir)
    names = get_label_names(train)
    hashes = load_or_build_hashes(train, Path(cfg.stats_dir) / "image_hashes.csv", cfg.rebuild_stats)

    def describe(i: int) -> str:
        img = train[i]["image"]
        return f"row {i:>4}  {names[hashes.label[i]]:<26} {img.width}x{img.height}"

    _header("STEP 4 - Duplicates (official train set)")
    exact = exact_duplicate_groups(hashes)
    print(f"Exact duplicates (identical file bytes, same MD5): {len(exact)} group(s)")
    for g in exact:
        print("  " + "  =  ".join(describe(i) for i in g))

    near = find_near_duplicates(hashes["phash"], cfg.dup_max_distance)
    near["same_class"] = hashes.label[near.a].to_numpy() == hashes.label[near.b].to_numpy()
    near["type"] = [
        duplicate_type(hashes.md5[a], hashes.md5[b], train[a]["image"].size, train[b]["image"].size)
        for a, b in zip(near.a, near.b)
    ]
    _header(f"All duplicate pairs (pHash Hamming distance <= {cfg.dup_max_distance} of 64 bits)")
    print(f"Pairs: {len(near)}   same class: {near.same_class.sum()}   different class: {(~near.same_class).sum()}")
    for r in near.itertuples():
        print(f"  [{r.type}] d={r.distance}:  {describe(r.a)}   ~   {describe(r.b)}")
    involved = sorted(set(near.a) | set(near.b))
    print(f"\nImages involved: {len(involved)}  ->  extra copies: {len(involved) - len(near)}")

    _header("Duplicate pairs by type")
    print(f"{'type':<20}{'pairs':>6}   meaning")
    meaning = {
        "exact": "identical file bytes (same MD5)",
        "resized": "same photo, different width x height",
        "edited / re-saved": "same photo and size, pixels changed (brightness/contrast or recompression)",
    }
    for t in DUPLICATE_TYPES:
        print(f"{t:<20}{(near.type == t).sum():>6}   {meaning[t]}")
    print(f"{'total':<20}{len(near):>6}")

    nearest = nearest_distances(hashes["phash"])
    _header("Distance from each image to its most similar other image")
    for d, n in zip(*np.unique(nearest, return_counts=True)):
        if d <= cfg.dup_max_distance + 6:
            print(f"  distance {d:>2}: {n:>5} images")
    print(f"  median: {np.median(nearest):.0f} bits  -> a typical image is far from every other image")

    lookalikes = find_near_duplicates(hashes["phash"], cfg.dup_max_distance + 4)
    lookalikes = lookalikes[lookalikes.distance > cfg.dup_max_distance].head(cfg.n_lookalikes)

    # ------------------------------------------------------------------ figures
    def pair(a: int, b: int, label: str) -> tuple:
        ia, ib = train[a]["image"], train[b]["image"]
        return (ia, ib, f"{names[hashes.label[a]]}\nrow {a} | {ia.width}x{ia.height}",
                f"{names[hashes.label[b]]}\nrow {b} | {ib.width}x{ib.height}", label)

    figs = Path(cfg.figures_dir)
    out_dups = plot_image_pairs([pair(r.a, r.b, f"{r.type}, d = {r.distance}") for r in near.itertuples()],
                                "Step 4 - duplicate pairs in the train set", figs / "step04_duplicate_pairs.png")
    out_look = plot_image_pairs([pair(r.a, r.b, f"d = {r.distance}") for r in lookalikes.itertuples()],
                                "Step 4 - closest pairs that are NOT duplicates",
                                figs / "step04_closest_non_duplicates.png")
    out_hist = plot_nearest_distance(nearest.tolist(), cfg.dup_max_distance, figs / "step04_nearest_distance.png")

    _header("Files")
    for p in [Path(cfg.stats_dir) / "image_hashes.csv", out_dups, out_look, out_hist]:
        print(f"  {p}")


def step5_samples(cfg: EDAConfig) -> None:
    """Random sample grid with captions, and rows of look-alike species (fine-grained task)."""
    train = load_cub_split(cfg.train_name, "train", cfg.cache_dir)
    names = get_label_names(train)
    rng = random.Random(cfg.seed)

    # The first caption only: proper caption splitting/cleaning is step 6.
    def first_caption(description: str) -> str:
        return description.split("\n")[0].strip()

    _header(f"STEP 5 - {cfg.n_samples} random training images (seed {cfg.seed})")
    idx = sorted(rng.sample(range(len(train)), cfg.n_samples))
    images, titles = [], []
    for i in idx:
        rec = train[i]
        caption = first_caption(rec["description"])
        print(f"row {i:>4}  {names[rec['label']]:<28} \"{caption}\"")
        images.append(rec["image"])
        titles.append(f"{names[rec['label']]} (row {i})\n" + textwrap.fill(f'"{caption}"', cfg.caption_wrap))
    out_grid = plot_image_grid(images, titles, cfg.sample_cols,
                               f"Step 5 - {cfg.n_samples} random training images with one of their captions",
                               Path(cfg.figures_dir) / "step05_random_samples.png")

    _header("How fine-grained is the task? Species sharing their last name word")
    families = pd.Series([n.split()[-1] for n in names]).value_counts()
    print(f"{'family word':<14}{'species':>8}")
    for word, k in families[families > 1].head(12).items():
        print(f"{word:<14}{k:>8}")
    print(f"... {int((families > 1).sum())} family words are shared by 2+ species, "
          f"covering {int(families[families > 1].sum())} of {len(names)} species")

    rows = []
    for word in cfg.similar_groups:
        group = [lbl for lbl, n in enumerate(names) if word in n.split()]
        _header(f"Look-alike group '{word}': {len(group)} species, one random image each")
        group_images, group_titles = [], []
        for lbl in group:
            i = rng.choice([j for j, y in enumerate(train["label"]) if y == lbl])
            rec = train[i]
            print(f"  {names[lbl]:<24} row {i:>4}  \"{first_caption(rec['description'])}\"")
            group_images.append(rec["image"])
            group_titles.append(f"{names[lbl]}\nrow {i}")
        rows.append((f"{word}s ({len(group)})", group_images, group_titles))
    out_similar = plot_image_rows(rows, "Step 5 - look-alike species: classes differ in small details",
                                  Path(cfg.figures_dir) / "step05_similar_species.png")

    _header("Files")
    for p in [out_grid, out_similar]:
        print(f"  {p}")


def step6_captions(cfg: EDAConfig) -> None:
    """Captions: count per image, duplicates, lengths, vocabulary, characters, typos, color/body words."""
    from spellchecker import SpellChecker  # only this step needs it

    train = load_cub_split(cfg.train_name, "train", cfg.cache_dir)
    descriptions = train["description"]  # text column only, no images decoded
    caps = caption_table(descriptions)
    figs = Path(cfg.figures_dir)

    # ------------------------------------------------------------ 1. structure
    _header("STEP 6 - Caption analysis (official train set)")
    raw_lines = pd.Series([len(d.split("\n")) for d in descriptions])
    per_image = caps.groupby("image").size().reindex(range(len(descriptions)), fill_value=0)
    print(f"Images: {len(descriptions)}   captions (non-empty): {len(caps)}")
    print(f"Raw lines per description (split on \\n): {raw_lines.value_counts().to_dict()}")
    print(f"Descriptions ending with \\n: {sum(d.endswith(chr(10)) for d in descriptions)}  "
          "-> the last split piece is always an empty string")
    print(f"Non-empty captions per image: {per_image.value_counts().to_dict()}")
    not_ten = per_image[per_image != 10]
    print(f"Images WITHOUT exactly 10 captions: {len(not_ten)}" + (f" -> {not_ten.to_dict()}" if len(not_ten) else ""))
    print(f"Captions containing uppercase letters: {(caps.caption != caps.caption.str.lower()).sum()}")

    # ------------------------------------------------------------ 2. duplicates
    _header("Duplicate captions")
    dup_in_image = caps.groupby("image")["caption"].apply(lambda s: len(s) - s.nunique())
    print(f"Images with a repeated caption: {(dup_in_image > 0).sum()}  "
          f"(repeats per image: {dup_in_image[dup_in_image > 0].value_counts().sort_index().to_dict()})")
    print(f"Repeated captions within images (extra copies): {dup_in_image.sum()}")
    ex = dup_in_image[dup_in_image > 0].index[0]
    rep = caps[caps.image == ex].caption.value_counts()
    print(f"  example, image {ex}: {rep[rep > 1].index[0]!r} x{rep.iloc[0]}")
    distinct = caps.caption.nunique()
    print(f"\nDistinct captions in the whole set: {distinct} of {len(caps)} ({distinct / len(caps):.1%})")
    print("Most repeated captions across images (template-like):")
    for text, n in caps.caption.value_counts().head(10).items():
        print(f"  {n:>4} x  {text!r}")

    # --------------------------------------------------------------- 3. length
    _header("Caption length")
    for col, unit in [("n_words", "words"), ("n_chars", "characters")]:
        q = caps[col].quantile([0.95, 0.99])
        print(f"{unit:<11} min {caps[col].min():>3} | median {caps[col].median():>4.0f} | mean {caps[col].mean():>5.1f} | "
              f"95th pct {q[0.95]:>4.0f} | 99th pct {q[0.99]:>4.0f} | max {caps[col].max():>3}")
    print("\nShortest captions:")
    for c in caps.nsmallest(5, "n_words").caption:
        print(f"  {c!r}")
    print("Longest caption:")
    print(f"  {caps.loc[caps.n_words.idxmax(), 'caption']!r}")

    # ----------------------------------------------------------- 4. vocabulary
    _header("Vocabulary (words = lowercase letters with optional apostrophe; digits/punctuation excluded)")
    words = pd.Series([w for c in caps.caption for w in tokenize(c)]).value_counts()
    print(f"Word occurrences (tokens): {words.sum():,}   distinct words (vocabulary): {len(words):,}")
    print(f"Words seen once: {(words == 1).sum():,}   twice: {(words == 2).sum():,}   "
          f"-> {((words <= 2).sum()) / len(words):.0%} of the vocabulary is rare")
    print(f"\n{'min frequency':<16}{'vocab size':>11}{'tokens covered':>16}")
    vocab_sizes = []
    for t in cfg.min_freq_options:
        kept = words[words >= t]
        vocab_sizes.append(len(kept))
        print(f">= {t:<13}{len(kept):>11,}{kept.sum() / words.sum():>16.2%}")
    print(f"\nTop {cfg.n_top_words} words:")
    print("  " + ", ".join(f"{w} ({n})" for w, n in words.head(cfg.n_top_words).items()))

    # ----------------------------------------------------------- 5. characters
    _header("Characters")
    text = "".join(caps.caption)
    punct = pd.Series([ch for ch in text if not ch.isalnum() and ch != " "]).value_counts()
    print("Punctuation / symbols: " + ", ".join(f"{ch!r} {n}" for ch, n in punct.items()))
    broken = caps[caps.caption.str.contains("�")]
    print(f"\nCaptions with the replacement character U+FFFD (broken encoding): {len(broken)} "
          f"({text.count(chr(0xFFFD))} characters)")
    if len(broken):
        print(f"  example (repr): {broken.caption.iloc[0][:80]!r}...")
        print("  pattern: every space was replaced by two U+FFFD characters -> the words are intact")
    other = pd.Series([ch for ch in text if ord(ch) > 127 and ch != "�"]).value_counts()
    print(f"Other non-ASCII characters: {other.to_dict() if len(other) else 'none'}")
    digits = caps[caps.caption.str.contains(r"\d")]
    print(f"Captions with digits: {len(digits)}, e.g. {digits.caption.iloc[0][-60:]!r}" if len(digits) else "Digits: none")

    # ---------------------------------------------------------------- 6. typos
    _header("Possible typos (words not in an English dictionary, pyspellchecker)")
    spell = SpellChecker()
    unknown = words[[w not in spell for w in words.index]]
    print(f"Unknown words: {len(unknown):,} distinct, {unknown.sum():,} occurrences "
          f"({unknown.sum() / words.sum():.2%} of all tokens)")
    print("\nMost frequent unknown words (bird vocabulary AND its misspellings):")
    print("  " + ", ".join(f"{w} ({n})" for w, n in unknown.head(30).items()))
    print("\nRare unknown words with the spellchecker's suggestion (NOT applied; many are wrong):")
    for w in unknown[unknown <= 2].index[:15]:
        example = next(c for c in caps.caption if w in tokenize(c))
        print(f"  {w:<14} -> {str(spell.correction(w)):<12} in {example[:70]!r}")

    # ------------------------------------------------------ 7. color/body words
    _header("Color words and body-part words")
    colors = [(w, int(words.get(w, 0))) for w in cfg.color_words]
    parts = [(w, int(words.get(w, 0))) for w in cfg.body_part_words]
    colors.sort(key=lambda x: -x[1])
    parts.sort(key=lambda x: -x[1])
    print("Colors:     " + ", ".join(f"{w} ({n})" for w, n in colors))
    print("Body parts: " + ", ".join(f"{w} ({n})" for w, n in parts))
    has_color = caps.caption.apply(lambda c: any(w in cfg.color_words for w in tokenize(c)))
    print(f"\nCaptions mentioning at least one color word: {has_color.mean():.1%}")

    # -------------------------------------------------------------- figures
    content = words[[w not in cfg.stop_words for w in words.index]]
    outs = [
        plot_caption_lengths(caps.n_words, caps.n_chars, figs / "step06_caption_lengths.png"),
        plot_word_bars(
            [(f"Top {cfg.n_top_words} words", list(words.head(cfg.n_top_words).items())),
             (f"Top {cfg.n_top_words} content words (stop words removed)", list(content.head(cfg.n_top_words).items()))],
            "Step 6 - most frequent words in the captions", figs / "step06_top_words.png"),
        plot_word_bars([("Color words", colors), ("Body-part words", parts)],
                       "Step 6 - what the captions talk about", figs / "step06_color_bodypart_words.png"),
        plot_word_frequency(words.tolist(), cfg.min_freq_options, vocab_sizes, figs / "step06_word_frequency.png"),
    ]

    # ------------------------------------------- examples of what needs fixing
    names = get_label_names(train)
    rows = []
    if len(broken):
        b = broken.iloc[0]
        rows.append((f"1. Broken encoding: {len(broken)} captions (spaces became U+FFFD)", train[int(b.image)]["image"], [
            (f"image {b.image} - {names[train[int(b.image)]['label']]}", False),
            ("raw:      " + b.caption, True),
            ("repaired: " + " ".join(b.caption.replace("�", " ").split()), False),
        ]))
    rep_img = int(dup_in_image.idxmax())  # the image with the most repeats
    rep_caps = caps[caps.image == rep_img].caption.tolist()
    counts = pd.Series(rep_caps).value_counts()
    rows.append((f"2. Repeated within one image: {(dup_in_image > 0).sum()} images, {dup_in_image.sum()} extra copies",
                 train[rep_img]["image"],
                 [(f"image {rep_img} - {names[train[rep_img]['label']]} - its 10 captions:", False)]
                 + [(f"{k + 1:>2}. {c}" + (f"   <- appears {counts[c]}x" if counts[c] > 1 else ""), counts[c] > 1)
                    for k, c in enumerate(rep_caps)]))
    short = caps[caps.n_words < 2]
    for _, s_row in short.iterrows():
        rows.append((f"3. Cut-off caption ({len(short)})", train[int(s_row.image)]["image"],
                     [(f"image {s_row.image} - {names[train[int(s_row.image)]['label']]}", False),
                      (f"caption: {s_row.caption!r}", True),
                      ("other captions of the same image, for comparison:", False)]
                     + [(f"   {c}", False) for c in caps[caps.image == s_row.image].caption.head(3)]))
    typo_lines = [("word        sentence it appears in", False)]
    for w in cfg.typo_examples:
        hit = next((c for c in caps.caption if w in tokenize(c)), None)
        if hit is not None:
            typo_lines.append((f"{w:<11} {hit}  ({int(words.get(w, 0))}x in total)", True))
    rows.append((f"4. Typos: {len(unknown):,} words not in a dictionary ({unknown.sum() / words.sum():.2%} of words)",
                 None, typo_lines))
    outs.append(plot_text_rows(rows, "Step 6 - caption problems found in the official train set",
                               figs / "step06_caption_problems.png"))

    template = caps.caption.value_counts().index[0]
    holders = caps[caps.caption == template].image.drop_duplicates()
    picked, seen = [], set()
    for i in holders:  # one image per species, to show it is used for different birds
        lbl = train[int(i)]["label"]
        if lbl not in seen:
            seen.add(lbl)
            picked.append(int(i))
        if len(picked) == cfg.n_template_examples:
            break
    outs.append(plot_image_rows(
        [(f"{len(holders)} images", [train[i]["image"] for i in picked],
          [f"{names[train[i]['label']]}\nimage {i}" for i in picked])],
        f'Step 6 - the same caption on different birds: "{template}"', figs / "step06_template_captions.png"))
    _header("Files")
    for p in outs:
        print(f"  {p}")


STEPS = {
    1: step1_inspect,
    2: step2_class_distribution,
    3: step3_image_properties,
    4: step4_duplicates,
    5: step5_samples,
    6: step6_captions,
}
