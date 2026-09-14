"""Transcript normalization for the aligner: numerals spelled out, token counts stable."""

import sys

import pytest

from vemsa.pipeline.normalize import (
    alignable_tokens,
    alignment_normalizer,
    number_speller,
    spell_number_sv,
)

# ---- the Swedish speller


@pytest.mark.parametrize(
    ("digits", "spelled"),
    [
        ("0", "noll"),
        ("1", "ett"),
        ("2", "två"),
        ("12", "tolv"),
        ("21", "tjugoett"),
        ("57", "femtiosju"),
        ("99", "nittionio"),
        ("100", "hundra"),
        ("102", "hundratvå"),
        ("110", "hundratio"),
        ("999", "niohundranittionio"),
        ("1000", "tusen"),
        ("1001", "tusenett"),
        ("1050", "tusenfemtio"),
        ("1100", "elvahundra"),  # 1100-1999 read in hundreds, as years and amounts are
        ("1500", "femtonhundra"),
        ("1999", "nittonhundranittionio"),
        ("2000", "tvåtusen"),
        ("2024", "tvåtusentjugofyra"),
        ("21000", "tjugoettusen"),  # ett+tusen collapses to one long t
        ("100000", "hundratusen"),
        ("999999", "niohundranittioniotusenniohundranittionio"),
        ("1000000", "enmiljon"),
        ("2000000", "tvåmiljoner"),
        ("21000000", "tjugoenmiljoner"),
        ("1234567", "enmiljontvåhundratrettiofyratusenfemhundrasextiosju"),
        ("08", "nollåtta"),  # leading zero: read digit by digit
        ("007", "nollnollsju"),
        ("0812345678", "nollåttaetttvåtrefyrafemsexsjuåtta"),
        ("1000000000", "ettnollnollnollnollnollnollnollnollnoll"),  # beyond the cardinals
    ],
)
def test_spell_number_sv(digits: str, spelled: str):
    assert spell_number_sv(digits) == spelled


def test_swedish_spellings_are_single_words():
    # the aligner contract: an expansion never adds whitespace, so the token
    # count of a text is the same before and after expansion
    for n in [*range(0, 2100), 21000, 999_999, 1_000_000, 21_000_000, 999_999_999]:
        spelled = spell_number_sv(str(n))
        assert spelled and spelled.split() == [spelled], n
        assert spelled.isalpha(), n


@pytest.mark.parametrize("language", ["sv", "SV", "sv-SE", " sv "])
def test_swedish_speller_is_picked_for_swedish_variants(language: str):
    assert number_speller(language) is spell_number_sv


@pytest.mark.parametrize("language", ["auto", "unknown", "", None, "en", "fi"])
def test_other_languages_keep_digits(language: str | None):
    assert number_speller(language) is None


# ---- the easyaligner normalizer


@pytest.fixture
def sv_normalizer():
    pytest.importorskip("easyaligner.text.normalization")
    return alignment_normalizer("sv")


def test_normalizer_builds_without_easyaligner(monkeypatch):
    """The engines build the normalizer before every pipeline run; only the run
    itself (which needs easyaligner anyway) may import it, so the stubbed-pipeline
    tests pass without the `align` extra."""
    monkeypatch.setitem(sys.modules, "easyaligner", None)
    monkeypatch.setitem(sys.modules, "easyaligner.text.normalization", None)
    normalize = alignment_normalizer("sv")
    with pytest.raises(ModuleNotFoundError):
        normalize("hej")


def test_numerals_are_spelled_for_the_model_but_reported_verbatim(sv_normalizer):
    tokens, mapping = sv_normalizer("Jag är 57 år")
    assert tokens == ["jag", "är", "femtiosju", "år"]
    numeral = mapping[2]
    # the aligned word keeps its original text and character offsets
    assert numeral["normalized_token"] == "femtiosju"
    assert numeral["text"] == "57"
    assert (numeral["start_char"], numeral["end_char"]) == (7, 9)
    assert "Jag är 57 år"[7:9] == "57"


@pytest.mark.parametrize(
    ("text", "tokens", "original"),
    [
        ("50-tal", ["femtiotal"], "50-tal"),  # digits glued to letters expand in place
        ("1,5", ["ettfem"], "1,5"),  # decimals: one word, timed by the audio, low score
        ("08:30", ["nollåttatrettio"], "08:30"),
        ("3:e", ["tree"], "3:e"),
        ("5%", ["fem"], "5%"),
        ("(2024)", ["tvåtusentjugofyra"], "(2024)"),
        ("１０２", ["hundratvå"], "１０２"),  # NFKC folds fullwidth digits first
    ],
)
def test_digit_runs_inside_tokens_stay_one_token(sv_normalizer, text, tokens, original):
    normalized, mapping = sv_normalizer(text)
    assert normalized == tokens
    assert [item["text"] for item in mapping] == [original]


def test_thousands_separated_by_spaces_stay_separate_tokens(sv_normalizer):
    # "1 500" is two tokens either way; only "1500" spells as one word
    tokens, mapping = sv_normalizer("1 500 kr")
    assert tokens == ["ett", "femhundra", "kr"]
    assert [item["text"] for item in mapping] == ["1", "500", "kr"]


def test_other_languages_reproduce_the_library_default():
    normalization = pytest.importorskip("easyaligner.text.normalization")
    for language in ["en", "auto", "unknown"]:
        normalize = alignment_normalizer(language)
        for text in ["Jag är 57 år", "Hej, och välkomna!", "Vi ses kl. 14:30 (om det går)."]:
            assert normalize(text) == normalization.text_normalizer(text), (language, text)


# ---- alignable_tokens: the count contract task=align relies on

NUMERAL_TEXTS = [
    "Jag är 57 år",
    "Hej. Jag är 57 år gammal och min mamma är 102.",
    "Vi ses kl. 08:30 på Sveavägen 12 (om det går).",
    "1 500 kr, d.v.s. 1,5 tusen — 100% säkert…",
    "50-talet, 3:e gången, år 2024",
    "0812345678",
    "...",
    "",
]


@pytest.mark.parametrize("text", NUMERAL_TEXTS)
@pytest.mark.parametrize("language", ["sv", "en", "auto"])
def test_alignable_tokens_matches_the_normalizer_in_every_language(text: str, language: str):
    pytest.importorskip("easyaligner.text.normalization")
    tokens, _ = alignment_normalizer(language)(text)
    assert alignable_tokens(text) == len(tokens), (language, text)


def test_alignable_tokens_is_unchanged_by_expansion():
    assert alignable_tokens("Jag är 57 år") == 4
    assert alignable_tokens("Hej. Jag är 57 år gammal och min mamma är 102.") == 11
