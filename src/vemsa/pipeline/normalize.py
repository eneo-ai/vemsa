"""Transcript normalization for forced alignment, shared by every aligner call.

easyaligner (and easytranscriber, which wraps it) normalizes each window's text
before aligning it: NFKC, lowercase, strip everything that is neither a word
character nor whitespace, split on whitespace. A per-character span map leads
every normalized token back to the original text, so the aligned word is still
reported with its original spelling and character offsets; only the string the
CTC model sees changes.

Vemsa's normalizer reproduces that default and adds one transform: digit runs
are spelled out as cardinals in the job's language. The default Swedish CTC
model (KBLab/wav2vec2-large-voxrex-swedish) has the digits 0-9 in its
vocabulary, but nobody ever *says* "5 7", so the digit tokens carry no acoustic
mass: aligning "57" against the audio of "femtiosju" yields a span score of
~1e-9 (0.0 after easyaligner's rounding) — the exact value Vemsa defines as
"interpolated, not aligned" — and squeezed timestamps. Spelled out, the same
word aligns like any other (verified on a synthetic clip: 0.93 vs 3.6e-9).

An expansion never contains whitespace ("1 500" is two tokens either way, and
"1500" spells to one), so the token count of a text is the same with or without
expansion and independent of language. task=align relies on that invariant to
hand aligned words back to the segments they came from (``alignable_tokens``).

Digit runs glued to letters or punctuation expand in place, which for the
common cases matches how they are read: "50-tal" -> "femtiotal", "1,5" ->
"ettfem" (spoken "en komma fem": low score, but timed by the audio), "08:30"
-> "nollåttatrettio". Ordinals ("3:e" -> "tree"), percentages ("5%" -> "fem")
and years spoken the other way ("2024" as "tjugohundratjugofyra") score low
but no longer exactly 0.0. Non-digit symbols the CTC vocabulary lacks are still
dropped by the punctuation strip, as before."""

import re
import unicodedata
from collections.abc import Callable

from vemsa.jobs.models import Language

# text -> (normalized tokens, easyaligner span mapping), easyaligner's contract
TextNormalizer = Callable[[str], tuple[list[str], list[dict]]]

_DIGIT_RUN = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")

_SV_ONES = [
    "noll",
    "ett",
    "två",
    "tre",
    "fyra",
    "fem",
    "sex",
    "sju",
    "åtta",
    "nio",
    "tio",
    "elva",
    "tolv",
    "tretton",
    "fjorton",
    "femton",
    "sexton",
    "sjutton",
    "arton",
    "nitton",
]
_SV_TENS = {
    2: "tjugo",
    3: "trettio",
    4: "fyrtio",
    5: "femtio",
    6: "sextio",
    7: "sjuttio",
    8: "åttio",
    9: "nittio",
}
_SV_MAX_CARDINAL = 999_999_999


def _sv_below_hundred(n: int) -> str:
    if n < 20:
        return _SV_ONES[n]
    tens, ones = divmod(n, 10)
    return _SV_TENS[tens] + (_SV_ONES[ones] if ones else "")


def _sv_below_thousand(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    spelled = ""
    if hundreds:
        # "hundra", not "etthundra": the unstressed form is how it is read aloud
        spelled += ("" if hundreds == 1 else _SV_ONES[hundreds]) + "hundra"
    if rest:
        spelled += _sv_below_hundred(rest)
    return spelled


def _sv_below_million(n: int) -> str:
    if 1100 <= n <= 1999:
        # read in hundreds ("nittonhundranittionio", "femtonhundra"): the form
        # years and round amounts take in speech
        rest = n % 100
        return _sv_below_hundred(n // 100) + "hundra" + (_sv_below_hundred(rest) if rest else "")
    thousands, rest = divmod(n, 1000)
    if thousands == 0:
        return _sv_below_thousand(rest)
    prefix = "" if thousands == 1 else _sv_below_thousand(thousands)
    # "ett" + "tusen" is written and said "ettusen" (one long t)
    spelled = (prefix + "tusen").replace("ttt", "tt")
    return spelled + (_sv_below_thousand(rest) if rest else "")


def spell_number_sv(digits: str) -> str:
    """Swedish cardinal for a digit run, as one whitespace-free word.

    Numbers a Swede reads as a figure ("femtiosju", "tvåtusentjugofyra",
    "enmiljon") are spelled as such up to 999 999 999. Anything read digit by
    digit — a leading zero (phone numbers, "08:30") or a longer run — is spelled
    digit by digit ("nollåttatrettio"), which is also what the speaker said."""
    if (len(digits) > 1 and digits[0] == "0") or int(digits) > _SV_MAX_CARDINAL:
        return "".join(_SV_ONES[int(digit)] for digit in digits)
    n = int(digits)
    if n == 0:
        return _SV_ONES[0]
    millions, rest = divmod(n, 1_000_000)
    if millions == 0:
        return _sv_below_million(rest)
    if millions == 1:
        spelled = "enmiljon"
    else:
        # miljon takes the common-gender "en": "tjugoen miljoner"
        prefix = _sv_below_thousand(millions)
        spelled = (prefix[:-3] + "en" if prefix.endswith("ett") else prefix) + "miljoner"
    return spelled + (_sv_below_million(rest) if rest else "")


# language (lower-cased, region stripped) -> digit-run speller; a language without
# an entry aligns its digits as-is, which the CTC model cannot hear
_NUMBER_SPELLERS: dict[str, Callable[[str], str]] = {"sv": spell_number_sv}


def number_speller(language: Language | str | None) -> Callable[[str], str] | None:
    """The digit-run speller for a job language, or None to leave digits alone.

    ``auto``/``unknown`` carry no signal: the numerals' language is unknown, so
    they stay digits (and align as badly as before) rather than being spelled in
    a guessed language."""
    if not language:
        return None
    key = language.strip().lower().split("-")[0]
    return _NUMBER_SPELLERS.get(key)


def alignment_normalizer(language: Language | str | None) -> TextNormalizer:
    """easyaligner ``text_normalizer_fn`` for a job language: the library
    default, plus digit runs spelled out for languages that have a speller."""
    from easyaligner.text.normalization import SpanMapNormalizer

    speller = number_speller(language)

    def normalize(text: str) -> tuple[list[str], list[dict]]:
        normalizer = SpanMapNormalizer(text)
        normalizer.transform(r"\S+", lambda m: unicodedata.normalize("NFKC", m.group()))
        if speller is not None:
            # keep the token count invariant: an expansion may not add whitespace
            normalizer.transform(
                _DIGIT_RUN.pattern, lambda m: _WHITESPACE.sub("", speller(m.group()))
            )
        normalizer.transform(r"\S+", lambda m: m.group().lower())
        normalizer.transform(r"[^\w\s]", "")
        normalizer.transform(r"\s+", " ")
        normalizer.transform(r"^\s+|\s+$", "")
        mapping = normalizer.get_token_map()
        return [item["normalized_token"] for item in mapping], mapping

    return normalize


_NON_ALIGNABLE = re.compile(r"[^\w\s]")


def alignable_tokens(text: str) -> int:
    """How many words the aligner will produce for ``text``, in any language.

    Mirrors the normalizer above without easyaligner installed: NFKC, drop
    everything that is neither a word character nor whitespace, split on
    whitespace. Digit expansion cannot change the count (it never adds
    whitespace, and a digit run is a word character run before and after), so
    this holds for every language; ``tests/test_normalize.py`` pins both
    equalities. A punctuation-only token ("—", "...") vanishes under the
    normalization and therefore counts as no word."""
    normalized = unicodedata.normalize("NFKC", text)
    return len(_NON_ALIGNABLE.sub("", normalized).split())
