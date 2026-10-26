# Vendored unchanged from https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-Arabic
# (normalization.py at commit 7c60cf2acb1ceb823efe937ddbc88e5e7b6e7dc2), Apache-2.0. Use it through transforms.VoxtralArabicNormalize.
"""Arabic ASR evaluation normalization.

Apply ``normalize_arabic_asr`` to both the reference and hypothesis before
computing word error rate. The rules remove diacritics and punctuation,
normalize spelling and clitics, map selected lexical variants, and convert
recognized spoken numbers to ASCII digits.

This is a fixed evaluation policy, not a general Arabic text normalizer.
It always maps ذ to د and ث to ت, but preserves ظ versus ض. It also changes
some word boundaries and Arabic/French loanword spellings. Scores depend on
these choices; use the same policy when comparing systems.

Usage::

    from normalization import normalize_arabic_asr

    reference = normalize_arabic_asr(reference)
    hypothesis = normalize_arabic_asr(hypothesis)

Only the Python standard library is required. ``trace_normalization`` returns
intermediate stages for inspecting a result.
"""

import math
import re
import unicodedata
from collections.abc import Sequence


# Generic Unicode and token patterns

LANG_TAG_RE = re.compile(r"\[(?:ar|fr|en)\]", re.IGNORECASE)
PUNCT_RE = re.compile(r"[،.؟!,?:;»«\"'؛\-–—…()\[\]{}\\/|_%+*=<>]")
ARABIC_WORD = r"[\u0621-\u063A\u0641-\u064A]+"
ARABIC_TOKEN_RE = re.compile(rf"^{ARABIC_WORD}$")

DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

# Evaluation policy: always-on phonetic Darija mappings.
PHONETIC_TRANSLATION = str.maketrans(
    {
        "ذ": "د",
        "ث": "ت",
    }
)


# Generic productive morphology

# Definite-article clitics. Longer combinations must run first.
_CLITIC_RULES = [
    (re.compile(rf"\bوبال({ARABIC_WORD})"), r"و ب ال\1"),
    (re.compile(rf"\bولل({ARABIC_WORD})"), r"و ل ال\1"),
    (re.compile(rf"\bفال({ARABIC_WORD})"), r"في ال\1"),
    (re.compile(rf"\bوال({ARABIC_WORD})"), r"و ال\1"),
    (re.compile(rf"\bبال({ARABIC_WORD})"), r"ب ال\1"),
    (re.compile(rf"\bلل({ARABIC_WORD})"), r"ل ال\1"),
]

_PREFIX_RULES = [
    ("ماكات", "ماكت"),
    ("ماكاد", "ماكد"),
    ("ماكاي", "ماكي"),
    ("ماكان", "ماكن"),
    ("كات", "كت"),
    ("كاد", "كد"),
    ("كاي", "كي"),
    ("تان", "تن"),
]

_ASPECT_MARKERS = {"كي", "كت", "كن", "تن"}
_MIN_LEN_WAW_ALEF = 4
_KAN_PREFIX_EXCEPTIONS = {"كان", "كانت", "كانو", "كانوا", "كانون", "كانونيا"}

# Generic pronoun-family suffixes.
_L_B_PRONOUN_SUFFIXES = {
    "ا": "ي",  # ليا / بيا
    "ك": "ك",  # ليك / بيك
    "كي": "ك",  # ليكي / بيكي
    "ه": "ه",  # ليه / بيه
    "ها": "ها",  # ليها / بيها
    "نا": "نا",  # لينا / بينا
    "كم": "كم",  # ليكم / بيكم
    "هم": "هم",  # ليهم / بيهم
}

_MAA_PRONOUN_SUFFIXES = {
    "ي": "ي",  # معاي
    "يا": "ي",  # معايا
    "ك": "ك",  # معاك
    "كي": "ك",  # معاكي
    "ه": "ه",  # معاه
    "ها": "ها",  # معاها
    "نا": "نا",  # معانا
    "كم": "كم",  # معاكم
    "هم": "هم",  # معاهم
}

# Bare negative forms are generalized only when they end in a high-confidence
# Darija negative ending. Productive aspect forms are handled separately.
_NEGATION_EXCEPTIONS = {
    "ماشي",  # lexical negative copula; do not split to ما شي
    "مراكش",  # place name
}
_NEGATIVE_ENDINGS = ("اش", "يش", "تش")

# No-final-ش negatives cannot be recognized safely by spelling alone because
# forms such as مكتوب and مكينة are ordinary lexical words. Keep only validated
# verb stems here; inflected forms sharing the stem are handled generically.
_NEGATIVE_NO_SH_STEMS = {"كون"}


# Lexical resources: hardcoded only where spelling is ambiguous

# Each group pairs a canonical spelling with its accepted variants.
_WORD_ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("لي", ("اللي", "للي", "لي")),
    ("هما", ("هوما", "هما")),
    ("كلشي", ("كولشي", "كلشي")),
    ("غدي", ("غادي", "غدي")),
    ("بلي", ("باللي", "بلي")),
    ("خصك", ("خاصك", "خصك")),
    ("ستوديوهات", ("استوديوهات", "ستوديوهات")),
    ("الما", ("الماء", "الما")),
    ("اليوم", ("ليوما", "اليوم")),
    ("يالله", ("يلاه", "يالله")),
    ("الاخر", ("الاخر", "اللخر", "لخر")),
    ("كتر", ("اكتر", "كتر")),
    ("اكسجين", ("الاوكسيجين", "الاكسجين", "اوكسيجين", "اكسجين")),
    ("ديالهم", ("ديالهوم", "ديالهم")),
    ("خترعناها", ("اختارعناها", "اخترعناها", "خترعناها")),
    ("طوموبيل", ("الطوموبيل", "طوموبيل")),
    ("ميشليفن", ("مشليفن", "ميشليفن")),
    ("مريكان", ("المريكان", "ميريكان", "مريكان")),
    # Ambiguous in isolation (كلو can also be a verb); kept as a validated alias.
    ("كله", ("كلو", "كله")),
    # Article-loss form validated in the corpus; unsafe as a global ل -> ال rule.
    ("البلاد", ("لبلاد", "البلاد")),
)

_LOANWORD_ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mecanique", ("الميكانيك", "ميكانيك", "mecanique")),
    ("electronique", ("الالكترونيك", "الكترونيك", "electronique")),
    ("decor", ("ديكور", "decor")),
    ("risque", ("ريسك", "risque")),
    ("garage", ("كاراج", "garage")),
    ("client", ("كليان", "client")),
    ("salaire", ("الصالير", "صالير", "salaire")),
    ("fantasia", ("الفانتازيا", "فانتازيا", "fantasia")),
    ("phloem", ("phloeom", "phloem")),
    ("xylem", ("xyleme", "xylem")),
    ("berrad", ("brad", "berrad")),
    ("boraq", ("bourraque", "boraq")),
    ("classique", ("كلاسيكيه", "كلاسيكية", "classique")),
)

_PHRASE_ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cash flow", ("الكاش فلو", "cash flow")),
)


def _build_alias_map(
    groups: Sequence[tuple[str, Sequence[str]]],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, variants in groups:
        for variant in variants:
            aliases[variant] = canonical
    return aliases


_WORD_MAP = _build_alias_map(_WORD_ALIAS_GROUPS)
_LOANWORD_MAP = _build_alias_map(_LOANWORD_ALIAS_GROUPS)
_PHRASE_MAP = _build_alias_map(_PHRASE_ALIAS_GROUPS)

# Single-letter attached ف/ب are ambiguous: ففاس is a clitic construction,
# while فلكلوري is a lexical word. These roots therefore remain resource data,
# not a blanket regular expression. Add newly validated roots here rather than
# splitting every token that begins with ف or ب.
_FI_ATTACHED_ROOTS = {
    "اكادير",
    "جهه",
    "فاس",
    "المغرب",
    "ورزازات",
    "المدينه",
    "باطن",
    "عمليه",
    "لون",
    "سله",
    "ميه",
    "ميات",
    "حسابك",
    "جبال",
    "شمال",
    "صفرو",
    "قلعه",
    "وقت",
    "مجالات",
    "مشاكل",
    "فخ",
    "بزاف",
    "كاع",
    "اوروبا",
    "افق",
    "الصويره",
    "مشليفن",
    "ميشليفن",
    "داره",
    "الطوموبيل",
    "طوموبيل",
}

_BI_ATTACHED_ROOTS = {"سويسرا"}

# Relational words whose final Darija ـو is a masculine pronoun, not a plural
# verb ending. This cannot be applied to every word ending in و.
_RELATIONAL_O_TO_H_STEMS = {"من", "عند", "ديال"}


# Spoken-number normalization

# Standalone واحد/واحدة/وحدة are intentionally excluded because they can be a
# numeral, an indefinite article, or the noun "unit". They may be recognized in
# a future context-aware parser, but not by a blind word replacement.
_UNITS = {
    "صفر": 0,
    "جوج": 2,
    "اتنين": 2,
    "اثنين": 2,
    "تنين": 2,
    "تلاته": 3,
    "ثلاثه": 3,
    "اربعه": 4,
    "ربعه": 4,
    "خمسه": 5,
    "سته": 6,
    "سبعه": 7,
    "تمانيه": 8,
    "ثمانيه": 8,
    "تسعه": 9,
}

_TEENS = {
    "عشره": 10,
    "حداش": 11,
    "احداش": 11,
    "طناش": 12,
    "اتناش": 12,
    "تلتاش": 13,
    "ثلتاش": 13,
    "ربعتاش": 14,
    "اربعتاش": 14,
    "خمستاش": 15,
    "ستاش": 16,
    "سبعتاش": 17,
    "تمنطاش": 18,
    "ثمنطاش": 18,
    "تسعتاش": 19,
}

_TENS = {
    "عشرين": 20,
    "شرين": 20,
    "تلاتين": 30,
    "ثلاثين": 30,
    "ربعين": 40,
    "اربعين": 40,
    "خمسين": 50,
    "ستين": 60,
    "سبعين": 70,
    "تمانين": 80,
    "ثمانين": 80,
    "تسعين": 90,
}

_HUNDREDS = {
    "ميه": 100,
    "مئه": 100,
    "مايه": 100,
    "ميتين": 200,
    "ماتين": 200,
    "تلتميه": 300,
    "ثلاثميه": 300,
    "ربعمية": 400,
    "اربعمية": 400,
    "خمسمية": 500,
    "ستمية": 600,
    "سبعمية": 700,
    "تمنمية": 800,
    "ثمانمية": 800,
    "تسعمية": 900,
}

_SCALES = {
    "الف": 1_000,
    "الاف": 1_000,
    "مليون": 1_000_000,
    "ملايين": 1_000_000,
}

_NUMBER_WORDS = set(_UNITS) | set(_TEENS) | set(_TENS) | set(_HUNDREDS) | set(_SCALES)


# Internal helpers


def _strip_combining_marks(text: str) -> str:
    """Remove Arabic tashkeel and Latin accents using Unicode decomposition."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _apply_phrase_map(text: str) -> str:
    for source, target in sorted(
        _PHRASE_MAP.items(), key=lambda item: len(item[0]), reverse=True
    ):
        text = re.sub(rf"(?<!\S){re.escape(source)}(?!\S)", target, text)
    return text


def _normalize_pronoun_family(word: str) -> str:
    """Normalize productive ل/ب/مع pronoun families without listing each word."""
    match = re.fullmatch(r"([لب])ي(ا|ك|كي|ه|ها|نا|كم|هم)", word)
    if match:
        base, suffix = match.groups()
        return base + _L_B_PRONOUN_SUFFIXES[suffix]

    match = re.fullmatch(r"معا(ي|يا|ك|كي|ه|ها|نا|كم|هم)", word)
    if match:
        return "مع" + _MAA_PRONOUN_SUFFIXES[match.group(1)]

    return word


def _normalize_negation(tokens: Sequence[str]) -> list[str]:
    """Canonicalize productive attached Darija negation to ``ما <verb>``."""
    output: list[str] = []

    for token in tokens:
        previous = output[-1] if output else None
        if token in _NEGATION_EXCEPTIONS or previous == "ما":
            output.append(token)
            continue

        # ماكيقولش / مابقاش -> ما كيقولش / ما بقاش.
        if token.startswith("ما") and len(token) >= 5 and token.endswith("ش"):
            output.extend(["ما", token[2:]])
            continue

        # مكيقولش / مكتناقشوش / مكنعرفش -> ما كيقولش / ما كتناقشوش / ما كنعرفش.
        if token.startswith("م"):
            body = token[1:]
            matching_marker = next(
                (marker for marker in _ASPECT_MARKERS if body.startswith(marker)),
                None,
            )
            if matching_marker is not None:
                stem = body[len(matching_marker) :]
                if body.endswith("ش") or any(
                    stem.startswith(root) for root in _NEGATIVE_NO_SH_STEMS
                ):
                    output.extend(["ما", body])
                    continue

            # Bare productive forms: مباغيش / متبداش / ممستقراش / مبقتش.
            if len(token) >= 5 and body.endswith(_NEGATIVE_ENDINGS):
                output.extend(["ما", body])
                continue

        output.append(token)

    return output


def _join_split_aspect_markers(tokens: Sequence[str]) -> list[str]:
    """Join tokenization variants such as ``كي دار`` -> ``كيدار``."""
    output: list[str] = []
    index = 0

    while index < len(tokens):
        if (
            tokens[index] in _ASPECT_MARKERS
            and index + 1 < len(tokens)
            and ARABIC_TOKEN_RE.fullmatch(tokens[index + 1])
        ):
            output.append(tokens[index] + tokens[index + 1])
            index += 2
        else:
            output.append(tokens[index])
            index += 1

    return output


def _number_value(tokens: Sequence[str]) -> int | None:
    """Parse a contiguous non-ambiguous Darija number-word sequence."""
    total = 0
    current = 0
    seen = False

    for token in tokens:
        normalized = token

        if normalized == "و":
            continue

        if normalized.startswith("و") and normalized[1:] in _NUMBER_WORDS:
            normalized = normalized[1:]

        if normalized in _UNITS:
            current += _UNITS[normalized]
        elif normalized in _TEENS:
            current += _TEENS[normalized]
        elif normalized in _TENS:
            current += _TENS[normalized]
        elif normalized in _HUNDREDS:
            current += _HUNDREDS[normalized]
        elif normalized in _SCALES:
            scale = _SCALES[normalized]
            current = max(current, 1) * scale
            total += current
            current = 0
        else:
            return None

        seen = True

    return total + current if seen else None


def _replace_number_phrases(tokens: Sequence[str]) -> list[str]:
    """Replace longest recognized number-word sequences with ASCII digits."""
    output: list[str] = []
    index = 0

    while index < len(tokens):
        matched = False
        max_end = min(len(tokens), index + 9)

        for end in range(max_end, index, -1):
            candidate = tokens[index:end]

            # A bare list such as "20 30 50" must remain three values, not
            # be added into 100. Multi-token compounds require a scale
            # (جوج الاف) or an explicit/attached conjunction (ميتين وربعين).
            if len(candidate) > 1:
                has_scale = any(token in _SCALES for token in candidate)
                has_conjunction = any(
                    token == "و"
                    or (token.startswith("و") and token[1:] in _NUMBER_WORDS)
                    for token in candidate
                )
                if not has_scale and not has_conjunction:
                    continue

            value = _number_value(candidate)
            if value is not None:
                output.append(str(value))
                index = end
                matched = True
                break

        if not matched:
            output.append(tokens[index])
            index += 1

    return output


def _normalize_relational_o_pronoun(word: str) -> str:
    """Normalize منو/عندو/ديالو/كلو only for validated relational stems."""
    if not word.endswith("و"):
        return word

    stem = word[:-1]
    if stem in _RELATIONAL_O_TO_H_STEMS:
        return stem + "ه"

    return word


def _normalize_word(word: str) -> str:
    """Apply sequential lexical and productive word-level rules."""
    word = _WORD_MAP.get(word, word)
    word = _LOANWORD_MAP.get(word, word)

    if not ARABIC_TOKEN_RE.fullmatch(word):
        return word

    # الضوء / الضو and الماء / الما.
    word = re.sub(r"([اوي])ء$", r"\1", word)

    # Productive pronoun families.
    word = _normalize_pronoun_family(word)

    # كا + ن... aspect spelling: كانحاولو -> كنحاولو. Protect lexical
    # and past-tense forms such as كان / كانت / كانوا / كانون.
    if word.startswith("كان") and len(word) > 4 and word not in _KAN_PREFIX_EXCEPTIONS:
        word = "كن" + word[3:]

    # Long-vowel aspect-prefix spellings.
    for long_prefix, short_prefix in _PREFIX_RULES:
        if word.startswith(long_prefix) and len(word) > len(long_prefix):
            word = short_prefix + word[len(long_prefix) :]
            break

    # Optional alif in plural وا.
    if word.endswith("وا") and len(word) >= _MIN_LEN_WAW_ALEF:
        word = word[:-1]

    # Attached plural pronoun vowel: ديالهوم -> ديالهم.
    if word.endswith("هوم") and len(word) > 4:
        word = word[:-3] + "هم"

    word = _normalize_relational_o_pronoun(word)

    # Re-run aliases because productive rules may expose a canonical variant.
    word = _WORD_MAP.get(word, word)
    word = _LOANWORD_MAP.get(word, word)
    return word


# Public API


def normalize_arabic_asr(text: str | float | None) -> str:
    """Apply the fixed Moroccan/Darija ASR evaluation policy.

    Apply identically to reference and hypothesis. ``None`` and floating-point
    NaN return an empty string. Other inputs are converted to strings.
    """
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""

    stages = _normalization_stages(str(text))
    return stages[-1][1]


def _normalization_stages(text: str) -> list[tuple[str, str]]:
    """Run normalization and return every stage for debugging/tests."""
    stages: list[tuple[str, str]] = [("input", text)]

    normalized = text.lower()
    normalized = LANG_TAG_RE.sub(" ", normalized)
    normalized = normalized.replace("\u0640", "")
    normalized = _strip_combining_marks(normalized)
    normalized = normalized.replace("ى", "ا")
    stages.append(("unicode_cleanup", normalized))

    # The Arabic Unicode block also contains digits. Fold their glyphs first so
    # the script-boundary rules cannot split a number such as ٢٠١١ into 2 0 1 1.
    normalized = normalized.translate(DIGIT_TRANSLATION)

    # Separate script and digit boundaries that were written without spaces.
    normalized = re.sub(r"(?<=[A-Za-z0-9])(?=[\u0600-\u06FF])", " ", normalized)
    normalized = re.sub(r"(?<=[\u0600-\u06FF])(?=[A-Za-z0-9])", " ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z\u0600-\u06FF])(?=\d)", " ", normalized)
    normalized = re.sub(r"(?<=\d)(?=[A-Za-z\u0600-\u06FF])", " ", normalized)

    # Conservative orthography and fixed phonetic evaluation policy.
    normalized = re.sub(
        r"ة(?=\s|$|[،.؟!,?:;»«\"'؛\-–—…()\[\]{}\\/|_%+*=<>])",
        "ه",
        normalized,
    )
    normalized = re.sub(r"[ڭڬگ]", "ق", normalized)
    normalized = normalized.translate(PHONETIC_TRANSLATION)
    stages.append(("orthography", normalized))

    # Punctuation becomes spaces so words never merge.
    normalized = PUNCT_RE.sub(" ", normalized)
    normalized = " ".join(normalized.split())

    # In mixed-script text, Arabic لا before a Latin token is commonly a
    # transcription of the French article "la": لا mecatronique -> la mecatronique.
    # Restrict the rule to a following Latin token so Arabic negation stays intact.
    normalized = re.sub(r"(?<!\S)لا(?=\s+[a-z])", "la", normalized)
    normalized = _apply_phrase_map(normalized)
    stages.append(("punctuation_and_phrases", normalized))

    # Protect known lexical aliases before generic clitic processing.
    protected_tokens = [_WORD_MAP.get(token, token) for token in normalized.split()]
    normalized = " ".join(protected_tokens)

    for pattern, replacement in _CLITIC_RULES:
        normalized = pattern.sub(replacement, normalized)

    expanded: list[str] = []
    for token in normalized.split():
        if token in {"ف", "فال"}:
            expanded.append("في")
        elif token.startswith("ف") and token[1:] in _FI_ATTACHED_ROOTS:
            expanded.extend(["في", token[1:]])
        elif token.startswith("ب") and token[1:] in _BI_ATTACHED_ROOTS:
            expanded.extend(["ب", token[1:]])
        else:
            expanded.append(token)

    # Split productive attached و only when the remainder is recognizable as a
    # grammatical form, not for arbitrary words beginning with و.
    tokens: list[str] = []
    for token in expanded:
        if token.startswith("و") and len(token) > 2:
            remainder = token[1:]
            if (
                remainder.startswith(tuple(_ASPECT_MARKERS))
                or remainder.startswith("ما")
                or remainder in _WORD_MAP
                or remainder in _LOANWORD_MAP
            ):
                tokens.extend(["و", remainder])
                continue
        tokens.append(token)

    stages.append(("clitics", " ".join(tokens)))

    tokens = _normalize_negation(tokens)
    tokens = _join_split_aspect_markers(tokens)
    stages.append(("negation_and_aspect", " ".join(tokens)))

    result = [_normalize_word(word) for word in tokens]
    result = _replace_number_phrases(result)
    final_text = _apply_phrase_map(" ".join(result))
    final_text = " ".join(final_text.split())
    stages.append(("normalized", final_text))
    return stages


def trace_normalization(text: str | float | None) -> list[tuple[str, str]]:
    """Return all named stages used to normalize ``text``."""
    if text is None:
        return [("input", ""), ("normalized", "")]
    if isinstance(text, float) and math.isnan(text):
        return [("input", str(text)), ("normalized", "")]
    return _normalization_stages(str(text))


__all__ = ["normalize_arabic_asr", "trace_normalization"]
