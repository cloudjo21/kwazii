import re
import emoji

emojis = "".join(emoji.EMOJI_DATA.keys())


def get_codes(b_code, e_code):
    return [c for c in range(b_code, e_code + 1)]


def get_codes_from_ranges(block_ranges):
    return [get_codes(block[0], block[1]) for block in block_ranges]


latin_1_supplement = (0x0080, 0x00FF)
latin_extended_additional_block = (0x1E00, 0x1EFF)
latin_extended_a_block = (0x0100, 0x017F)
latin_extended_b_block = (0x0180, 0x024F)
latin_extended_c_block = (0x2C60, 0x2C7F)
latin_extended_d_block = (0xA720, 0xA7FF)

punctuations = (0x2000, 0x206F)
math_operators = (0x2200, 0x22FF)

cjk_symbols_and_puncs = (0x3000, 0x303F)
cjk_unified_ideographs_extension_a = (0x3400, 0x4DBF)
cjk_unified_ideographs = (0x4E00, 0x9FFF)
cjk_unified_ideographs_extension_b = (0x20000, 0x2A6DF)
cjk_unified_ideographs_extension_c = (0x2A700, 0x2B73F)
cjk_unified_ideographs_extension_d = (0x2B740, 0x2B81F)
cjk_unified_ideographs_extension_e = (0x2B820, 0x2CEAF)
cjk_unified_ideographs_extension_f = (0x2CEB0, 0x2EBEF)
cjk_unified_ideographs_supplement = (0x2F800, 0x2FA1F)
cjk_compatibility_ideographs = (0xF900, 0xFAFF)
cjk_radicals_supplement = (0x2E80, 0x2EFF)
kangxi_radicals = (0x2F00, 0x2FDF)

hiragana = (0x3040, 0x309F)
katakana = (0x30A0, 0x30FF)
katakana_phonetic = (0x31F0, 0x31FF)

misc_symbols = (0x2600, 0x26FF)
half_full_sized_char = (0xFF00, 0xFFEF)


target_block_range = [
    latin_1_supplement,
    latin_extended_additional_block,
    latin_extended_a_block,
    latin_extended_b_block,
    latin_extended_c_block,
    latin_extended_d_block,
    punctuations,
    math_operators,
    cjk_symbols_and_puncs,
    cjk_unified_ideographs_extension_a,
    cjk_unified_ideographs,
    cjk_unified_ideographs_extension_b,
    cjk_unified_ideographs_extension_c,
    cjk_unified_ideographs_extension_d,
    cjk_unified_ideographs_extension_e,
    cjk_unified_ideographs_extension_f,
    cjk_unified_ideographs_supplement,
    cjk_compatibility_ideographs,
    cjk_radicals_supplement,
    kangxi_radicals,
    hiragana,
    katakana,
    katakana_phonetic,
    misc_symbols,
    half_full_sized_char,
]

code_blocks = get_codes_from_ranges(target_block_range)

candi_char_codes = [c for block in code_blocks for c in block]

prohibit_char_codes = get_codes_from_ranges(
    [
        (0x0080, 0x00A0),
        (0x00AD, 0x00AD),
        (0x2000, 0x200F),
        (0x2028, 0x202F),
        (0x205F, 0x206F),
        (0x3000, 0x3000),
        (0x3002, 0x3002),
        (0x300C, 0x300D),
        (0x302A, 0x302F),
        (0xFF00, 0xFF0F),
        (0xFF1A, 0xFF1E),
        (0xFF20, 0xFF20),
        (0xFF3B, 0xFF40),
        (0xFF5B, 0xFF63),
        (0xFF9E, 0xFFEF),
        # symbols in latin_1_supplement
        (0x00A1, 0x00BF),
    ]
)


valid_char_codes = "".join(
    [
        chr(c)
        for c in candi_char_codes
        if not any([c in p_codes for p_codes in prohibit_char_codes])
    ]
)

pattern4korean = re.compile(
    f"[^ .,?!/@$%~％·∼()\x00-\x7fㄱ-ㅣ가-힣{emojis}{valid_char_codes}]+"
)
strict_pattern4korean = re.compile("[^\x00-\x7fㄱ-ㅣ가-힣]+")
# pattern4korean = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
url_pattern = re.compile(
    r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)

pattern4korean_only_alphabet = re.compile(
    f"[^\x20A-Za-zㄱ-ㅣ가-힣{emojis}{valid_char_codes}]+"
)


def preprocess_korean(text, strict=False):
    if strict is False:
        text = pattern4korean_only_alphabet.sub("", text)
    else:
        text = strict_pattern4korean.sub("", text)
    text = url_pattern.sub("", text)
    text = text.strip()
    return text
