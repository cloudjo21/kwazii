import re
import emoji

emojis = ''.join(emoji.EMOJI_DATA.keys())


def get_codes(b_code, e_code):
    return [c for c in range(b_code, e_code+1)]


def get_codes_from_ranges(block_ranges):
    return [get_codes(block[0], block[1]) for block in block_ranges]


latin_1_supplement = (0x0080, 0x00ff)
latin_extended_additional_block = (0x1e00, 0x1eff)
latin_extended_a_block = (0x0100, 0x017f)
latin_extended_b_block = (0x0180, 0x024f)
latin_extended_c_block = (0x2c60, 0x2c7f)
latin_extended_d_block = (0xa720, 0xa7ff)

punctuations = (0x2000, 0x206f)
math_operators = (0x2200, 0x22ff)

cjk_symbols_and_puncs = (0x3000, 0x303f)
cjk_unified_ideographs_extension_a = (0x3400, 0x4dbf)
cjk_unified_ideographs = (0x4e00, 0x9fff)
cjk_unified_ideographs_extension_b = (0x20000, 0x2a6df)
cjk_unified_ideographs_extension_c = (0x2a700, 0x2b73f)
cjk_unified_ideographs_extension_d = (0x2b740, 0x2b81f)
cjk_unified_ideographs_extension_e = (0x2b820, 0x2ceaf)
cjk_unified_ideographs_extension_f = (0x2ceb0, 0x2ebef)
cjk_unified_ideographs_supplement = (0x2f800, 0x2fa1f)
cjk_compatibility_ideographs = (0xf900, 0xfaff)
cjk_radicals_supplement = (0x2e80, 0x2eff)
kangxi_radicals = (0x2f00, 0x2fdf)

hiragana = (0x3040, 0x309f)
katakana = (0x30a0, 0x30ff)
katakana_phonetic = (0x31f0, 0x31ff)

misc_symbols = (0x2600, 0x26ff)
half_full_sized_char = (0xff00, 0xffef)


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
    half_full_sized_char
]

code_blocks = get_codes_from_ranges(target_block_range)

candi_char_codes = [c for block in code_blocks for c in block]

prohibit_char_codes = get_codes_from_ranges([
    (0x0080, 0x00a0),
    (0x00ad, 0x00ad),
    (0x2000, 0x200f),
    (0x2028, 0x202f),
    (0x205f, 0x206f),
    (0x3000, 0x3000),
    (0x3002, 0x3002),
    (0x300c, 0x300d),
    (0x302a, 0x302f),
    (0xff00, 0xff0f),
    (0xff1a, 0xff1e),
    (0xff20, 0xff20),
    (0xff3b, 0xff40),
    (0xff5b, 0xff63),
    (0xff9e, 0xffef),

    # symbols in latin_1_supplement
    (0x00a1, 0x00bf)
])


valid_char_codes = "".join([chr(c) for c in candi_char_codes if not any([c in p_codes for p_codes in prohibit_char_codes])])

pattern4korean = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}{valid_char_codes}]+')
strict_pattern4korean = re.compile(f'[^\x00-\x7Fㄱ-ㅣ가-힣]+')
# pattern4korean = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

pattern4korean_only_alphabet = re.compile(f'[^\x20A-Za-zㄱ-ㅣ가-힣{emojis}{valid_char_codes}]+')


def preprocess_korean(text, strict=False):
    if strict is False:
        text = pattern4korean_only_alphabet.sub('', text)
    else:
        text = strict_pattern4korean.sub('', text)
    text = url_pattern.sub('', text)
    text = text.strip()
    return text
