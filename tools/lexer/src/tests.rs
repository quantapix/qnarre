use super::*;

use expect_test::{expect, Expect};

fn check_raw_str(s: &str, expected: Result<u8, RawStrError>) {
    let s = &format!("r{}", s);
    let mut cursor = Cursor::new(s);
    cursor.bump();
    let res = cursor.raw_double_quoted_string(0);
    assert_eq!(res, expected);
}

#[test]
fn test_naked_raw_str() {
    check_raw_str(r#""abc""#, Ok(0));
}

#[test]
fn test_raw_no_start() {
    check_raw_str(r##""abc"#"##, Ok(0));
}

#[test]
fn test_too_many_terminators() {
    check_raw_str(r###"#"abc"##"###, Ok(1));
}

#[test]
fn test_unterminated() {
    check_raw_str(
        r#"#"abc"#,
        Err(RawStrError::NoTerminator {
            expected: 1,
            found: 0,
            possible_terminator_offset: None,
        }),
    );
    check_raw_str(
        r###"##"abc"#"###,
        Err(RawStrError::NoTerminator {
            expected: 2,
            found: 1,
            possible_terminator_offset: Some(7),
        }),
    );
    check_raw_str(
        r###"##"abc#"###,
        Err(RawStrError::NoTerminator {
            expected: 2,
            found: 0,
            possible_terminator_offset: None,
        }),
    )
}

#[test]
fn test_invalid_start() {
    check_raw_str(r##"#~"abc"#"##, Err(RawStrError::InvalidStarter { bad_char: '~' }));
}

#[test]
fn test_unterminated_no_pound() {
    check_raw_str(
        r#"""#,
        Err(RawStrError::NoTerminator {
            expected: 0,
            found: 0,
            possible_terminator_offset: None,
        }),
    );
}

#[test]
fn test_too_many_hashes() {
    let max_count = u8::MAX;
    let hashes1 = "#".repeat(max_count as usize);
    let hashes2 = "#".repeat(max_count as usize + 1);
    let middle = "\"abc\"";
    let s1 = [&hashes1, middle, &hashes1].join("");
    let s2 = [&hashes2, middle, &hashes2].join("");
    check_raw_str(&s1, Ok(255));
    check_raw_str(
        &s2,
        Err(RawStrError::TooManyDelimiters {
            found: u32::from(max_count) + 1,
        }),
    );
}

#[test]
fn test_valid_shebang() {
    let input = "#!/usr/bin/rustrun\nlet x = 5;";
    assert_eq!(strip_shebang(input), Some(18));
}

#[test]
fn test_invalid_shebang_valid_rust_syntax() {
    let input = "#!    [bad_attribute]";
    assert_eq!(strip_shebang(input), None);
}

#[test]
fn test_shebang_second_line() {
    let input = "\n#!/bin/bash";
    assert_eq!(strip_shebang(input), None);
}

#[test]
fn test_shebang_space() {
    let input = "#!    /bin/bash";
    assert_eq!(strip_shebang(input), Some(input.len()));
}

#[test]
fn test_shebang_empty_shebang() {
    let input = "#!    \n[attribute(foo)]";
    assert_eq!(strip_shebang(input), None);
}

#[test]
fn test_invalid_shebang_comment() {
    let input = "#!//bin/ami/a/comment\n[";
    assert_eq!(strip_shebang(input), None)
}

#[test]
fn test_invalid_shebang_another_comment() {
    let input = "#!/*bin/ami/a/comment*/\n[attribute";
    assert_eq!(strip_shebang(input), None)
}

#[test]
fn test_shebang_valid_rust_after() {
    let input = "#!/*bin/ami/a/comment*/\npub fn main() {}";
    assert_eq!(strip_shebang(input), Some(23))
}

#[test]
fn test_shebang_followed_by_attrib() {
    let input = "#!/bin/rust-scripts\n#![allow_unused(true)]";
    assert_eq!(strip_shebang(input), Some(19));
}

fn check_lexing(src: &str, expect: Expect) {
    let actual: String = tokenize(src).map(|token| format!("{:?}\n", token)).collect();
    expect.assert_eq(&actual)
}

#[test]
fn smoke_test() {
    check_lexing(
        "/* my source file */ fn main() { println!(\"zebra\"); }\n",
        expect![[r#"
            Token { kind: BlockComment { doc_style: None, terminated: true }, len: 20 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Ident, len: 2 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Ident, len: 4 }
            Token { kind: OpenParen, len: 1 }
            Token { kind: CloseParen, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: OpenBrace, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Ident, len: 7 }
            Token { kind: Bang, len: 1 }
            Token { kind: OpenParen, len: 1 }
            Token { kind: Literal { kind: Str { terminated: true }, suffix_start: 7 }, len: 7 }
            Token { kind: CloseParen, len: 1 }
            Token { kind: Semi, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: CloseBrace, len: 1 }
            Token { kind: Whitespace, len: 1 }
        "#]],
    )
}

#[test]
fn comment_flavors() {
    check_lexing(
        r"
// line
//// line as well
/// outer doc line
//! inner doc line
/* block */
/**/
/*** also block */
/** outer doc block */
/*! inner doc block */
",
        expect![[r#"
            Token { kind: Whitespace, len: 1 }
            Token { kind: LineComment { doc_style: None }, len: 7 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: LineComment { doc_style: None }, len: 17 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: LineComment { doc_style: Some(Outer) }, len: 18 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: LineComment { doc_style: Some(Inner) }, len: 18 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: BlockComment { doc_style: None, terminated: true }, len: 11 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: BlockComment { doc_style: None, terminated: true }, len: 4 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: BlockComment { doc_style: None, terminated: true }, len: 18 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: BlockComment { doc_style: Some(Outer), terminated: true }, len: 22 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: BlockComment { doc_style: Some(Inner), terminated: true }, len: 22 }
            Token { kind: Whitespace, len: 1 }
        "#]],
    )
}

#[test]
fn nested_block_comments() {
    check_lexing(
        "/* /* */ */'a'",
        expect![[r#"
            Token { kind: BlockComment { doc_style: None, terminated: true }, len: 11 }
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 3 }, len: 3 }
        "#]],
    )
}

#[test]
fn characters() {
    check_lexing(
        "'a' ' ' '\\n'",
        expect![[r#"
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 4 }, len: 4 }
        "#]],
    );
}

#[test]
fn lifetime() {
    check_lexing(
        "'abc",
        expect![[r#"
            Token { kind: Lifetime { starts_with_number: false }, len: 4 }
        "#]],
    );
}

#[test]
fn raw_string() {
    check_lexing(
        "r###\"\"#a\\b\x00c\"\"###",
        expect![[r#"
            Token { kind: Literal { kind: RawStr { n_hashes: Some(3) }, suffix_start: 17 }, len: 17 }
        "#]],
    )
}

#[test]
fn literal_suffixes() {
    check_lexing(
        r####"
'a'
b'a'
"a"
b"a"
1234
0b101
0xABC
1.0
1.0e10
2us
r###"raw"###suffix
br###"raw"###suffix
"####,
        expect![[r#"
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Byte { terminated: true }, suffix_start: 4 }, len: 4 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Str { terminated: true }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: ByteStr { terminated: true }, suffix_start: 4 }, len: 4 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 4 }, len: 4 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Binary, empty_int: false }, suffix_start: 5 }, len: 5 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Hexadecimal, empty_int: false }, suffix_start: 5 }, len: 5 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Float { base: Decimal, empty_exponent: false }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Float { base: Decimal, empty_exponent: false }, suffix_start: 6 }, len: 6 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: RawStr { n_hashes: Some(3) }, suffix_start: 12 }, len: 18 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: RawByteStr { n_hashes: Some(3) }, suffix_start: 13 }, len: 19 }
            Token { kind: Whitespace, len: 1 }
        "#]],
    )
}

mod unescape {
    use super::super::unescape::*;
    use std::ops::Range;

    #[test]
    fn test_unescape_char_bad() {
        fn check(literal_text: &str, expected_error: EscapeError) {
            assert_eq!(unescape_char(literal_text), Err(expected_error));
        }

        check("", EscapeError::ZeroChars);
        check(r"\", EscapeError::LoneSlash);

        check("\n", EscapeError::EscapeOnlyChar);
        check("\t", EscapeError::EscapeOnlyChar);
        check("'", EscapeError::EscapeOnlyChar);
        check("\r", EscapeError::BareCarriageReturn);

        check("spam", EscapeError::MoreThanOneChar);
        check(r"\x0ff", EscapeError::MoreThanOneChar);
        check(r#"\"a"#, EscapeError::MoreThanOneChar);
        check(r"\na", EscapeError::MoreThanOneChar);
        check(r"\ra", EscapeError::MoreThanOneChar);
        check(r"\ta", EscapeError::MoreThanOneChar);
        check(r"\\a", EscapeError::MoreThanOneChar);
        check(r"\'a", EscapeError::MoreThanOneChar);
        check(r"\0a", EscapeError::MoreThanOneChar);
        check(r"\u{0}x", EscapeError::MoreThanOneChar);
        check(r"\u{1F63b}}", EscapeError::MoreThanOneChar);

        check(r"\v", EscapeError::InvalidEscape);
        check(r"\💩", EscapeError::InvalidEscape);
        check(r"\●", EscapeError::InvalidEscape);
        check("\\\r", EscapeError::InvalidEscape);

        check(r"\x", EscapeError::TooShortHexEscape);
        check(r"\x0", EscapeError::TooShortHexEscape);
        check(r"\xf", EscapeError::TooShortHexEscape);
        check(r"\xa", EscapeError::TooShortHexEscape);
        check(r"\xx", EscapeError::InvalidCharInHexEscape);
        check(r"\xы", EscapeError::InvalidCharInHexEscape);
        check(r"\x🦀", EscapeError::InvalidCharInHexEscape);
        check(r"\xtt", EscapeError::InvalidCharInHexEscape);
        check(r"\xff", EscapeError::OutOfRangeHexEscape);
        check(r"\xFF", EscapeError::OutOfRangeHexEscape);
        check(r"\x80", EscapeError::OutOfRangeHexEscape);

        check(r"\u", EscapeError::NoBraceInUnicodeEscape);
        check(r"\u[0123]", EscapeError::NoBraceInUnicodeEscape);
        check(r"\u{0x}", EscapeError::InvalidCharInUnicodeEscape);
        check(r"\u{", EscapeError::UnclosedUnicodeEscape);
        check(r"\u{0000", EscapeError::UnclosedUnicodeEscape);
        check(r"\u{}", EscapeError::EmptyUnicodeEscape);
        check(r"\u{_0000}", EscapeError::LeadingUnderscoreUnicodeEscape);
        check(r"\u{0000000}", EscapeError::OverlongUnicodeEscape);
        check(r"\u{FFFFFF}", EscapeError::OutOfRangeUnicodeEscape);
        check(r"\u{ffffff}", EscapeError::OutOfRangeUnicodeEscape);
        check(r"\u{ffffff}", EscapeError::OutOfRangeUnicodeEscape);

        check(r"\u{DC00}", EscapeError::LoneSurrogateUnicodeEscape);
        check(r"\u{DDDD}", EscapeError::LoneSurrogateUnicodeEscape);
        check(r"\u{DFFF}", EscapeError::LoneSurrogateUnicodeEscape);

        check(r"\u{D800}", EscapeError::LoneSurrogateUnicodeEscape);
        check(r"\u{DAAA}", EscapeError::LoneSurrogateUnicodeEscape);
        check(r"\u{DBFF}", EscapeError::LoneSurrogateUnicodeEscape);
    }

    #[test]
    fn test_unescape_char_good() {
        fn check(literal_text: &str, expected_char: char) {
            assert_eq!(unescape_char(literal_text), Ok(expected_char));
        }

        check("a", 'a');
        check("ы", 'ы');
        check("🦀", '🦀');

        check(r#"\""#, '"');
        check(r"\n", '\n');
        check(r"\r", '\r');
        check(r"\t", '\t');
        check(r"\\", '\\');
        check(r"\'", '\'');
        check(r"\0", '\0');

        check(r"\x00", '\0');
        check(r"\x5a", 'Z');
        check(r"\x5A", 'Z');
        check(r"\x7f", 127 as char);

        check(r"\u{0}", '\0');
        check(r"\u{000000}", '\0');
        check(r"\u{41}", 'A');
        check(r"\u{0041}", 'A');
        check(r"\u{00_41}", 'A');
        check(r"\u{4__1__}", 'A');
        check(r"\u{1F63b}", '😻');
    }

    #[test]
    fn test_unescape_str_warn() {
        fn check(literal: &str, expected: &[(Range<usize>, Result<char, EscapeError>)]) {
            let mut unescaped = Vec::with_capacity(literal.len());
            unescape_literal(literal, Mode::Str, &mut |range, res| unescaped.push((range, res)));
            assert_eq!(unescaped, expected);
        }

        check("\\\n", &[]);
        check("\\\n ", &[]);

        check(
            "\\\n \u{a0} x",
            &[
                (0..5, Err(EscapeError::UnskippedWhitespaceWarning)),
                (3..5, Ok('\u{a0}')),
                (5..6, Ok(' ')),
                (6..7, Ok('x')),
            ],
        );
        check(
            "\\\n  \n  x",
            &[(0..7, Err(EscapeError::MultipleSkippedLinesWarning)), (7..8, Ok('x'))],
        );
    }

    #[test]
    fn test_unescape_str_good() {
        fn check(literal_text: &str, expected: &str) {
            let mut buf = Ok(String::with_capacity(literal_text.len()));
            unescape_literal(literal_text, Mode::Str, &mut |range, c| {
                if let Ok(b) = &mut buf {
                    match c {
                        Ok(c) => b.push(c),
                        Err(e) => buf = Err((range, e)),
                    }
                }
            });
            assert_eq!(buf.as_deref(), Ok(expected))
        }

        check("foo", "foo");
        check("", "");
        check(" \t\n", " \t\n");

        check("hello \\\n     world", "hello world");
        check("thread's", "thread's")
    }

    #[test]
    fn test_unescape_byte_bad() {
        fn check(literal_text: &str, expected_error: EscapeError) {
            assert_eq!(unescape_byte(literal_text), Err(expected_error));
        }

        check("", EscapeError::ZeroChars);
        check(r"\", EscapeError::LoneSlash);

        check("\n", EscapeError::EscapeOnlyChar);
        check("\t", EscapeError::EscapeOnlyChar);
        check("'", EscapeError::EscapeOnlyChar);
        check("\r", EscapeError::BareCarriageReturn);

        check("spam", EscapeError::MoreThanOneChar);
        check(r"\x0ff", EscapeError::MoreThanOneChar);
        check(r#"\"a"#, EscapeError::MoreThanOneChar);
        check(r"\na", EscapeError::MoreThanOneChar);
        check(r"\ra", EscapeError::MoreThanOneChar);
        check(r"\ta", EscapeError::MoreThanOneChar);
        check(r"\\a", EscapeError::MoreThanOneChar);
        check(r"\'a", EscapeError::MoreThanOneChar);
        check(r"\0a", EscapeError::MoreThanOneChar);

        check(r"\v", EscapeError::InvalidEscape);
        check(r"\💩", EscapeError::InvalidEscape);
        check(r"\●", EscapeError::InvalidEscape);

        check(r"\x", EscapeError::TooShortHexEscape);
        check(r"\x0", EscapeError::TooShortHexEscape);
        check(r"\xa", EscapeError::TooShortHexEscape);
        check(r"\xf", EscapeError::TooShortHexEscape);
        check(r"\xx", EscapeError::InvalidCharInHexEscape);
        check(r"\xы", EscapeError::InvalidCharInHexEscape);
        check(r"\x🦀", EscapeError::InvalidCharInHexEscape);
        check(r"\xtt", EscapeError::InvalidCharInHexEscape);

        check(r"\u", EscapeError::NoBraceInUnicodeEscape);
        check(r"\u[0123]", EscapeError::NoBraceInUnicodeEscape);
        check(r"\u{0x}", EscapeError::InvalidCharInUnicodeEscape);
        check(r"\u{", EscapeError::UnclosedUnicodeEscape);
        check(r"\u{0000", EscapeError::UnclosedUnicodeEscape);
        check(r"\u{}", EscapeError::EmptyUnicodeEscape);
        check(r"\u{_0000}", EscapeError::LeadingUnderscoreUnicodeEscape);
        check(r"\u{0000000}", EscapeError::OverlongUnicodeEscape);

        check("ы", EscapeError::NonAsciiCharInByte);
        check("🦀", EscapeError::NonAsciiCharInByte);

        check(r"\u{0}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{000000}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{41}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{0041}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{00_41}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{4__1__}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{1F63b}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{0}x", EscapeError::UnicodeEscapeInByte);
        check(r"\u{1F63b}}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{FFFFFF}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{ffffff}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{ffffff}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{DC00}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{DDDD}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{DFFF}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{D800}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{DAAA}", EscapeError::UnicodeEscapeInByte);
        check(r"\u{DBFF}", EscapeError::UnicodeEscapeInByte);
    }

    #[test]
    fn test_unescape_byte_good() {
        fn check(literal_text: &str, expected_byte: u8) {
            assert_eq!(unescape_byte(literal_text), Ok(expected_byte));
        }

        check("a", b'a');

        check(r#"\""#, b'"');
        check(r"\n", b'\n');
        check(r"\r", b'\r');
        check(r"\t", b'\t');
        check(r"\\", b'\\');
        check(r"\'", b'\'');
        check(r"\0", b'\0');

        check(r"\x00", b'\0');
        check(r"\x5a", b'Z');
        check(r"\x5A", b'Z');
        check(r"\x7f", 127);
        check(r"\x80", 128);
        check(r"\xff", 255);
        check(r"\xFF", 255);
    }

    #[test]
    fn test_unescape_byte_str_good() {
        fn check(literal_text: &str, expected: &[u8]) {
            let mut buf = Ok(Vec::with_capacity(literal_text.len()));
            unescape_literal(literal_text, Mode::ByteStr, &mut |range, c| {
                if let Ok(b) = &mut buf {
                    match c {
                        Ok(c) => b.push(byte_from_char(c)),
                        Err(e) => buf = Err((range, e)),
                    }
                }
            });
            assert_eq!(buf.as_deref(), Ok(expected))
        }

        check("foo", b"foo");
        check("", b"");
        check(" \t\n", b" \t\n");

        check("hello \\\n     world", b"hello world");
        check("thread's", b"thread's")
    }

    #[test]
    fn test_unescape_raw_str() {
        fn check(literal: &str, expected: &[(Range<usize>, Result<char, EscapeError>)]) {
            let mut unescaped = Vec::with_capacity(literal.len());
            unescape_literal(literal, Mode::RawStr, &mut |range, res| unescaped.push((range, res)));
            assert_eq!(unescaped, expected);
        }

        check("\r", &[(0..1, Err(EscapeError::BareCarriageReturnInRawString))]);
        check(
            "\rx",
            &[(0..1, Err(EscapeError::BareCarriageReturnInRawString)), (1..2, Ok('x'))],
        );
    }

    #[test]
    fn test_unescape_raw_byte_str() {
        fn check(literal: &str, expected: &[(Range<usize>, Result<char, EscapeError>)]) {
            let mut unescaped = Vec::with_capacity(literal.len());
            unescape_literal(literal, Mode::RawByteStr, &mut |range, res| {
                unescaped.push((range, res))
            });
            assert_eq!(unescaped, expected);
        }

        check("\r", &[(0..1, Err(EscapeError::BareCarriageReturnInRawString))]);
        check("🦀", &[(0..4, Err(EscapeError::NonAsciiCharInByte))]);
        check("🦀a", &[(0..4, Err(EscapeError::NonAsciiCharInByte)), (4..5, Ok('a'))]);
    }
}
