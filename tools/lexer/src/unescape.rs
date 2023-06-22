use std::ops::Range;
use std::str::Chars;

#[cfg(test)]
mod tests;

#[derive(Debug, PartialEq, Eq)]
pub enum EscapeError {
    ZeroChars,
    MoreThanOneChar,
    LoneSlash,
    InvalidEscape,
    BareCarriageReturn,
    BareCarriageReturnInRawString,
    EscapeOnlyChar,
    TooShortHexEscape,
    InvalidCharInHexEscape,
    OutOfRangeHexEscape,
    NoBraceInUnicodeEscape,
    InvalidCharInUnicodeEscape,
    EmptyUnicodeEscape,
    UnclosedUnicodeEscape,
    LeadingUnderscoreUnicodeEscape,
    OverlongUnicodeEscape,
    LoneSurrogateUnicodeEscape,
    OutOfRangeUnicodeEscape,
    UnicodeEscapeInByte,
    NonAsciiCharInByte,
    UnskippedWhitespaceWarning,
    MultipleSkippedLinesWarning,
}

impl EscapeError {
    pub fn is_fatal(&self) -> bool {
        !matches!(
            self,
            EscapeError::UnskippedWhitespaceWarning | EscapeError::MultipleSkippedLinesWarning
        )
    }
}

pub fn unescape_literal<F>(src: &str, mode: Mode, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<char, EscapeError>),
{
    match mode {
        Mode::Char | Mode::Byte => {
            let mut chars = src.chars();
            let res = unescape_char_or_byte(&mut chars, mode == Mode::Byte);
            callback(0..(src.len() - chars.as_str().len()), res);
        },
        Mode::Str | Mode::ByteStr => unescape_str_common(src, mode, callback),
        Mode::RawStr | Mode::RawByteStr => unescape_raw_str_or_raw_byte_str(src, mode == Mode::RawByteStr, callback),
        Mode::CStr | Mode::RawCStr => unreachable!(),
    }
}

pub enum CStrUnit {
    Byte(u8),
    Char(char),
}
impl From<u8> for CStrUnit {
    fn from(value: u8) -> Self {
        CStrUnit::Byte(value)
    }
}
impl From<char> for CStrUnit {
    fn from(value: char) -> Self {
        CStrUnit::Char(value)
    }
}

pub fn unescape_c_string<F>(src: &str, mode: Mode, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<CStrUnit, EscapeError>),
{
    if mode == Mode::RawCStr {
        unescape_raw_str_or_raw_byte_str(src, mode.characters_should_be_ascii(), &mut |r, result| {
            callback(r, result.map(CStrUnit::Char))
        });
    } else {
        unescape_str_common(src, mode, callback);
    }
}

pub fn unescape_char(src: &str) -> Result<char, EscapeError> {
    unescape_char_or_byte(&mut src.chars(), false)
}

pub fn unescape_byte(src: &str) -> Result<u8, EscapeError> {
    unescape_char_or_byte(&mut src.chars(), true).map(byte_from_char)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Char,
    Str,
    Byte,
    ByteStr,
    RawStr,
    RawByteStr,
    CStr,
    RawCStr,
}
impl Mode {
    pub fn in_double_quotes(self) -> bool {
        match self {
            Mode::Str | Mode::ByteStr | Mode::RawStr | Mode::RawByteStr | Mode::CStr | Mode::RawCStr => true,
            Mode::Char | Mode::Byte => false,
        }
    }
    pub fn ascii_escapes_should_be_ascii(self) -> bool {
        match self {
            Mode::Char | Mode::Str | Mode::RawStr => true,
            Mode::Byte | Mode::ByteStr | Mode::RawByteStr | Mode::CStr | Mode::RawCStr => false,
        }
    }
    pub fn characters_should_be_ascii(self) -> bool {
        match self {
            Mode::Byte | Mode::ByteStr | Mode::RawByteStr => true,
            Mode::Char | Mode::Str | Mode::RawStr | Mode::CStr | Mode::RawCStr => false,
        }
    }
    pub fn is_unicode_escape_disallowed(self) -> bool {
        match self {
            Mode::Byte | Mode::ByteStr | Mode::RawByteStr => true,
            Mode::Char | Mode::Str | Mode::RawStr | Mode::CStr | Mode::RawCStr => false,
        }
    }
    pub fn prefix_noraw(self) -> &'static str {
        match self {
            Mode::Byte | Mode::ByteStr | Mode::RawByteStr => "b",
            Mode::CStr | Mode::RawCStr => "c",
            Mode::Char | Mode::Str | Mode::RawStr => "",
        }
    }
}

fn scan_escape<T: From<u8> + From<char>>(chars: &mut Chars<'_>, mode: Mode) -> Result<T, EscapeError> {
    let res = match chars.next().ok_or(EscapeError::LoneSlash)? {
        '"' => b'"',
        'n' => b'\n',
        'r' => b'\r',
        't' => b'\t',
        '\\' => b'\\',
        '\'' => b'\'',
        '0' => b'\0',
        'x' => {
            let hi = chars.next().ok_or(EscapeError::TooShortHexEscape)?;
            let hi = hi.to_digit(16).ok_or(EscapeError::InvalidCharInHexEscape)?;
            let lo = chars.next().ok_or(EscapeError::TooShortHexEscape)?;
            let lo = lo.to_digit(16).ok_or(EscapeError::InvalidCharInHexEscape)?;
            let value = hi * 16 + lo;
            if mode.ascii_escapes_should_be_ascii() && !is_ascii(value) {
                return Err(EscapeError::OutOfRangeHexEscape);
            }
            value as u8
        },
        'u' => return scan_unicode(chars, mode.is_unicode_escape_disallowed()).map(Into::into),
        _ => return Err(EscapeError::InvalidEscape),
    };
    Ok(res.into())
}

fn scan_unicode(chars: &mut Chars<'_>, is_unicode_escape_disallowed: bool) -> Result<char, EscapeError> {
    if chars.next() != Some('{') {
        return Err(EscapeError::NoBraceInUnicodeEscape);
    }
    let mut n_digits = 1;
    let mut value: u32 = match chars.next().ok_or(EscapeError::UnclosedUnicodeEscape)? {
        '_' => return Err(EscapeError::LeadingUnderscoreUnicodeEscape),
        '}' => return Err(EscapeError::EmptyUnicodeEscape),
        c => c.to_digit(16).ok_or(EscapeError::InvalidCharInUnicodeEscape)?,
    };
    loop {
        match chars.next() {
            None => return Err(EscapeError::UnclosedUnicodeEscape),
            Some('_') => continue,
            Some('}') => {
                if n_digits > 6 {
                    return Err(EscapeError::OverlongUnicodeEscape);
                }
                if is_unicode_escape_disallowed {
                    return Err(EscapeError::UnicodeEscapeInByte);
                }
                break std::char::from_u32(value).ok_or({
                    if value > 0x10FFFF {
                        EscapeError::OutOfRangeUnicodeEscape
                    } else {
                        EscapeError::LoneSurrogateUnicodeEscape
                    }
                });
            },
            Some(c) => {
                let digit: u32 = c.to_digit(16).ok_or(EscapeError::InvalidCharInUnicodeEscape)?;
                n_digits += 1;
                if n_digits > 6 {
                    continue;
                }
                value = value * 16 + digit;
            },
        };
    }
}

#[inline]
fn ascii_check(c: char, characters_should_be_ascii: bool) -> Result<char, EscapeError> {
    if characters_should_be_ascii && !c.is_ascii() {
        Err(EscapeError::NonAsciiCharInByte)
    } else {
        Ok(c)
    }
}

fn unescape_char_or_byte(chars: &mut Chars<'_>, is_byte: bool) -> Result<char, EscapeError> {
    let c = chars.next().ok_or(EscapeError::ZeroChars)?;
    let res = match c {
        '\\' => scan_escape(chars, if is_byte { Mode::Byte } else { Mode::Char }),
        '\n' | '\t' | '\'' => Err(EscapeError::EscapeOnlyChar),
        '\r' => Err(EscapeError::BareCarriageReturn),
        _ => ascii_check(c, is_byte),
    }?;
    if chars.next().is_some() {
        return Err(EscapeError::MoreThanOneChar);
    }
    Ok(res)
}

fn unescape_str_common<F, T: From<u8> + From<char>>(src: &str, mode: Mode, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<T, EscapeError>),
{
    let mut chars = src.chars();
    while let Some(c) = chars.next() {
        let start = src.len() - chars.as_str().len() - c.len_utf8();
        let res = match c {
            '\\' => match chars.clone().next() {
                Some('\n') => {
                    skip_ascii_whitespace(&mut chars, start, &mut |range, err| callback(range, Err(err)));
                    continue;
                },
                _ => scan_escape::<T>(&mut chars, mode),
            },
            '\n' => Ok(b'\n'.into()),
            '\t' => Ok(b'\t'.into()),
            '"' => Err(EscapeError::EscapeOnlyChar),
            '\r' => Err(EscapeError::BareCarriageReturn),
            _ => ascii_check(c, mode.characters_should_be_ascii()).map(Into::into),
        };
        let end = src.len() - chars.as_str().len();
        callback(start..end, res.map(Into::into));
    }
}

fn skip_ascii_whitespace<F>(chars: &mut Chars<'_>, start: usize, callback: &mut F)
where
    F: FnMut(Range<usize>, EscapeError),
{
    let tail = chars.as_str();
    let first_non_space = tail
        .bytes()
        .position(|b| b != b' ' && b != b'\t' && b != b'\n' && b != b'\r')
        .unwrap_or(tail.len());
    if tail[1..first_non_space].contains('\n') {
        let end = start + first_non_space + 1;
        callback(start..end, EscapeError::MultipleSkippedLinesWarning);
    }
    let tail = &tail[first_non_space..];
    if let Some(c) = tail.chars().next() {
        if c.is_whitespace() {
            let end = start + first_non_space + c.len_utf8() + 1;
            callback(start..end, EscapeError::UnskippedWhitespaceWarning);
        }
    }
    *chars = tail.chars();
}

fn unescape_raw_str_or_raw_byte_str<F>(src: &str, is_byte: bool, callback: &mut F)
where
    F: FnMut(Range<usize>, Result<char, EscapeError>),
{
    let mut chars = src.chars();
    while let Some(c) = chars.next() {
        let start = src.len() - chars.as_str().len() - c.len_utf8();
        let res = match c {
            '\r' => Err(EscapeError::BareCarriageReturnInRawString),
            _ => ascii_check(c, is_byte),
        };
        let end = src.len() - chars.as_str().len();
        callback(start..end, res);
    }
}

#[inline]
pub fn byte_from_char(c: char) -> u8 {
    let res = c as u32;
    debug_assert!(res <= u8::MAX as u32, "guaranteed because of Mode::ByteStr");
    res as u8
}

fn is_ascii(x: u32) -> bool {
    x <= 0x7F
}
