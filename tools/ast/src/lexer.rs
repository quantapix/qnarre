#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LiteralKind {
    Int { base: Base, empty_int: bool },
    Float { base: Base, empty_exponent: bool },
    Char { terminated: bool },
    Byte { terminated: bool },
    Str { terminated: bool },
    ByteStr { terminated: bool },
    CStr { terminated: bool },
    RawStr { n_hashes: Option<u8> },
    RawByteStr { n_hashes: Option<u8> },
    RawCStr { n_hashes: Option<u8> },
}
use LiteralKind::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    LineComment {
        doc_style: Option<DocStyle>,
    },
    BlockComment {
        doc_style: Option<DocStyle>,
        terminated: bool,
    },
    Whitespace,
    Ident,
    InvalidIdent,
    RawIdent,
    UnknownPrefix,
    Literal {
        kind: LiteralKind,
        suffix_start: u32,
    },
    Lifetime {
        starts_with_number: bool,
    },
    Semi,
    Comma,
    Dot,
    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    OpenBracket,
    CloseBracket,
    At,
    Pound,
    Tilde,
    Question,
    Colon,
    Dollar,
    Eq,
    Bang,
    Lt,
    Gt,
    Minus,
    And,
    Or,
    Plus,
    Star,
    Slash,
    Caret,
    Percent,
    Unknown,
    Eof,
}
use TokenKind::*;

#[derive(Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub len: u32,
}
impl Token {
    fn new(kind: TokenKind, len: u32) -> Token {
        Token { kind, len }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocStyle {
    Outer,
    Inner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RawStrError {
    InvalidStarter {
        bad_char: char,
    },
    NoTerminator {
        expected: u32,
        found: u32,
        possible_terminator_offset: Option<u32>,
    },
    TooManyDelimiters {
        found: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Base {
    Binary = 2,
    Octal = 8,
    Decimal = 10,
    Hexadecimal = 16,
}

pub fn strip_shebang(input: &str) -> Option<usize> {
    if let Some(input_tail) = input.strip_prefix("#!") {
        let next_non_whitespace_token = tokenize(input_tail).map(|tok| tok.kind).find(|tok| {
            !matches!(
                tok,
                TokenKind::Whitespace
                    | TokenKind::LineComment { doc_style: None }
                    | TokenKind::BlockComment { doc_style: None, .. }
            )
        });
        if next_non_whitespace_token != Some(TokenKind::OpenBracket) {
            return Some(2 + input_tail.lines().next().unwrap_or_default().len());
        }
    }
    None
}
#[inline]
pub fn validate_raw_str(input: &str, prefix_len: u32) -> Result<(), RawStrError> {
    debug_assert!(!input.is_empty());
    let mut cursor = Cursor::new(input);
    for _ in 0..prefix_len {
        cursor.bump().unwrap();
    }
    cursor.raw_double_quoted_string(prefix_len).map(|_| ())
}
pub fn tokenize(input: &str) -> impl Iterator<Item = Token> + '_ {
    let mut cursor = Cursor::new(input);
    std::iter::from_fn(move || {
        let token = cursor.advance_token();
        if token.kind != TokenKind::Eof {
            Some(token)
        } else {
            None
        }
    })
}
pub fn is_whitespace(c: char) -> bool {
    matches!(
        c,
        // Usual ASCII suspects
        '\u{0009}'   // \t
        | '\u{000A}' // \n
        | '\u{000B}' // vertical tab
        | '\u{000C}' // form feed
        | '\u{000D}' // \r
        | '\u{0020}' // space
        // NEXT LINE from latin1
        | '\u{0085}'
        // Bidi markers
        | '\u{200E}' // LEFT-TO-RIGHT MARK
        | '\u{200F}' // RIGHT-TO-LEFT MARK
        // Dedicated whitespace characters from Unicode
        | '\u{2028}' // LINE SEPARATOR
        | '\u{2029}' // PARAGRAPH SEPARATOR
    )
}
pub fn is_id_start(c: char) -> bool {
    c == '_' || unicode_xid::UnicodeXID::is_xid_start(c)
}
pub fn is_id_continue(c: char) -> bool {
    unicode_xid::UnicodeXID::is_xid_continue(c)
}
pub fn is_ident(string: &str) -> bool {
    let mut chars = string.chars();
    if let Some(start) = chars.next() {
        is_id_start(start) && chars.all(is_id_continue)
    } else {
        false
    }
}

mod cursor {
    use std::str::Chars;

    pub const EOF_CHAR: char = '\0';

    pub struct Cursor<'a> {
        len_remaining: usize,
        chars: Chars<'a>,
        #[cfg(debug_assertions)]
        prev: char,
    }
    impl<'a> Cursor<'a> {
        pub fn new(input: &'a str) -> Cursor<'a> {
            Cursor {
                len_remaining: input.len(),
                chars: input.chars(),
                #[cfg(debug_assertions)]
                prev: EOF_CHAR,
            }
        }
        pub fn prev(&self) -> char {
            #[cfg(debug_assertions)]
            {
                self.prev
            }
            #[cfg(not(debug_assertions))]
            {
                EOF_CHAR
            }
        }
        pub fn first(&self) -> char {
            self.chars.clone().next().unwrap_or(EOF_CHAR)
        }
        pub fn second(&self) -> char {
            let mut iter = self.chars.clone();
            iter.next();
            iter.next().unwrap_or(EOF_CHAR)
        }
        pub fn is_eof(&self) -> bool {
            self.chars.as_str().is_empty()
        }
        pub fn pos_within_token(&self) -> u32 {
            (self.len_remaining - self.chars.as_str().len()) as u32
        }
        pub fn reset_pos_within_token(&mut self) {
            self.len_remaining = self.chars.as_str().len();
        }
        pub fn bump(&mut self) -> Option<char> {
            let c = self.chars.next()?;
            #[cfg(debug_assertions)]
            {
                self.prev = c;
            }
            Some(c)
        }
        pub fn eat_while(&mut self, mut predicate: impl FnMut(char) -> bool) {
            while predicate(self.first()) && !self.is_eof() {
                self.bump();
            }
        }
    }
}
pub use cursor::{Cursor, EOF_CHAR};

impl Cursor<'_> {
    pub fn advance_token(&mut self) -> Token {
        let first_char = match self.bump() {
            Some(c) => c,
            None => return Token::new(TokenKind::Eof, 0),
        };
        let token_kind = match first_char {
            '/' => match self.first() {
                '/' => self.line_comment(),
                '*' => self.block_comment(),
                _ => Slash,
            },
            c if is_whitespace(c) => self.whitespace(),
            'r' => match (self.first(), self.second()) {
                ('#', c1) if is_id_start(c1) => self.raw_ident(),
                ('#', _) | ('"', _) => {
                    let res = self.raw_double_quoted_string(1);
                    let suffix_start = self.pos_within_token();
                    if res.is_ok() {
                        self.eat_literal_suffix();
                    }
                    let kind = RawStr { n_hashes: res.ok() };
                    Literal { kind, suffix_start }
                },
                _ => self.ident_or_unknown_prefix(),
            },
            'b' => self.c_or_byte_string(
                |terminated| ByteStr { terminated },
                |n_hashes| RawByteStr { n_hashes },
                Some(|terminated| Byte { terminated }),
            ),
            'c' => self.c_or_byte_string(|terminated| CStr { terminated }, |n_hashes| RawCStr { n_hashes }, None),
            c if is_id_start(c) => self.ident_or_unknown_prefix(),
            c @ '0'..='9' => {
                let literal_kind = self.number(c);
                let suffix_start = self.pos_within_token();
                self.eat_literal_suffix();
                TokenKind::Literal {
                    kind: literal_kind,
                    suffix_start,
                }
            },
            ';' => Semi,
            ',' => Comma,
            '.' => Dot,
            '(' => OpenParen,
            ')' => CloseParen,
            '{' => OpenBrace,
            '}' => CloseBrace,
            '[' => OpenBracket,
            ']' => CloseBracket,
            '@' => At,
            '#' => Pound,
            '~' => Tilde,
            '?' => Question,
            ':' => Colon,
            '$' => Dollar,
            '=' => Eq,
            '!' => Bang,
            '<' => Lt,
            '>' => Gt,
            '-' => Minus,
            '&' => And,
            '|' => Or,
            '+' => Plus,
            '*' => Star,
            '^' => Caret,
            '%' => Percent,
            '\'' => self.lifetime_or_char(),
            '"' => {
                let terminated = self.double_quoted_string();
                let suffix_start = self.pos_within_token();
                if terminated {
                    self.eat_literal_suffix();
                }
                let kind = Str { terminated };
                Literal { kind, suffix_start }
            },
            c if !c.is_ascii() && unic_emoji_char::is_emoji(c) => self.fake_ident_or_unknown_prefix(),
            _ => Unknown,
        };
        let res = Token::new(token_kind, self.pos_within_token());
        self.reset_pos_within_token();
        res
    }
    fn line_comment(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '/' && self.first() == '/');
        self.bump();
        let doc_style = match self.first() {
            '!' => Some(DocStyle::Inner),
            '/' if self.second() != '/' => Some(DocStyle::Outer),
            _ => None,
        };
        self.eat_while(|c| c != '\n');
        LineComment { doc_style }
    }
    fn block_comment(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '/' && self.first() == '*');
        self.bump();
        let doc_style = match self.first() {
            '!' => Some(DocStyle::Inner),
            '*' if !matches!(self.second(), '*' | '/') => Some(DocStyle::Outer),
            _ => None,
        };
        let mut depth = 1usize;
        while let Some(c) = self.bump() {
            match c {
                '/' if self.first() == '*' => {
                    self.bump();
                    depth += 1;
                },
                '*' if self.first() == '/' => {
                    self.bump();
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                },
                _ => (),
            }
        }
        BlockComment {
            doc_style,
            terminated: depth == 0,
        }
    }
    fn whitespace(&mut self) -> TokenKind {
        debug_assert!(is_whitespace(self.prev()));
        self.eat_while(is_whitespace);
        Whitespace
    }
    fn raw_ident(&mut self) -> TokenKind {
        debug_assert!(self.prev() == 'r' && self.first() == '#' && is_id_start(self.second()));
        self.bump();
        self.eat_identifier();
        RawIdent
    }
    fn ident_or_unknown_prefix(&mut self) -> TokenKind {
        debug_assert!(is_id_start(self.prev()));
        self.eat_while(is_id_continue);
        match self.first() {
            '#' | '"' | '\'' => UnknownPrefix,
            c if !c.is_ascii() && unic_emoji_char::is_emoji(c) => self.fake_ident_or_unknown_prefix(),
            _ => Ident,
        }
    }
    fn fake_ident_or_unknown_prefix(&mut self) -> TokenKind {
        self.eat_while(|c| {
            unicode_xid::UnicodeXID::is_xid_continue(c)
                || (!c.is_ascii() && unic_emoji_char::is_emoji(c))
                || c == '\u{200d}'
        });
        match self.first() {
            '#' | '"' | '\'' => UnknownPrefix,
            _ => InvalidIdent,
        }
    }
    fn c_or_byte_string(
        &mut self,
        mk_kind: impl FnOnce(bool) -> LiteralKind,
        mk_kind_raw: impl FnOnce(Option<u8>) -> LiteralKind,
        single_quoted: Option<fn(bool) -> LiteralKind>,
    ) -> TokenKind {
        match (self.first(), self.second(), single_quoted) {
            ('\'', _, Some(mk_kind)) => {
                self.bump();
                let terminated = self.single_quoted_string();
                let suffix_start = self.pos_within_token();
                if terminated {
                    self.eat_literal_suffix();
                }
                let kind = mk_kind(terminated);
                Literal { kind, suffix_start }
            },
            ('"', _, _) => {
                self.bump();
                let terminated = self.double_quoted_string();
                let suffix_start = self.pos_within_token();
                if terminated {
                    self.eat_literal_suffix();
                }
                let kind = mk_kind(terminated);
                Literal { kind, suffix_start }
            },
            ('r', '"', _) | ('r', '#', _) => {
                self.bump();
                let res = self.raw_double_quoted_string(2);
                let suffix_start = self.pos_within_token();
                if res.is_ok() {
                    self.eat_literal_suffix();
                }
                let kind = mk_kind_raw(res.ok());
                Literal { kind, suffix_start }
            },
            _ => self.ident_or_unknown_prefix(),
        }
    }
    fn number(&mut self, first_digit: char) -> LiteralKind {
        debug_assert!('0' <= self.prev() && self.prev() <= '9');
        let mut base = Base::Decimal;
        if first_digit == '0' {
            match self.first() {
                'b' => {
                    base = Base::Binary;
                    self.bump();
                    if !self.eat_decimal_digits() {
                        return Int { base, empty_int: true };
                    }
                },
                'o' => {
                    base = Base::Octal;
                    self.bump();
                    if !self.eat_decimal_digits() {
                        return Int { base, empty_int: true };
                    }
                },
                'x' => {
                    base = Base::Hexadecimal;
                    self.bump();
                    if !self.eat_hexadecimal_digits() {
                        return Int { base, empty_int: true };
                    }
                },
                '0'..='9' | '_' => {
                    self.eat_decimal_digits();
                },
                '.' | 'e' | 'E' => {},
                _ => return Int { base, empty_int: false },
            }
        } else {
            self.eat_decimal_digits();
        };
        match self.first() {
            '.' if self.second() != '.' && !is_id_start(self.second()) => {
                self.bump();
                let mut empty_exponent = false;
                if self.first().is_ascii_digit() {
                    self.eat_decimal_digits();
                    match self.first() {
                        'e' | 'E' => {
                            self.bump();
                            empty_exponent = !self.eat_float_exponent();
                        },
                        _ => (),
                    }
                }
                Float { base, empty_exponent }
            },
            'e' | 'E' => {
                self.bump();
                let empty_exponent = !self.eat_float_exponent();
                Float { base, empty_exponent }
            },
            _ => Int { base, empty_int: false },
        }
    }
    fn lifetime_or_char(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '\'');
        let can_be_a_lifetime = if self.second() == '\'' {
            false
        } else {
            is_id_start(self.first()) || self.first().is_ascii_digit()
        };
        if !can_be_a_lifetime {
            let terminated = self.single_quoted_string();
            let suffix_start = self.pos_within_token();
            if terminated {
                self.eat_literal_suffix();
            }
            let kind = Char { terminated };
            return Literal { kind, suffix_start };
        }
        let starts_with_number = self.first().is_ascii_digit();
        self.bump();
        self.eat_while(is_id_continue);
        if self.first() == '\'' {
            self.bump();
            let kind = Char { terminated: true };
            Literal {
                kind,
                suffix_start: self.pos_within_token(),
            }
        } else {
            Lifetime { starts_with_number }
        }
    }
    fn single_quoted_string(&mut self) -> bool {
        debug_assert!(self.prev() == '\'');
        if self.second() == '\'' && self.first() != '\\' {
            self.bump();
            self.bump();
            return true;
        }
        loop {
            match self.first() {
                '\'' => {
                    self.bump();
                    return true;
                },
                '/' => break,
                '\n' if self.second() != '\'' => break,
                EOF_CHAR if self.is_eof() => break,
                '\\' => {
                    self.bump();
                    self.bump();
                },
                _ => {
                    self.bump();
                },
            }
        }
        false
    }
    fn double_quoted_string(&mut self) -> bool {
        debug_assert!(self.prev() == '"');
        while let Some(c) = self.bump() {
            match c {
                '"' => {
                    return true;
                },
                '\\' if self.first() == '\\' || self.first() == '"' => {
                    self.bump();
                },
                _ => (),
            }
        }
        false
    }
    fn raw_double_quoted_string(&mut self, prefix_len: u32) -> Result<u8, RawStrError> {
        let n_hashes = self.raw_string_unvalidated(prefix_len)?;
        match u8::try_from(n_hashes) {
            Ok(num) => Ok(num),
            Err(_) => Err(RawStrError::TooManyDelimiters { found: n_hashes }),
        }
    }
    fn raw_string_unvalidated(&mut self, prefix_len: u32) -> Result<u32, RawStrError> {
        debug_assert!(self.prev() == 'r');
        let start_pos = self.pos_within_token();
        let mut possible_terminator_offset = None;
        let mut max_hashes = 0;
        let mut eaten = 0;
        while self.first() == '#' {
            eaten += 1;
            self.bump();
        }
        let n_start_hashes = eaten;
        match self.bump() {
            Some('"') => (),
            c => {
                let c = c.unwrap_or(EOF_CHAR);
                return Err(RawStrError::InvalidStarter { bad_char: c });
            },
        }
        loop {
            self.eat_while(|c| c != '"');
            if self.is_eof() {
                return Err(RawStrError::NoTerminator {
                    expected: n_start_hashes,
                    found: max_hashes,
                    possible_terminator_offset,
                });
            }
            self.bump();
            let mut n_end_hashes = 0;
            while self.first() == '#' && n_end_hashes < n_start_hashes {
                n_end_hashes += 1;
                self.bump();
            }
            if n_end_hashes == n_start_hashes {
                return Ok(n_start_hashes);
            } else if n_end_hashes > max_hashes {
                possible_terminator_offset = Some(self.pos_within_token() - start_pos - n_end_hashes + prefix_len);
                max_hashes = n_end_hashes;
            }
        }
    }
    fn eat_decimal_digits(&mut self) -> bool {
        let mut has_digits = false;
        loop {
            match self.first() {
                '_' => {
                    self.bump();
                },
                '0'..='9' => {
                    has_digits = true;
                    self.bump();
                },
                _ => break,
            }
        }
        has_digits
    }
    fn eat_hexadecimal_digits(&mut self) -> bool {
        let mut has_digits = false;
        loop {
            match self.first() {
                '_' => {
                    self.bump();
                },
                '0'..='9' | 'a'..='f' | 'A'..='F' => {
                    has_digits = true;
                    self.bump();
                },
                _ => break,
            }
        }
        has_digits
    }
    fn eat_float_exponent(&mut self) -> bool {
        debug_assert!(self.prev() == 'e' || self.prev() == 'E');
        if self.first() == '-' || self.first() == '+' {
            self.bump();
        }
        self.eat_decimal_digits()
    }
    fn eat_literal_suffix(&mut self) {
        self.eat_identifier();
    }
    fn eat_identifier(&mut self) {
        if !is_id_start(self.first()) {
            return;
        }
        self.bump();
        self.eat_while(is_id_continue);
    }
}

pub mod unescape {
    use std::ops::Range;
    use std::str::Chars;

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
            Mode::RawStr | Mode::RawByteStr => {
                unescape_raw_str_or_raw_byte_str(src, mode == Mode::RawByteStr, callback)
            },
            Mode::CStr | Mode::RawCStr => unreachable!(),
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

    #[inline]
    pub fn byte_from_char(c: char) -> u8 {
        let res = c as u32;
        debug_assert!(res <= u8::MAX as u32, "guaranteed because of Mode::ByteStr");
        res as u8
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
    fn is_ascii(x: u32) -> bool {
        x <= 0x7F
    }
}

#[cfg(test)]
mod tests {
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
}
