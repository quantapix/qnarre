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
mod tests;
