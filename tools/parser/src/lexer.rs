use std::str::Chars;
use unic_emoji_char::is_emoji;
use unicode_xid::UnicodeXID;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LitKind {
    Byte { terminated: bool },
    Char { terminated: bool },
    Int { base: Base, empty: bool },
    Float { base: Base, empty: bool },
    ByteStr { terminated: bool },
    CStr { terminated: bool },
    Str { terminated: bool },
    RawByteStr { n_hashes: Option<u8> },
    RawCStr { n_hashes: Option<u8> },
    RawStr { n_hashes: Option<u8> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokKind {
    LineComment { style: Option<DocStyle> },
    BlockComment { style: Option<DocStyle>, terminated: bool },
    Whitespace,
    Ident,
    InvalidIdent,
    RawIdent,
    UnknownPrefix,
    Lit { kind: LitKind, suff_start: u32 },
    Lifetime { starts_with_num: bool },
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

#[derive(Debug)]
pub struct Token {
    pub kind: TokKind,
    pub len: u32,
}
impl Token {
    fn new(kind: TokKind, len: u32) -> Token {
        Token { kind, len }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocStyle {
    Outer,
    Inner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RawStrErr {
    InvalidStart {
        bad: char,
    },
    NoTerminator {
        expected: u32,
        found: u32,
        possible_offset: Option<u32>,
    },
    ManyDelimiters {
        found: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Base {
    Bin = 2,
    Oct = 8,
    Dec = 10,
    Hex = 16,
}

pub fn strip_shebang(x: &str) -> Option<usize> {
    if let Some(y) = x.strip_prefix("#!") {
        use TokKind::*;
        let next = tokenize(y).map(|x| x.kind).find(|x| {
            !matches!(
                x,
                Whitespace | LineComment { style: None } | BlockComment { style: None, .. }
            )
        });
        if next != Some(OpenBracket) {
            return Some(2 + y.lines().next().unwrap_or_default().len());
        }
    }
    None
}
#[inline]
pub fn validate_raw_str(x: &str, pre_len: u32) -> Result<(), RawStrErr> {
    debug_assert!(!x.is_empty());
    let mut c = Cursor::new(x);
    for _ in 0..pre_len {
        c.bump().unwrap();
    }
    c.raw_dbl_quoted_str(pre_len).map(|_| ())
}
pub fn tokenize(x: &str) -> impl Iterator<Item = Token> + '_ {
    let mut c = Cursor::new(x);
    std::iter::from_fn(move || {
        let y = c.advance_tok();
        if y.kind != TokKind::Eof {
            Some(y)
        } else {
            None
        }
    })
}
pub fn is_whitespace(x: char) -> bool {
    matches!(
        x,
        '\u{0009}'   // \t
        | '\u{000A}' // \n
        | '\u{000B}' // vertical tab
        | '\u{000C}' // form feed
        | '\u{000D}' // \r
        | '\u{0020}' // space
        | '\u{0085}' // NEXT LINE from latin1
        | '\u{200E}' // LEFT-TO-RIGHT MARK
        | '\u{200F}' // RIGHT-TO-LEFT MARK
        | '\u{2028}' // LINE SEPARATOR
        | '\u{2029}' // PARAGRAPH SEPARATOR
    )
}
pub fn is_id_start(x: char) -> bool {
    x == '_' || UnicodeXID::is_xid_start(x)
}
pub fn is_id_cont(x: char) -> bool {
    UnicodeXID::is_xid_continue(x)
}
pub fn is_ident(x: &str) -> bool {
    let mut xs = x.chars();
    if let Some(x) = xs.next() {
        is_id_start(x) && xs.all(is_id_cont)
    } else {
        false
    }
}

pub const EOF_CHAR: char = '\0';

pub struct Cursor<'a> {
    n_rest: usize,
    chars: Chars<'a>,
    #[cfg(debug_assertions)]
    prev: char,
}
impl<'a> Cursor<'a> {
    pub fn new(x: &'a str) -> Cursor<'a> {
        Cursor {
            n_rest: x.len(),
            chars: x.chars(),
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
        let mut it = self.chars.clone();
        it.next();
        it.next().unwrap_or(EOF_CHAR)
    }
    pub fn is_eof(&self) -> bool {
        self.chars.as_str().is_empty()
    }
    pub fn pos_in_tok(&self) -> u32 {
        (self.n_rest - self.chars.as_str().len()) as u32
    }
    pub fn reset_pos_in_tok(&mut self) {
        self.n_rest = self.chars.as_str().len();
    }
    pub fn bump(&mut self) -> Option<char> {
        let y = self.chars.next()?;
        #[cfg(debug_assertions)]
        {
            self.prev = y;
        }
        Some(y)
    }
    pub fn eat_while(&mut self, mut pred: impl FnMut(char) -> bool) {
        while pred(self.first()) && !self.is_eof() {
            self.bump();
        }
    }
}
impl Cursor<'_> {
    pub fn advance_tok(&mut self) -> Token {
        use LitKind::*;
        use TokKind::*;
        let first = match self.bump() {
            Some(x) => x,
            None => return Token::new(Eof, 0),
        };
        let y = match first {
            '/' => match self.first() {
                '/' => self.line_comment(),
                '*' => self.block_comment(),
                _ => Slash,
            },
            x if is_whitespace(x) => self.whitespace(),
            'r' => match (self.first(), self.second()) {
                ('#', x) if is_id_start(x) => self.raw_ident(),
                ('#', _) | ('"', _) => {
                    let y = self.raw_dbl_quoted_str(1);
                    let suff_start = self.pos_in_tok();
                    if y.is_ok() {
                        self.eat_lit_suff();
                    }
                    let kind = RawStr { n_hashes: y.ok() };
                    Lit { kind, suff_start }
                },
                _ => self.ident_or_unknown_pre(),
            },
            'b' => self.c_or_byte_str(
                |terminated| ByteStr { terminated },
                |n_hashes| RawByteStr { n_hashes },
                Some(|terminated| Byte { terminated }),
            ),
            'c' => self.c_or_byte_str(|terminated| CStr { terminated }, |n_hashes| RawCStr { n_hashes }, None),
            x if is_id_start(x) => self.ident_or_unknown_pre(),
            x @ '0'..='9' => {
                let kind = self.number(x);
                let suff_start = self.pos_in_tok();
                self.eat_lit_suff();
                Lit { kind, suff_start }
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
                let terminated = self.dbl_quoted_str();
                let suff_start = self.pos_in_tok();
                if terminated {
                    self.eat_lit_suff();
                }
                let kind = Str { terminated };
                Lit { kind, suff_start }
            },
            x if !x.is_ascii() && is_emoji(x) => self.fake_or_unknown_pre(),
            _ => Unknown,
        };
        let y = Token::new(y, self.pos_in_tok());
        self.reset_pos_in_tok();
        y
    }
    fn line_comment(&mut self) -> TokKind {
        debug_assert!(self.prev() == '/' && self.first() == '/');
        self.bump();
        use DocStyle::*;
        let style = match self.first() {
            '!' => Some(Inner),
            '/' if self.second() != '/' => Some(Outer),
            _ => None,
        };
        self.eat_while(|x| x != '\n');
        TokKind::LineComment { style }
    }
    fn block_comment(&mut self) -> TokKind {
        debug_assert!(self.prev() == '/' && self.first() == '*');
        self.bump();
        use DocStyle::*;
        let style = match self.first() {
            '!' => Some(Inner),
            '*' if !matches!(self.second(), '*' | '/') => Some(Outer),
            _ => None,
        };
        let mut depth = 1usize;
        while let Some(x) = self.bump() {
            match x {
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
        TokKind::BlockComment {
            style,
            terminated: depth == 0,
        }
    }
    fn whitespace(&mut self) -> TokKind {
        debug_assert!(is_whitespace(self.prev()));
        self.eat_while(is_whitespace);
        TokKind::Whitespace
    }
    fn raw_ident(&mut self) -> TokKind {
        debug_assert!(self.prev() == 'r' && self.first() == '#' && is_id_start(self.second()));
        self.bump();
        self.eat_ident();
        TokKind::RawIdent
    }
    fn ident_or_unknown_pre(&mut self) -> TokKind {
        debug_assert!(is_id_start(self.prev()));
        self.eat_while(is_id_cont);
        use TokKind::*;
        match self.first() {
            '#' | '"' | '\'' => UnknownPrefix,
            x if !x.is_ascii() && is_emoji(x) => self.fake_or_unknown_pre(),
            _ => Ident,
        }
    }
    fn fake_or_unknown_pre(&mut self) -> TokKind {
        self.eat_while(|x| UnicodeXID::is_xid_continue(x) || (!x.is_ascii() && is_emoji(x)) || x == '\u{200d}');
        use TokKind::*;
        match self.first() {
            '#' | '"' | '\'' => UnknownPrefix,
            _ => InvalidIdent,
        }
    }
    fn c_or_byte_str(
        &mut self,
        f_kind: impl FnOnce(bool) -> LitKind,
        f_raw: impl FnOnce(Option<u8>) -> LitKind,
        single_quoted: Option<fn(bool) -> LitKind>,
    ) -> TokKind {
        use TokKind::*;
        match (self.first(), self.second(), single_quoted) {
            ('\'', _, Some(f_kind)) => {
                self.bump();
                let terminated = self.single_quoted_str();
                let suff_start = self.pos_in_tok();
                if terminated {
                    self.eat_lit_suff();
                }
                let kind = f_kind(terminated);
                Lit { kind, suff_start }
            },
            ('"', _, _) => {
                self.bump();
                let terminated = self.dbl_quoted_str();
                let suff_start = self.pos_in_tok();
                if terminated {
                    self.eat_lit_suff();
                }
                let kind = f_kind(terminated);
                Lit { kind, suff_start }
            },
            ('r', '"', _) | ('r', '#', _) => {
                self.bump();
                let y = self.raw_dbl_quoted_str(2);
                let suff_start = self.pos_in_tok();
                if y.is_ok() {
                    self.eat_lit_suff();
                }
                let kind = f_raw(y.ok());
                Lit { kind, suff_start }
            },
            _ => self.ident_or_unknown_pre(),
        }
    }
    fn number(&mut self, first: char) -> LitKind {
        debug_assert!('0' <= self.prev() && self.prev() <= '9');
        use Base::*;
        use LitKind::*;
        let mut base = Dec;
        if first == '0' {
            match self.first() {
                'b' => {
                    base = Bin;
                    self.bump();
                    if !self.eat_dec_digits() {
                        return Int { base, empty: true };
                    }
                },
                'o' => {
                    base = Oct;
                    self.bump();
                    if !self.eat_dec_digits() {
                        return Int { base, empty: true };
                    }
                },
                'x' => {
                    base = Hex;
                    self.bump();
                    if !self.eat_hex_digits() {
                        return Int { base, empty: true };
                    }
                },
                '0'..='9' | '_' => {
                    self.eat_dec_digits();
                },
                '.' | 'e' | 'E' => {},
                _ => return Int { base, empty: false },
            }
        } else {
            self.eat_dec_digits();
        };
        match self.first() {
            '.' if self.second() != '.' && !is_id_start(self.second()) => {
                self.bump();
                let mut empty = false;
                if self.first().is_ascii_digit() {
                    self.eat_dec_digits();
                    match self.first() {
                        'e' | 'E' => {
                            self.bump();
                            empty = !self.eat_float_exp();
                        },
                        _ => (),
                    }
                }
                Float { base, empty }
            },
            'e' | 'E' => {
                self.bump();
                let empty = !self.eat_float_exp();
                Float { base, empty }
            },
            _ => Int { base, empty: false },
        }
    }
    fn lifetime_or_char(&mut self) -> TokKind {
        debug_assert!(self.prev() == '\'');
        use LitKind::*;
        use TokKind::*;
        let lifetime = if self.second() == '\'' {
            false
        } else {
            is_id_start(self.first()) || self.first().is_ascii_digit()
        };
        if !lifetime {
            let terminated = self.single_quoted_str();
            let suff_start = self.pos_in_tok();
            if terminated {
                self.eat_lit_suff();
            }
            let kind = Char { terminated };
            return Lit { kind, suff_start };
        }
        let starts_with_num = self.first().is_ascii_digit();
        self.bump();
        self.eat_while(is_id_cont);
        if self.first() == '\'' {
            self.bump();
            let kind = Char { terminated: true };
            Lit {
                kind,
                suff_start: self.pos_in_tok(),
            }
        } else {
            Lifetime { starts_with_num }
        }
    }
    fn single_quoted_str(&mut self) -> bool {
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
    fn dbl_quoted_str(&mut self) -> bool {
        debug_assert!(self.prev() == '"');
        while let Some(x) = self.bump() {
            match x {
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
    fn raw_dbl_quoted_str(&mut self, pre_len: u32) -> Result<u8, RawStrErr> {
        let found = self.raw_str_unvalidated(pre_len)?;
        match u8::try_from(found) {
            Ok(x) => Ok(x),
            Err(_) => Err(RawStrErr::ManyDelimiters { found }),
        }
    }
    fn raw_str_unvalidated(&mut self, pre_len: u32) -> Result<u32, RawStrErr> {
        debug_assert!(self.prev() == 'r');
        let start_pos = self.pos_in_tok();
        let mut possible_offset = None;
        let mut found = 0;
        let mut eaten = 0;
        while self.first() == '#' {
            eaten += 1;
            self.bump();
        }
        let expected = eaten;
        match self.bump() {
            Some('"') => (),
            x => {
                let bad = x.unwrap_or(EOF_CHAR);
                return Err(RawStrErr::InvalidStart { bad });
            },
        }
        loop {
            self.eat_while(|x| x != '"');
            if self.is_eof() {
                return Err(RawStrErr::NoTerminator {
                    expected,
                    found,
                    possible_offset,
                });
            }
            self.bump();
            let mut y = 0;
            while self.first() == '#' && y < expected {
                y += 1;
                self.bump();
            }
            if y == expected {
                return Ok(expected);
            } else if y > found {
                possible_offset = Some(self.pos_in_tok() - start_pos - y + pre_len);
                found = y;
            }
        }
    }
    fn eat_dec_digits(&mut self) -> bool {
        let mut y = false;
        loop {
            match self.first() {
                '_' => {
                    self.bump();
                },
                '0'..='9' => {
                    y = true;
                    self.bump();
                },
                _ => break,
            }
        }
        y
    }
    fn eat_hex_digits(&mut self) -> bool {
        let mut y = false;
        loop {
            match self.first() {
                '_' => {
                    self.bump();
                },
                '0'..='9' | 'a'..='f' | 'A'..='F' => {
                    y = true;
                    self.bump();
                },
                _ => break,
            }
        }
        y
    }
    fn eat_float_exp(&mut self) -> bool {
        debug_assert!(self.prev() == 'e' || self.prev() == 'E');
        if self.first() == '-' || self.first() == '+' {
            self.bump();
        }
        self.eat_dec_digits()
    }
    fn eat_lit_suff(&mut self) {
        self.eat_ident();
    }
    fn eat_ident(&mut self) {
        if !is_id_start(self.first()) {
            return;
        }
        self.bump();
        self.eat_while(is_id_cont);
    }
}

pub mod unescape {
    use std::{ops::Range, str::Chars};

    pub enum CStrUnit {
        Byte(u8),
        Char(char),
    }
    impl From<u8> for CStrUnit {
        fn from(x: u8) -> Self {
            CStrUnit::Byte(x)
        }
    }
    impl From<char> for CStrUnit {
        fn from(x: char) -> Self {
            CStrUnit::Char(x)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Mode {
        Byte,
        Char,
        ByteStr,
        CStr,
        Str,
        RawByteStr,
        RawCStr,
        RawStr,
    }
    impl Mode {
        pub fn in_dbl_quotes(self) -> bool {
            use Mode::*;
            match self {
                Str | ByteStr | RawStr | RawByteStr | CStr | RawCStr => true,
                Char | Byte => false,
            }
        }
        pub fn ascii_escapes_should_be_ascii(self) -> bool {
            use Mode::*;
            match self {
                Char | Str | RawStr => true,
                Byte | ByteStr | RawByteStr | CStr | RawCStr => false,
            }
        }
        pub fn chars_should_be_ascii(self) -> bool {
            use Mode::*;
            match self {
                Byte | ByteStr | RawByteStr => true,
                Char | Str | RawStr | CStr | RawCStr => false,
            }
        }
        pub fn is_unicode_escape_disallowed(self) -> bool {
            use Mode::*;
            match self {
                Byte | ByteStr | RawByteStr => true,
                Char | Str | RawStr | CStr | RawCStr => false,
            }
        }
        pub fn prefix_noraw(self) -> &'static str {
            use Mode::*;
            match self {
                Byte | ByteStr | RawByteStr => "b",
                CStr | RawCStr => "c",
                Char | Str | RawStr => "",
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    pub enum EscErr {
        ZeroChars,
        ManyChars,
        OneSlash,
        InvalidEsc,
        BareCarriageReturn,
        CarriageReturnInRaw,
        EscOnlyChar,
        ShortHexEsc,
        InvalidInHexEsc,
        OutOfRangeHexEsc,
        NoBraceInUniEsc,
        InvalidInUniEsc,
        EmptyUniEsc,
        UnclosedUniEsc,
        UnderscoreUniEsc,
        LongUniEsc,
        LoneSurrogateUniEsc,
        OutOfRangeUniEsc,
        UniEscInByte,
        NonAsciiInByte,
        UnskippedWhitespace,
        ManySkippedLines,
    }
    impl EscErr {
        pub fn is_fatal(&self) -> bool {
            use EscErr::*;
            !matches!(self, UnskippedWhitespace | ManySkippedLines)
        }
    }

    pub fn unesc_lit<F>(x: &str, m: Mode, cb: &mut F)
    where
        F: FnMut(Range<usize>, Result<char, EscErr>),
    {
        use Mode::*;
        match m {
            Char | Byte => {
                let mut xs = x.chars();
                let y = unescape_char_or_byte(&mut xs, m == Byte);
                cb(0..(x.len() - xs.as_str().len()), y);
            },
            Str | ByteStr => unescape_str_common(x, m, cb),
            RawStr | RawByteStr => unescape_raw_str_or_byte_str(x, m == RawByteStr, cb),
            CStr | RawCStr => unreachable!(),
        }
    }
    pub fn unesc_c_str<F>(x: &str, m: Mode, cb: &mut F)
    where
        F: FnMut(Range<usize>, Result<CStrUnit, EscErr>),
    {
        if m == Mode::RawCStr {
            unescape_raw_str_or_byte_str(x, m.chars_should_be_ascii(), &mut |r, result| {
                cb(r, result.map(CStrUnit::Char))
            });
        } else {
            unescape_str_common(x, m, cb);
        }
    }
    pub fn unesc_char(x: &str) -> Result<char, EscErr> {
        unescape_char_or_byte(&mut x.chars(), false)
    }
    pub fn unesc_byte(x: &str) -> Result<u8, EscErr> {
        unescape_char_or_byte(&mut x.chars(), true).map(byte_from_char)
    }

    #[inline]
    pub fn byte_from_char(x: char) -> u8 {
        let y = x as u32;
        debug_assert!(y <= u8::MAX as u32, "guaranteed because of Mode::ByteStr");
        y as u8
    }

    fn scan_esc<T: From<u8> + From<char>>(x: &mut Chars<'_>, m: Mode) -> Result<T, EscErr> {
        use EscErr::*;
        let y = match x.next().ok_or(OneSlash)? {
            '"' => b'"',
            'n' => b'\n',
            'r' => b'\r',
            't' => b'\t',
            '\\' => b'\\',
            '\'' => b'\'',
            '0' => b'\0',
            'x' => {
                let hi = x.next().ok_or(ShortHexEsc)?;
                let hi = hi.to_digit(16).ok_or(InvalidInHexEsc)?;
                let lo = x.next().ok_or(ShortHexEsc)?;
                let lo = lo.to_digit(16).ok_or(InvalidInHexEsc)?;
                let y = hi * 16 + lo;
                if m.ascii_escapes_should_be_ascii() && !is_ascii(y) {
                    return Err(OutOfRangeHexEsc);
                }
                y as u8
            },
            'u' => return scan_uni(x, m.is_unicode_escape_disallowed()).map(Into::into),
            _ => return Err(InvalidEsc),
        };
        Ok(y.into())
    }
    fn scan_uni(x: &mut Chars<'_>, no_uni_esc: bool) -> Result<char, EscErr> {
        use EscErr::*;
        if x.next() != Some('{') {
            return Err(NoBraceInUniEsc);
        }
        let mut n = 1;
        let mut y: u32 = match x.next().ok_or(UnclosedUniEsc)? {
            '_' => return Err(UnderscoreUniEsc),
            '}' => return Err(EmptyUniEsc),
            x => x.to_digit(16).ok_or(InvalidInUniEsc)?,
        };
        loop {
            match x.next() {
                None => return Err(UnclosedUniEsc),
                Some('_') => continue,
                Some('}') => {
                    if n > 6 {
                        return Err(LongUniEsc);
                    }
                    if no_uni_esc {
                        return Err(UniEscInByte);
                    }
                    break std::char::from_u32(y).ok_or({
                        if y > 0x10FFFF {
                            OutOfRangeUniEsc
                        } else {
                            LoneSurrogateUniEsc
                        }
                    });
                },
                Some(x) => {
                    let x: u32 = x.to_digit(16).ok_or(InvalidInUniEsc)?;
                    n += 1;
                    if n > 6 {
                        continue;
                    }
                    y = y * 16 + x;
                },
            };
        }
    }
    #[inline]
    fn ascii_check(x: char, ascii: bool) -> Result<char, EscErr> {
        if ascii && !x.is_ascii() {
            Err(EscErr::NonAsciiInByte)
        } else {
            Ok(x)
        }
    }
    fn unescape_char_or_byte(x: &mut Chars<'_>, is_byte: bool) -> Result<char, EscErr> {
        let c = x.next().ok_or(EscErr::ZeroChars)?;
        use Mode::*;
        let y = match c {
            '\\' => scan_esc(x, if is_byte { Byte } else { Char }),
            '\n' | '\t' | '\'' => Err(EscErr::EscOnlyChar),
            '\r' => Err(EscErr::BareCarriageReturn),
            _ => ascii_check(c, is_byte),
        }?;
        if x.next().is_some() {
            return Err(EscErr::ManyChars);
        }
        Ok(y)
    }
    fn unescape_str_common<F, T: From<u8> + From<char>>(x: &str, m: Mode, cb: &mut F)
    where
        F: FnMut(Range<usize>, Result<T, EscErr>),
    {
        let mut xs = x.chars();
        use EscErr::*;
        while let Some(c) = xs.next() {
            let beg = x.len() - xs.as_str().len() - c.len_utf8();
            let y = match c {
                '\\' => match xs.clone().next() {
                    Some('\n') => {
                        skip_ascii_whitespace(&mut xs, beg, &mut |x, e| cb(x, Err(e)));
                        continue;
                    },
                    _ => scan_esc::<T>(&mut xs, m),
                },
                '\n' => Ok(b'\n'.into()),
                '\t' => Ok(b'\t'.into()),
                '"' => Err(EscOnlyChar),
                '\r' => Err(BareCarriageReturn),
                _ => ascii_check(c, m.chars_should_be_ascii()).map(Into::into),
            };
            let end = x.len() - xs.as_str().len();
            cb(beg..end, y.map(Into::into));
        }
    }
    fn skip_ascii_whitespace<F>(x: &mut Chars<'_>, beg: usize, cb: &mut F)
    where
        F: FnMut(Range<usize>, EscErr),
    {
        let tail = x.as_str();
        let first_non_space = tail
            .bytes()
            .position(|x| x != b' ' && x != b'\t' && x != b'\n' && x != b'\r')
            .unwrap_or(tail.len());
        use EscErr::*;
        if tail[1..first_non_space].contains('\n') {
            let end = beg + first_non_space + 1;
            cb(beg..end, ManySkippedLines);
        }
        let tail = &tail[first_non_space..];
        if let Some(c) = tail.chars().next() {
            if c.is_whitespace() {
                let end = beg + first_non_space + c.len_utf8() + 1;
                cb(beg..end, UnskippedWhitespace);
            }
        }
        *x = tail.chars();
    }
    fn unescape_raw_str_or_byte_str<F>(x: &str, is_byte: bool, cb: &mut F)
    where
        F: FnMut(Range<usize>, Result<char, EscErr>),
    {
        let mut xs = x.chars();
        while let Some(c) = xs.next() {
            let beg = x.len() - xs.as_str().len() - c.len_utf8();
            let y = match c {
                '\r' => Err(EscErr::CarriageReturnInRaw),
                _ => ascii_check(c, is_byte),
            };
            let end = x.len() - xs.as_str().len();
            cb(beg..end, y);
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

    fn check_raw_str(s: &str, expected: Result<u8, RawStrErr>) {
        let s = &format!("r{}", s);
        let mut cursor = Cursor::new(s);
        cursor.bump();
        let res = cursor.raw_dbl_quoted_str(0);
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
            Err(RawStrErr::NoTerminator {
                expected: 1,
                found: 0,
                possible_offset: None,
            }),
        );
        check_raw_str(
            r###"##"abc"#"###,
            Err(RawStrErr::NoTerminator {
                expected: 2,
                found: 1,
                possible_offset: Some(7),
            }),
        );
        check_raw_str(
            r###"##"abc#"###,
            Err(RawStrErr::NoTerminator {
                expected: 2,
                found: 0,
                possible_offset: None,
            }),
        )
    }
    #[test]
    fn test_invalid_start() {
        check_raw_str(r##"#~"abc"#"##, Err(RawStrErr::InvalidStart { bad: '~' }));
    }
    #[test]
    fn test_unterminated_no_pound() {
        check_raw_str(
            r#"""#,
            Err(RawStrErr::NoTerminator {
                expected: 0,
                found: 0,
                possible_offset: None,
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
            Err(RawStrErr::ManyDelimiters {
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
                Token { kind: Literal { kind: Str { terminated: true }, suff_start: 7 }, len: 7 }
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
                Token { kind: Literal { kind: Char { terminated: true }, suff_start: 3 }, len: 3 }
            "#]],
        )
    }
    #[test]
    fn characters() {
        check_lexing(
            "'a' ' ' '\\n'",
            expect![[r#"
                Token { kind: Literal { kind: Char { terminated: true }, suff_start: 3 }, len: 3 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: Char { terminated: true }, suff_start: 3 }, len: 3 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: Char { terminated: true }, suff_start: 4 }, len: 4 }
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
                Token { kind: Literal { kind: RawStr { n_hashes: Some(3) }, suff_start: 17 }, len: 17 }
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
                Token { kind: Literal { kind: Char { terminated: true }, suff_start: 3 }, len: 3 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: Byte { terminated: true }, suff_start: 4 }, len: 4 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: Str { terminated: true }, suff_start: 3 }, len: 3 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: ByteStr { terminated: true }, suff_start: 4 }, len: 4 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: Int { base: Decimal, empty: false }, suff_start: 4 }, len: 4 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: Int { base: Binary, empty: false }, suff_start: 5 }, len: 5 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: Int { base: Hexadecimal, empty: false }, suff_start: 5 }, len: 5 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: Float { base: Decimal, empty: false }, suff_start: 3 }, len: 3 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: Float { base: Decimal, empty: false }, suff_start: 6 }, len: 6 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: Int { base: Decimal, empty: false }, suff_start: 1 }, len: 3 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: RawStr { n_hashes: Some(3) }, suff_start: 12 }, len: 18 }
                Token { kind: Whitespace, len: 1 }
                Token { kind: Literal { kind: RawByteStr { n_hashes: Some(3) }, suff_start: 13 }, len: 19 }
                Token { kind: Whitespace, len: 1 }
            "#]],
        )
    }

    mod unescape {
        use super::super::unescape::*;
        use std::ops::Range;

        #[test]
        fn test_unescape_char_bad() {
            fn check(literal_text: &str, expected_error: EscErr) {
                assert_eq!(unesc_char(literal_text), Err(expected_error));
            }
            check("", EscErr::ZeroChars);
            check(r"\", EscErr::OneSlash);
            check("\n", EscErr::EscOnlyChar);
            check("\t", EscErr::EscOnlyChar);
            check("'", EscErr::EscOnlyChar);
            check("\r", EscErr::BareCarriageReturn);
            check("spam", EscErr::ManyChars);
            check(r"\x0ff", EscErr::ManyChars);
            check(r#"\"a"#, EscErr::ManyChars);
            check(r"\na", EscErr::ManyChars);
            check(r"\ra", EscErr::ManyChars);
            check(r"\ta", EscErr::ManyChars);
            check(r"\\a", EscErr::ManyChars);
            check(r"\'a", EscErr::ManyChars);
            check(r"\0a", EscErr::ManyChars);
            check(r"\u{0}x", EscErr::ManyChars);
            check(r"\u{1F63b}}", EscErr::ManyChars);
            check(r"\v", EscErr::InvalidEsc);
            check(r"\💩", EscErr::InvalidEsc);
            check(r"\●", EscErr::InvalidEsc);
            check("\\\r", EscErr::InvalidEsc);
            check(r"\x", EscErr::ShortHexEsc);
            check(r"\x0", EscErr::ShortHexEsc);
            check(r"\xf", EscErr::ShortHexEsc);
            check(r"\xa", EscErr::ShortHexEsc);
            check(r"\xx", EscErr::InvalidInHexEsc);
            check(r"\xы", EscErr::InvalidInHexEsc);
            check(r"\x🦀", EscErr::InvalidInHexEsc);
            check(r"\xtt", EscErr::InvalidInHexEsc);
            check(r"\xff", EscErr::OutOfRangeHexEsc);
            check(r"\xFF", EscErr::OutOfRangeHexEsc);
            check(r"\x80", EscErr::OutOfRangeHexEsc);
            check(r"\u", EscErr::NoBraceInUniEsc);
            check(r"\u[0123]", EscErr::NoBraceInUniEsc);
            check(r"\u{0x}", EscErr::InvalidInUniEsc);
            check(r"\u{", EscErr::UnclosedUniEsc);
            check(r"\u{0000", EscErr::UnclosedUniEsc);
            check(r"\u{}", EscErr::EmptyUniEsc);
            check(r"\u{_0000}", EscErr::UnderscoreUniEsc);
            check(r"\u{0000000}", EscErr::LongUniEsc);
            check(r"\u{FFFFFF}", EscErr::OutOfRangeUniEsc);
            check(r"\u{ffffff}", EscErr::OutOfRangeUniEsc);
            check(r"\u{ffffff}", EscErr::OutOfRangeUniEsc);
            check(r"\u{DC00}", EscErr::LoneSurrogateUniEsc);
            check(r"\u{DDDD}", EscErr::LoneSurrogateUniEsc);
            check(r"\u{DFFF}", EscErr::LoneSurrogateUniEsc);
            check(r"\u{D800}", EscErr::LoneSurrogateUniEsc);
            check(r"\u{DAAA}", EscErr::LoneSurrogateUniEsc);
            check(r"\u{DBFF}", EscErr::LoneSurrogateUniEsc);
        }
        #[test]
        fn test_unescape_char_good() {
            fn check(literal_text: &str, expected_char: char) {
                assert_eq!(unesc_char(literal_text), Ok(expected_char));
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
            fn check(literal: &str, expected: &[(Range<usize>, Result<char, EscErr>)]) {
                let mut unescaped = Vec::with_capacity(literal.len());
                unesc_lit(literal, Mode::Str, &mut |range, res| unescaped.push((range, res)));
                assert_eq!(unescaped, expected);
            }
            check("\\\n", &[]);
            check("\\\n ", &[]);
            check(
                "\\\n \u{a0} x",
                &[
                    (0..5, Err(EscErr::UnskippedWhitespace)),
                    (3..5, Ok('\u{a0}')),
                    (5..6, Ok(' ')),
                    (6..7, Ok('x')),
                ],
            );
            check("\\\n  \n  x", &[(0..7, Err(EscErr::ManySkippedLines)), (7..8, Ok('x'))]);
        }
        #[test]
        fn test_unescape_str_good() {
            fn check(literal_text: &str, expected: &str) {
                let mut buf = Ok(String::with_capacity(literal_text.len()));
                unesc_lit(literal_text, Mode::Str, &mut |range, c| {
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
            fn check(literal_text: &str, expected_error: EscErr) {
                assert_eq!(unesc_byte(literal_text), Err(expected_error));
            }
            check("", EscErr::ZeroChars);
            check(r"\", EscErr::OneSlash);
            check("\n", EscErr::EscOnlyChar);
            check("\t", EscErr::EscOnlyChar);
            check("'", EscErr::EscOnlyChar);
            check("\r", EscErr::BareCarriageReturn);
            check("spam", EscErr::ManyChars);
            check(r"\x0ff", EscErr::ManyChars);
            check(r#"\"a"#, EscErr::ManyChars);
            check(r"\na", EscErr::ManyChars);
            check(r"\ra", EscErr::ManyChars);
            check(r"\ta", EscErr::ManyChars);
            check(r"\\a", EscErr::ManyChars);
            check(r"\'a", EscErr::ManyChars);
            check(r"\0a", EscErr::ManyChars);
            check(r"\v", EscErr::InvalidEsc);
            check(r"\💩", EscErr::InvalidEsc);
            check(r"\●", EscErr::InvalidEsc);
            check(r"\x", EscErr::ShortHexEsc);
            check(r"\x0", EscErr::ShortHexEsc);
            check(r"\xa", EscErr::ShortHexEsc);
            check(r"\xf", EscErr::ShortHexEsc);
            check(r"\xx", EscErr::InvalidInHexEsc);
            check(r"\xы", EscErr::InvalidInHexEsc);
            check(r"\x🦀", EscErr::InvalidInHexEsc);
            check(r"\xtt", EscErr::InvalidInHexEsc);
            check(r"\u", EscErr::NoBraceInUniEsc);
            check(r"\u[0123]", EscErr::NoBraceInUniEsc);
            check(r"\u{0x}", EscErr::InvalidInUniEsc);
            check(r"\u{", EscErr::UnclosedUniEsc);
            check(r"\u{0000", EscErr::UnclosedUniEsc);
            check(r"\u{}", EscErr::EmptyUniEsc);
            check(r"\u{_0000}", EscErr::UnderscoreUniEsc);
            check(r"\u{0000000}", EscErr::LongUniEsc);
            check("ы", EscErr::NonAsciiInByte);
            check("🦀", EscErr::NonAsciiInByte);
            check(r"\u{0}", EscErr::UniEscInByte);
            check(r"\u{000000}", EscErr::UniEscInByte);
            check(r"\u{41}", EscErr::UniEscInByte);
            check(r"\u{0041}", EscErr::UniEscInByte);
            check(r"\u{00_41}", EscErr::UniEscInByte);
            check(r"\u{4__1__}", EscErr::UniEscInByte);
            check(r"\u{1F63b}", EscErr::UniEscInByte);
            check(r"\u{0}x", EscErr::UniEscInByte);
            check(r"\u{1F63b}}", EscErr::UniEscInByte);
            check(r"\u{FFFFFF}", EscErr::UniEscInByte);
            check(r"\u{ffffff}", EscErr::UniEscInByte);
            check(r"\u{ffffff}", EscErr::UniEscInByte);
            check(r"\u{DC00}", EscErr::UniEscInByte);
            check(r"\u{DDDD}", EscErr::UniEscInByte);
            check(r"\u{DFFF}", EscErr::UniEscInByte);
            check(r"\u{D800}", EscErr::UniEscInByte);
            check(r"\u{DAAA}", EscErr::UniEscInByte);
            check(r"\u{DBFF}", EscErr::UniEscInByte);
        }
        #[test]
        fn test_unescape_byte_good() {
            fn check(literal_text: &str, expected_byte: u8) {
                assert_eq!(unesc_byte(literal_text), Ok(expected_byte));
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
                unesc_lit(literal_text, Mode::ByteStr, &mut |range, c| {
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
            fn check(literal: &str, expected: &[(Range<usize>, Result<char, EscErr>)]) {
                let mut unescaped = Vec::with_capacity(literal.len());
                unesc_lit(literal, Mode::RawStr, &mut |range, res| unescaped.push((range, res)));
                assert_eq!(unescaped, expected);
            }
            check("\r", &[(0..1, Err(EscErr::CarriageReturnInRaw))]);
            check("\rx", &[(0..1, Err(EscErr::CarriageReturnInRaw)), (1..2, Ok('x'))]);
        }
        #[test]
        fn test_unescape_raw_byte_str() {
            fn check(literal: &str, expected: &[(Range<usize>, Result<char, EscErr>)]) {
                let mut unescaped = Vec::with_capacity(literal.len());
                unesc_lit(literal, Mode::RawByteStr, &mut |range, res| {
                    unescaped.push((range, res))
                });
                assert_eq!(unescaped, expected);
            }
            check("\r", &[(0..1, Err(EscErr::CarriageReturnInRaw))]);
            check("🦀", &[(0..4, Err(EscErr::NonAsciiInByte))]);
            check("🦀a", &[(0..4, Err(EscErr::NonAsciiInByte)), (4..5, Ok('a'))]);
        }
    }
}
