use crate::{
    lexer,
    SyntaxKind::{self, *},
    T,
};
use std::ops;

struct LexErr {
    msg: String,
    tok: u32,
}

pub struct Lexed<'a> {
    txt: &'a str,
    kinds: Vec<SyntaxKind>,
    start: Vec<u32>,
    errs: Vec<LexErr>,
}
impl<'a> Lexed<'a> {
    pub fn new(x: &'a str) -> Lexed<'a> {
        let mut c = Converter::new(x);
        if let Some(x) = lexer::strip_shebang(x) {
            c.res.push(SHEBANG, c.off);
            c.off = x;
        };
        for tok in lexer::tokenize(&x[c.off..]) {
            let txt = &x[c.off..][..tok.len as usize];
            c.extend_tok(&tok.kind, txt);
        }
        c.finalize_with_eof()
    }
    pub fn single_token(x: &'a str) -> Option<(SyntaxKind, Option<String>)> {
        if x.is_empty() {
            return None;
        }
        let y = lexer::tokenize(x).next()?;
        if y.len as usize != x.len() {
            return None;
        }
        let mut c = Converter::new(x);
        c.extend_tok(&y.kind, x);
        match &*c.res.kinds {
            [x] => Some((*x, c.res.errs.pop().map(|x| x.msg))),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &str {
        self.txt
    }
    pub fn len(&self) -> usize {
        self.kinds.len() - 1
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn kind(&self, i: usize) -> SyntaxKind {
        assert!(i < self.len());
        self.kinds[i]
    }
    pub fn text(&self, i: usize) -> &str {
        self.range_text(i..i + 1)
    }
    pub fn range_text(&self, r: ops::Range<usize>) -> &str {
        assert!(r.start < r.end && r.end <= self.len());
        let lo = self.start[r.start] as usize;
        let hi = self.start[r.end] as usize;
        &self.txt[lo..hi]
    }
    pub fn text_range(&self, i: usize) -> ops::Range<usize> {
        assert!(i < self.len());
        let lo = self.start[i] as usize;
        let hi = self.start[i + 1] as usize;
        lo..hi
    }
    pub fn text_start(&self, i: usize) -> usize {
        assert!(i <= self.len());
        self.start[i] as usize
    }
    pub fn text_len(&self, i: usize) -> usize {
        assert!(i < self.len());
        let r = self.text_range(i);
        r.end - r.start
    }
    pub fn err(&self, i: usize) -> Option<&str> {
        assert!(i < self.len());
        let err = self.errs.binary_search_by_key(&(i as u32), |x| x.tok).ok()?;
        Some(self.errs[err].msg.as_str())
    }
    pub fn errs(&self) -> impl Iterator<Item = (usize, &str)> + '_ {
        self.errs.iter().map(|x| (x.tok as usize, x.msg.as_str()))
    }
    fn push(&mut self, x: SyntaxKind, off: usize) {
        self.kinds.push(x);
        self.start.push(off as u32);
    }
}

struct Converter<'a> {
    res: Lexed<'a>,
    off: usize,
}
impl<'a> Converter<'a> {
    fn new(txt: &'a str) -> Self {
        Self {
            res: Lexed {
                txt,
                kinds: Vec::new(),
                start: Vec::new(),
                errs: Vec::new(),
            },
            off: 0,
        }
    }
    fn finalize_with_eof(mut self) -> Lexed<'a> {
        self.res.push(EOF, self.off);
        self.res
    }
    fn push(&mut self, x: SyntaxKind, len: usize, err: Option<&str>) {
        self.res.push(x, self.off);
        self.off += len;
        if let Some(err) = err {
            let tok = self.res.len() as u32;
            let msg = err.to_string();
            self.res.errs.push(LexErr { msg, tok });
        }
    }
    fn extend_tok(&mut self, x: &lexer::TokKind, txt: &str) {
        let mut err = "";
        let y = {
            use lexer::TokKind::*;
            match x {
                LineComment { style: _ } => COMMENT,
                BlockComment { style: _, terminated } => {
                    if !terminated {
                        err = "Missing trailing `*/` symbols to terminate the block comment";
                    }
                    COMMENT
                },
                Whitespace => WHITESPACE,
                Ident if txt == "_" => UNDERSCORE,
                Ident => SyntaxKind::from_keyword(txt).unwrap_or(IDENT),
                InvalidIdent => {
                    err = "Ident contains invalid characters";
                    IDENT
                },
                RawIdent => IDENT,
                Lit { kind, .. } => {
                    self.extend_lit(txt.len(), kind);
                    return;
                },
                Lifetime {
                    starts_with_num: starts_with_number,
                } => {
                    if *starts_with_number {
                        err = "Lifetime name cannot start with a number";
                    }
                    LIFETIME_IDENT
                },
                Semi => T![;],
                Comma => T![,],
                Dot => T![.],
                OpenParen => T!['('],
                CloseParen => T![')'],
                OpenBrace => T!['{'],
                CloseBrace => T!['}'],
                OpenBracket => T!['['],
                CloseBracket => T![']'],
                At => T![@],
                Pound => T![#],
                Tilde => T![~],
                Question => T![?],
                Colon => T![:],
                Dollar => T![$],
                Eq => T![=],
                Bang => T![!],
                Lt => T![<],
                Gt => T![>],
                Minus => T![-],
                And => T![&],
                Or => T![|],
                Plus => T![+],
                Star => T![*],
                Slash => T![/],
                Caret => T![^],
                Percent => T![%],
                Unknown => ERROR,
                UnknownPrefix => {
                    err = "unknown literal prefix";
                    IDENT
                },
                Eof => EOF,
            }
        };
        let err = if err.is_empty() { None } else { Some(err) };
        self.push(y, txt.len(), err);
    }
    fn extend_lit(&mut self, len: usize, kind: &lexer::LitKind) {
        let mut err = "";
        use lexer::LitKind::*;
        let y = match *kind {
            Int { empty_int, base: _ } => {
                if empty_int {
                    err = "Missing digits after the integer base prefix";
                }
                INT_NUMBER
            },
            Float {
                empty_exp: empty_exponent,
                base: _,
            } => {
                if empty_exponent {
                    err = "Missing digits after the exponent symbol";
                }
                FLOAT_NUMBER
            },
            Char { terminated } => {
                if !terminated {
                    err = "Missing trailing `'` symbol to terminate the character literal";
                }
                CHAR
            },
            Byte { terminated } => {
                if !terminated {
                    err = "Missing trailing `'` symbol to terminate the byte literal";
                }
                BYTE
            },
            Str { terminated } => {
                if !terminated {
                    err = "Missing trailing `\"` symbol to terminate the string literal";
                }
                STRING
            },
            ByteStr { terminated } => {
                if !terminated {
                    err = "Missing trailing `\"` symbol to terminate the byte string literal";
                }
                BYTE_STRING
            },
            CStr { terminated } => {
                if !terminated {
                    err = "Missing trailing `\"` symbol to terminate the string literal";
                }
                C_STRING
            },
            RawStr { n_hashes } => {
                if n_hashes.is_none() {
                    err = "Invalid raw string literal";
                }
                STRING
            },
            RawByteStr { n_hashes } => {
                if n_hashes.is_none() {
                    err = "Invalid raw string literal";
                }
                BYTE_STRING
            },
            RawCStr { n_hashes } => {
                if n_hashes.is_none() {
                    err = "Invalid raw string literal";
                }
                C_STRING
            },
        };
        let err = if err.is_empty() { None } else { Some(err) };
        self.push(y, len, err);
    }
}
