pub mod case {
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum Case {
        Sensitive,
        Insensitive,
    }
}
pub mod classify {
    use crate::ast::*;
    pub fn expr_requires_semi_to_be_stmt(e: &Expr) -> bool {
        !matches!(
            e.kind,
            ExprKind::If(..)
                | ExprKind::Match(..)
                | ExprKind::Block(..)
                | ExprKind::While(..)
                | ExprKind::Loop(..)
                | ExprKind::ForLoop(..)
                | ExprKind::TryBlock(..)
                | ExprKind::ConstBlock(..)
        )
    }
    pub fn expr_trailing_brace(mut expr: &Expr) -> Option<&Expr> {
        use ExprKind::*;
        loop {
            match &expr.kind {
                AddrOf(_, _, e)
                | Assign(_, e, _)
                | AssignOp(_, _, e)
                | Binary(_, _, e)
                | Break(_, Some(e))
                | Let(_, e, _)
                | Range(_, Some(e), _)
                | Ret(Some(e))
                | Unary(_, e)
                | Yield(Some(e)) => {
                    expr = e;
                },
                Closure(closure) => {
                    expr = &closure.body;
                },
                Async(..) | Block(..) | ForLoop(..) | If(..) | Loop(..) | Match(..) | Struct(..) | TryBlock(..)
                | While(..) => break Some(expr),
                _ => break None,
            }
        }
    }
}
pub mod comments {
    use crate::{ast::token::CommentKind, lexer};
    use rustc_span::{source_map::SourceMap, BytePos, CharPos, FileName, Pos, Symbol};

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum CommentStyle {
        Isolated,
        Trailing,
        Mixed,
        BlankLine,
    }

    #[derive(Clone)]
    pub struct Comment {
        pub style: CommentStyle,
        pub lines: Vec<String>,
        pub pos: BytePos,
    }

    #[inline]
    pub fn may_have_doc_links(s: &str) -> bool {
        s.contains('[')
    }
    pub fn beautify_doc_string(data: Symbol, kind: CommentKind) -> Symbol {
        fn get_vertical_trim(lines: &[&str]) -> Option<(usize, usize)> {
            let mut i = 0;
            let mut j = lines.len();
            if !lines.is_empty() && lines[0].chars().all(|c| c == '*') {
                i += 1;
            }
            if j > i && !lines[j - 1].is_empty() && lines[j - 1].chars().all(|c| c == '*') {
                j -= 1;
            }
            if i != 0 || j != lines.len() {
                Some((i, j))
            } else {
                None
            }
        }
        fn get_horizontal_trim(lines: &[&str], kind: CommentKind) -> Option<String> {
            let mut i = usize::MAX;
            let mut first = true;
            let lines = match kind {
                CommentKind::Block => {
                    let mut i = lines
                        .first()
                        .map(|l| if l.trim_start().starts_with('*') { 0 } else { 1 })
                        .unwrap_or(0);
                    let mut j = lines.len();
                    while i < j && lines[i].trim().is_empty() {
                        i += 1;
                    }
                    while j > i && lines[j - 1].trim().is_empty() {
                        j -= 1;
                    }
                    &lines[i..j]
                },
                CommentKind::Line => lines,
            };
            for line in lines {
                for (j, c) in line.chars().enumerate() {
                    if j > i || !"* \t".contains(c) {
                        return None;
                    }
                    if c == '*' {
                        if first {
                            i = j;
                            first = false;
                        } else if i != j {
                            return None;
                        }
                        break;
                    }
                }
                if i >= line.len() {
                    return None;
                }
            }
            if lines.is_empty() {
                None
            } else {
                Some(lines[0][..i].into())
            }
        }
        let data_s = data.as_str();
        if data_s.contains('\n') {
            let mut lines = data_s.lines().collect::<Vec<&str>>();
            let mut changes = false;
            let lines = if let Some((i, j)) = get_vertical_trim(&lines) {
                changes = true;
                &mut lines[i..j]
            } else {
                &mut lines
            };
            if let Some(horizontal) = get_horizontal_trim(lines, kind) {
                changes = true;
                for line in lines.iter_mut() {
                    if let Some(tmp) = line.strip_prefix(&horizontal) {
                        *line = tmp;
                        if kind == CommentKind::Block
                            && (*line == "*" || line.starts_with("* ") || line.starts_with("**"))
                        {
                            *line = &line[1..];
                        }
                    }
                }
            }
            if changes {
                return Symbol::intern(&lines.join("\n"));
            }
        }
        data
    }
    fn all_whitespace(s: &str, col: CharPos) -> Option<usize> {
        let mut idx = 0;
        for (i, ch) in s.char_indices().take(col.to_usize()) {
            if !ch.is_whitespace() {
                return None;
            }
            idx = i + ch.len_utf8();
        }
        Some(idx)
    }
    fn trim_whitespace_prefix(s: &str, col: CharPos) -> &str {
        let len = s.len();
        match all_whitespace(s, col) {
            Some(col) => {
                if col < len {
                    &s[col..]
                } else {
                    ""
                }
            },
            None => s,
        }
    }
    fn split_block_comment_into_lines(text: &str, col: CharPos) -> Vec<String> {
        let mut res: Vec<String> = vec![];
        let mut lines = text.lines();
        res.extend(lines.next().map(|it| it.to_string()));
        for line in lines {
            res.push(trim_whitespace_prefix(line, col).to_string())
        }
        res
    }
    pub fn gather_comments(sm: &SourceMap, path: FileName, src: String) -> Vec<Comment> {
        let sm = SourceMap::new(sm.path_mapping().clone());
        let source_file = sm.new_source_file(path, src);
        let text = (*source_file.src.as_ref().unwrap()).clone();
        let text: &str = text.as_str();
        let start_bpos = source_file.start_pos;
        let mut pos = 0;
        let mut comments: Vec<Comment> = Vec::new();
        let mut code_to_the_left = false;
        if let Some(shebang_len) = lexer::strip_shebang(text) {
            comments.push(Comment {
                style: CommentStyle::Isolated,
                lines: vec![text[..shebang_len].to_string()],
                pos: start_bpos,
            });
            pos += shebang_len;
        }
        for token in lexer::tokenize(&text[pos..]) {
            let token_text = &text[pos..pos + token.len as usize];
            use lexer::TokKind::*;
            match token.kind {
                Whitespace => {
                    if let Some(mut idx) = token_text.find('\n') {
                        code_to_the_left = false;
                        while let Some(next_newline) = &token_text[idx + 1..].find('\n') {
                            idx += 1 + next_newline;
                            comments.push(Comment {
                                style: CommentStyle::BlankLine,
                                lines: vec![],
                                pos: start_bpos + BytePos((pos + idx) as u32),
                            });
                        }
                    }
                },
                BlockComment { doc_style, .. } => {
                    if doc_style.is_none() {
                        let code_to_the_right =
                            !matches!(text[pos + token.len as usize..].chars().next(), Some('\r' | '\n'));
                        let style = match (code_to_the_left, code_to_the_right) {
                            (_, true) => CommentStyle::Mixed,
                            (false, false) => CommentStyle::Isolated,
                            (true, false) => CommentStyle::Trailing,
                        };
                        let pos_in_file = start_bpos + BytePos(pos as u32);
                        let line_begin_in_file = source_file.line_begin_pos(pos_in_file);
                        let line_begin_pos = (line_begin_in_file - start_bpos).to_usize();
                        let col = CharPos(text[line_begin_pos..pos].chars().count());
                        let lines = split_block_comment_into_lines(token_text, col);
                        comments.push(Comment {
                            style,
                            lines,
                            pos: pos_in_file,
                        })
                    }
                },
                LineComment { style } => {
                    if style.is_none() {
                        comments.push(Comment {
                            style: if code_to_the_left {
                                CommentStyle::Trailing
                            } else {
                                CommentStyle::Isolated
                            },
                            lines: vec![token_text.to_string()],
                            pos: start_bpos + BytePos(pos as u32),
                        })
                    }
                },
                _ => {
                    code_to_the_left = true;
                },
            }
            pos += token.len as usize;
        }
        comments
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use rustc_span::create_default_session_globals_then;

        #[test]
        fn test_block_doc_comment_1() {
            create_default_session_globals_then(|| {
                let comment = "\n * Test \n **  Test\n *   Test\n";
                let stripped = beautify_doc_string(Symbol::intern(comment), CommentKind::Block);
                assert_eq!(stripped.as_str(), " Test \n*  Test\n   Test");
            })
        }

        #[test]
        fn test_block_doc_comment_2() {
            create_default_session_globals_then(|| {
                let comment = "\n * Test\n *  Test\n";
                let stripped = beautify_doc_string(Symbol::intern(comment), CommentKind::Block);
                assert_eq!(stripped.as_str(), " Test\n  Test");
            })
        }

        #[test]
        fn test_block_doc_comment_3() {
            create_default_session_globals_then(|| {
                let comment = "\n let a: *i32;\n *a = 5;\n";
                let stripped = beautify_doc_string(Symbol::intern(comment), CommentKind::Block);
                assert_eq!(stripped.as_str(), "let a: *i32;\n*a = 5;");
            })
        }

        #[test]
        fn test_line_doc_comment() {
            create_default_session_globals_then(|| {
                let stripped = beautify_doc_string(Symbol::intern(" test"), CommentKind::Line);
                assert_eq!(stripped.as_str(), " test");
                let stripped = beautify_doc_string(Symbol::intern("! test"), CommentKind::Line);
                assert_eq!(stripped.as_str(), "! test");
                let stripped = beautify_doc_string(Symbol::intern("test"), CommentKind::Line);
                assert_eq!(stripped.as_str(), "test");
                let stripped = beautify_doc_string(Symbol::intern("!test"), CommentKind::Line);
                assert_eq!(stripped.as_str(), "!test");
            })
        }

        #[test]
        fn test_doc_blocks() {
            create_default_session_globals_then(|| {
                let stripped = beautify_doc_string(Symbol::intern(" # Returns\n     *\n     "), CommentKind::Block);
                assert_eq!(stripped.as_str(), " # Returns\n\n");

                let stripped =
                    beautify_doc_string(Symbol::intern("\n     * # Returns\n     *\n     "), CommentKind::Block);
                assert_eq!(stripped.as_str(), " # Returns\n\n");

                let stripped = beautify_doc_string(Symbol::intern("\n *     a\n "), CommentKind::Block);
                assert_eq!(stripped.as_str(), "     a\n");
            })
        }
    }
}
pub mod literal {
    use crate::{
        ast::{
            token::{self, Token},
            *,
        },
        lexer::unescape::{byte_from_char, unesc_byte, unesc_c_str, unesc_char, unesc_lit, CStrUnit, Mode},
    };
    use rustc_span::{
        symbol::{kw, sym, Symbol},
        Span,
    };
    use std::{ascii, fmt, ops::Range, str};

    pub fn escape_string_symbol(symbol: Symbol) -> Symbol {
        let s = symbol.as_str();
        let escaped = s.escape_default().to_string();
        if s == escaped {
            symbol
        } else {
            Symbol::intern(&escaped)
        }
    }
    pub fn escape_char_symbol(ch: char) -> Symbol {
        let s: String = ch.escape_default().map(Into::<char>::into).collect();
        Symbol::intern(&s)
    }
    pub fn escape_byte_str_symbol(bytes: &[u8]) -> Symbol {
        let s = bytes.escape_ascii().to_string();
        Symbol::intern(&s)
    }

    #[derive(Debug)]
    pub enum LitError {
        LexerError,
        InvalidSuffix,
        InvalidIntSuffix,
        InvalidFloatSuffix,
        NonDecimalFloat(u32),
        IntTooLarge(u32),
        NulInCStr(Range<usize>),
    }
    impl LitKind {
        pub fn from_token_lit(lit: token::Lit) -> Result<LitKind, LitError> {
            let token::Lit { kind, symbol, suffix } = lit;
            if suffix.is_some() && !kind.may_have_suffix() {
                return Err(LitError::InvalidSuffix);
            }
            Ok(match kind {
                token::Bool => {
                    assert!(symbol.is_bool_lit());
                    LitKind::Bool(symbol == kw::True)
                },
                token::Byte => {
                    return unesc_byte(symbol.as_str())
                        .map(LitKind::Byte)
                        .map_err(|_| LitError::LexerError);
                },
                token::Char => {
                    return unesc_char(symbol.as_str())
                        .map(LitKind::Char)
                        .map_err(|_| LitError::LexerError);
                },
                token::Integer => return integer_lit(symbol, suffix),
                token::Float => return float_lit(symbol, suffix),
                token::Str => {
                    let s = symbol.as_str();
                    let symbol = if s.contains(['\\', '\r']) {
                        let mut buf = String::with_capacity(s.len());
                        let mut error = Ok(());
                        unesc_lit(
                            s,
                            Mode::Str,
                            &mut #[inline(always)]
                            |_, unescaped_char| match unescaped_char {
                                Ok(c) => buf.push(c),
                                Err(err) => {
                                    if err.is_fatal() {
                                        error = Err(LitError::LexerError);
                                    }
                                },
                            },
                        );
                        error?;
                        Symbol::intern(&buf)
                    } else {
                        symbol
                    };
                    LitKind::Str(symbol, StrStyle::Cooked)
                },
                token::StrRaw(n) => {
                    let s = symbol.as_str();
                    let symbol = if s.contains('\r') {
                        let mut buf = String::with_capacity(s.len());
                        let mut error = Ok(());
                        unesc_lit(s, Mode::RawStr, &mut |_, unescaped_char| match unescaped_char {
                            Ok(c) => buf.push(c),
                            Err(err) => {
                                if err.is_fatal() {
                                    error = Err(LitError::LexerError);
                                }
                            },
                        });
                        error?;
                        Symbol::intern(&buf)
                    } else {
                        symbol
                    };
                    LitKind::Str(symbol, StrStyle::Raw(n))
                },
                token::ByteStr => {
                    let s = symbol.as_str();
                    let mut buf = Vec::with_capacity(s.len());
                    let mut error = Ok(());
                    unesc_lit(s, Mode::ByteStr, &mut |_, c| match c {
                        Ok(c) => buf.push(byte_from_char(c)),
                        Err(err) => {
                            if err.is_fatal() {
                                error = Err(LitError::LexerError);
                            }
                        },
                    });
                    error?;
                    LitKind::ByteStr(buf.into(), StrStyle::Cooked)
                },
                token::ByteStrRaw(n) => {
                    let s = symbol.as_str();
                    let bytes = if s.contains('\r') {
                        let mut buf = Vec::with_capacity(s.len());
                        let mut error = Ok(());
                        unesc_lit(s, Mode::RawByteStr, &mut |_, c| match c {
                            Ok(c) => buf.push(byte_from_char(c)),
                            Err(err) => {
                                if err.is_fatal() {
                                    error = Err(LitError::LexerError);
                                }
                            },
                        });
                        error?;
                        buf
                    } else {
                        symbol.to_string().into_bytes()
                    };
                    LitKind::ByteStr(bytes.into(), StrStyle::Raw(n))
                },
                token::CStr => {
                    let s = symbol.as_str();
                    let mut buf = Vec::with_capacity(s.len());
                    let mut error = Ok(());
                    unesc_c_str(s, Mode::CStr, &mut |span, c| match c {
                        Ok(CStrUnit::Byte(0) | CStrUnit::Char('\0')) => {
                            error = Err(LitError::NulInCStr(span));
                        },
                        Ok(CStrUnit::Byte(b)) => buf.push(b),
                        Ok(CStrUnit::Char(c)) if c.len_utf8() == 1 => buf.push(c as u8),
                        Ok(CStrUnit::Char(c)) => buf.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
                        Err(err) => {
                            if err.is_fatal() {
                                error = Err(LitError::LexerError);
                            }
                        },
                    });
                    error?;
                    buf.push(0);
                    LitKind::CStr(buf.into(), StrStyle::Cooked)
                },
                token::CStrRaw(n) => {
                    let s = symbol.as_str();
                    let mut buf = Vec::with_capacity(s.len());
                    let mut error = Ok(());
                    unesc_c_str(s, Mode::RawCStr, &mut |span, c| match c {
                        Ok(CStrUnit::Byte(0) | CStrUnit::Char('\0')) => {
                            error = Err(LitError::NulInCStr(span));
                        },
                        Ok(CStrUnit::Byte(b)) => buf.push(b),
                        Ok(CStrUnit::Char(c)) if c.len_utf8() == 1 => buf.push(c as u8),
                        Ok(CStrUnit::Char(c)) => buf.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
                        Err(err) => {
                            if err.is_fatal() {
                                error = Err(LitError::LexerError);
                            }
                        },
                    });
                    error?;
                    buf.push(0);
                    LitKind::CStr(buf.into(), StrStyle::Raw(n))
                },
                token::Err => LitKind::Err,
            })
        }
    }
    impl fmt::Display for LitKind {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match *self {
                LitKind::Byte(b) => {
                    let b: String = ascii::escape_default(b).map(Into::<char>::into).collect();
                    write!(f, "b'{b}'")?;
                },
                LitKind::Char(ch) => write!(f, "'{}'", escape_char_symbol(ch))?,
                LitKind::Str(sym, StrStyle::Cooked) => write!(f, "\"{}\"", escape_string_symbol(sym))?,
                LitKind::Str(sym, StrStyle::Raw(n)) => write!(
                    f,
                    "r{delim}\"{string}\"{delim}",
                    delim = "#".repeat(n as usize),
                    string = sym
                )?,
                LitKind::ByteStr(ref bytes, StrStyle::Cooked) => write!(f, "b\"{}\"", escape_byte_str_symbol(bytes))?,
                LitKind::ByteStr(ref bytes, StrStyle::Raw(n)) => {
                    let symbol = str::from_utf8(bytes).unwrap();
                    write!(
                        f,
                        "br{delim}\"{string}\"{delim}",
                        delim = "#".repeat(n as usize),
                        string = symbol
                    )?;
                },
                LitKind::CStr(ref bytes, StrStyle::Cooked) => write!(f, "c\"{}\"", escape_byte_str_symbol(bytes))?,
                LitKind::CStr(ref bytes, StrStyle::Raw(n)) => {
                    let symbol = str::from_utf8(bytes).unwrap();
                    write!(f, "cr{delim}\"{symbol}\"{delim}", delim = "#".repeat(n as usize),)?;
                },
                LitKind::Int(n, ty) => {
                    write!(f, "{n}")?;
                    match ty {
                        LitIntType::Unsigned(ty) => write!(f, "{}", ty.name())?,
                        LitIntType::Signed(ty) => write!(f, "{}", ty.name())?,
                        LitIntType::Unsuffixed => {},
                    }
                },
                LitKind::Float(symbol, ty) => {
                    write!(f, "{symbol}")?;
                    match ty {
                        LitFloatType::Suffixed(ty) => write!(f, "{}", ty.name())?,
                        LitFloatType::Unsuffixed => {},
                    }
                },
                LitKind::Bool(b) => write!(f, "{}", if b { "true" } else { "false" })?,
                LitKind::Err => {
                    write!(f, "<bad-literal>")?;
                },
            }
            Ok(())
        }
    }

    impl MetaItemLit {
        pub fn from_token_lit(token_lit: token::Lit, span: Span) -> Result<MetaItemLit, LitError> {
            Ok(MetaItemLit {
                symbol: token_lit.symbol,
                suffix: token_lit.suffix,
                kind: LitKind::from_token_lit(token_lit)?,
                span,
            })
        }
        pub fn as_token_lit(&self) -> token::Lit {
            let kind = match self.kind {
                LitKind::Bool(_) => token::Bool,
                LitKind::Str(_, StrStyle::Cooked) => token::Str,
                LitKind::Str(_, StrStyle::Raw(n)) => token::StrRaw(n),
                LitKind::ByteStr(_, StrStyle::Cooked) => token::ByteStr,
                LitKind::ByteStr(_, StrStyle::Raw(n)) => token::ByteStrRaw(n),
                LitKind::CStr(_, StrStyle::Cooked) => token::CStr,
                LitKind::CStr(_, StrStyle::Raw(n)) => token::CStrRaw(n),
                LitKind::Byte(_) => token::Byte,
                LitKind::Char(_) => token::Char,
                LitKind::Int(..) => token::Integer,
                LitKind::Float(..) => token::Float,
                LitKind::Err => token::Err,
            };
            token::Lit::new(kind, self.symbol, self.suffix)
        }
        pub fn from_token(token: &Token) -> Option<MetaItemLit> {
            token::Lit::from_token(token).and_then(|token_lit| MetaItemLit::from_token_lit(token_lit, token.span).ok())
        }
    }
    fn strip_underscores(symbol: Symbol) -> Symbol {
        let s = symbol.as_str();
        if s.contains('_') {
            let mut s = s.to_string();
            s.retain(|c| c != '_');
            return Symbol::intern(&s);
        }
        symbol
    }
    fn filtered_float_lit(symbol: Symbol, suffix: Option<Symbol>, base: u32) -> Result<LitKind, LitError> {
        debug!("filtered_float_lit: {:?}, {:?}, {:?}", symbol, suffix, base);
        if base != 10 {
            return Err(LitError::NonDecimalFloat(base));
        }
        Ok(match suffix {
            Some(suf) => LitKind::Float(
                symbol,
                LitFloatType::Suffixed(match suf {
                    sym::f32 => FloatTy::F32,
                    sym::f64 => FloatTy::F64,
                    _ => return Err(LitError::InvalidFloatSuffix),
                }),
            ),
            None => LitKind::Float(symbol, LitFloatType::Unsuffixed),
        })
    }
    fn float_lit(symbol: Symbol, suffix: Option<Symbol>) -> Result<LitKind, LitError> {
        debug!("float_lit: {:?}, {:?}", symbol, suffix);
        filtered_float_lit(strip_underscores(symbol), suffix, 10)
    }
    fn integer_lit(symbol: Symbol, suffix: Option<Symbol>) -> Result<LitKind, LitError> {
        debug!("integer_lit: {:?}, {:?}", symbol, suffix);
        let symbol = strip_underscores(symbol);
        let s = symbol.as_str();
        let base = match s.as_bytes() {
            [b'0', b'x', ..] => 16,
            [b'0', b'o', ..] => 8,
            [b'0', b'b', ..] => 2,
            _ => 10,
        };
        let ty = match suffix {
            Some(suf) => match suf {
                sym::isize => LitIntType::Signed(IntTy::Isize),
                sym::i8 => LitIntType::Signed(IntTy::I8),
                sym::i16 => LitIntType::Signed(IntTy::I16),
                sym::i32 => LitIntType::Signed(IntTy::I32),
                sym::i64 => LitIntType::Signed(IntTy::I64),
                sym::i128 => LitIntType::Signed(IntTy::I128),
                sym::usize => LitIntType::Unsigned(UintTy::Usize),
                sym::u8 => LitIntType::Unsigned(UintTy::U8),
                sym::u16 => LitIntType::Unsigned(UintTy::U16),
                sym::u32 => LitIntType::Unsigned(UintTy::U32),
                sym::u64 => LitIntType::Unsigned(UintTy::U64),
                sym::u128 => LitIntType::Unsigned(UintTy::U128),
                _ if suf.as_str().starts_with('f') => return filtered_float_lit(symbol, suffix, base),
                _ => return Err(LitError::InvalidIntSuffix),
            },
            _ => LitIntType::Unsuffixed,
        };
        let s = &s[if base != 10 { 2 } else { 0 }..];
        u128::from_str_radix(s, base).map(|i| LitKind::Int(i, ty)).map_err(|_| {
            let from_lexer = base < 10 && s.chars().any(|c| c.to_digit(10).is_some_and(|d| d >= base));
            if from_lexer {
                LitError::LexerError
            } else {
                LitError::IntTooLarge(base)
            }
        })
    }
}
pub mod parser {
    use crate::ast::{
        token::{self, BinOpToken, Token},
        *,
    };
    use rustc_span::symbol::kw;

    #[derive(Copy, Clone, PartialEq, Debug)]
    pub enum AssocOp {
        Add,
        Subtract,
        Multiply,
        Divide,
        Modulus,
        LAnd,
        LOr,
        BitXor,
        BitAnd,
        BitOr,
        ShiftLeft,
        ShiftRight,
        Equal,
        Less,
        LessEqual,
        NotEqual,
        Greater,
        GreaterEqual,
        Assign,
        AssignOp(BinOpToken),
        As,
        DotDot,
        DotDotEq,
    }
    impl AssocOp {
        pub fn from_token(t: &Token) -> Option<AssocOp> {
            use AssocOp::*;
            match t.kind {
                token::BinOpEq(k) => Some(AssignOp(k)),
                token::Eq => Some(Assign),
                token::BinOp(BinOpToken::Star) => Some(Multiply),
                token::BinOp(BinOpToken::Slash) => Some(Divide),
                token::BinOp(BinOpToken::Percent) => Some(Modulus),
                token::BinOp(BinOpToken::Plus) => Some(Add),
                token::BinOp(BinOpToken::Minus) => Some(Subtract),
                token::BinOp(BinOpToken::Shl) => Some(ShiftLeft),
                token::BinOp(BinOpToken::Shr) => Some(ShiftRight),
                token::BinOp(BinOpToken::And) => Some(BitAnd),
                token::BinOp(BinOpToken::Caret) => Some(BitXor),
                token::BinOp(BinOpToken::Or) => Some(BitOr),
                token::Lt => Some(Less),
                token::Le => Some(LessEqual),
                token::Ge => Some(GreaterEqual),
                token::Gt => Some(Greater),
                token::EqEq => Some(Equal),
                token::Ne => Some(NotEqual),
                token::AndAnd => Some(LAnd),
                token::OrOr => Some(LOr),
                token::DotDot => Some(DotDot),
                token::DotDotEq => Some(DotDotEq),
                token::DotDotDot => Some(DotDotEq),
                token::LArrow => Some(Less),
                _ if t.is_keyword(kw::As) => Some(As),
                _ => None,
            }
        }
        pub fn from_ast_binop(op: BinOpKind) -> Self {
            use AssocOp::*;
            match op {
                BinOpKind::Lt => Less,
                BinOpKind::Gt => Greater,
                BinOpKind::Le => LessEqual,
                BinOpKind::Ge => GreaterEqual,
                BinOpKind::Eq => Equal,
                BinOpKind::Ne => NotEqual,
                BinOpKind::Mul => Multiply,
                BinOpKind::Div => Divide,
                BinOpKind::Rem => Modulus,
                BinOpKind::Add => Add,
                BinOpKind::Sub => Subtract,
                BinOpKind::Shl => ShiftLeft,
                BinOpKind::Shr => ShiftRight,
                BinOpKind::BitAnd => BitAnd,
                BinOpKind::BitXor => BitXor,
                BinOpKind::BitOr => BitOr,
                BinOpKind::And => LAnd,
                BinOpKind::Or => LOr,
            }
        }
        pub fn precedence(&self) -> usize {
            use AssocOp::*;
            match *self {
                As => 14,
                Multiply | Divide | Modulus => 13,
                Add | Subtract => 12,
                ShiftLeft | ShiftRight => 11,
                BitAnd => 10,
                BitXor => 9,
                BitOr => 8,
                Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => 7,
                LAnd => 6,
                LOr => 5,
                DotDot | DotDotEq => 4,
                Assign | AssignOp(_) => 2,
            }
        }
        pub fn fixity(&self) -> Fixity {
            use AssocOp::*;
            match *self {
                Assign | AssignOp(_) => Fixity::Right,
                As | Multiply | Divide | Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd | BitXor
                | BitOr | Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual | LAnd | LOr => Fixity::Left,
                DotDot | DotDotEq => Fixity::None,
            }
        }
        pub fn is_comparison(&self) -> bool {
            use AssocOp::*;
            match *self {
                Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => true,
                Assign | AssignOp(_) | As | Multiply | Divide | Modulus | Add | Subtract | ShiftLeft | ShiftRight
                | BitAnd | BitXor | BitOr | LAnd | LOr | DotDot | DotDotEq => false,
            }
        }
        pub fn is_assign_like(&self) -> bool {
            use AssocOp::*;
            match *self {
                Assign | AssignOp(_) => true,
                Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual | As | Multiply | Divide | Modulus
                | Add | Subtract | ShiftLeft | ShiftRight | BitAnd | BitXor | BitOr | LAnd | LOr | DotDot
                | DotDotEq => false,
            }
        }
        pub fn to_ast_binop(&self) -> Option<BinOpKind> {
            use AssocOp::*;
            match *self {
                Less => Some(BinOpKind::Lt),
                Greater => Some(BinOpKind::Gt),
                LessEqual => Some(BinOpKind::Le),
                GreaterEqual => Some(BinOpKind::Ge),
                Equal => Some(BinOpKind::Eq),
                NotEqual => Some(BinOpKind::Ne),
                Multiply => Some(BinOpKind::Mul),
                Divide => Some(BinOpKind::Div),
                Modulus => Some(BinOpKind::Rem),
                Add => Some(BinOpKind::Add),
                Subtract => Some(BinOpKind::Sub),
                ShiftLeft => Some(BinOpKind::Shl),
                ShiftRight => Some(BinOpKind::Shr),
                BitAnd => Some(BinOpKind::BitAnd),
                BitXor => Some(BinOpKind::BitXor),
                BitOr => Some(BinOpKind::BitOr),
                LAnd => Some(BinOpKind::And),
                LOr => Some(BinOpKind::Or),
                Assign | AssignOp(_) | As | DotDot | DotDotEq => None,
            }
        }
        pub fn can_continue_expr_unambiguously(&self) -> bool {
            use AssocOp::*;
            matches!(
                self,
                BitXor | // `{ 42 } ^ 3`
            Assign | // `{ 42 } = { 42 }`
            Divide | // `{ 42 } / 42`
            Modulus | // `{ 42 } % 2`
            ShiftRight | // `{ 42 } >> 2`
            LessEqual | // `{ 42 } <= 3`
            Greater | // `{ 42 } > 3`
            GreaterEqual | // `{ 42 } >= 3`
            AssignOp(_) | // `{ 42 } +=`
            As // `{ 42 } as usize`
            )
        }
    }

    #[derive(PartialEq, Debug)]
    pub enum Fixity {
        Left,
        Right,
        None,
    }

    pub const PREC_CLOSURE: i8 = -40;
    pub const PREC_JUMP: i8 = -30;
    pub const PREC_RANGE: i8 = -10;
    pub const PREC_PREFIX: i8 = 50;
    pub const PREC_POSTFIX: i8 = 60;
    pub const PREC_PAREN: i8 = 99;
    pub const PREC_FORCE_PAREN: i8 = 100;

    #[derive(Debug, Clone, Copy)]
    pub enum ExprPrecedence {
        Closure,
        Break,
        Continue,
        Ret,
        Yield,
        Yeet,
        Become,
        Range,
        Binary(BinOpKind),
        Cast,
        Assign,
        AssignOp,
        AddrOf,
        Let,
        Unary,
        Call,
        MethodCall,
        Field,
        Index,
        Try,
        InlineAsm,
        OffsetOf,
        Mac,
        FormatArgs,
        Array,
        Repeat,
        Tup,
        Lit,
        Path,
        Paren,
        If,
        While,
        ForLoop,
        Loop,
        Match,
        ConstBlock,
        Block,
        TryBlock,
        Struct,
        Async,
        Await,
        Err,
    }
    impl ExprPrecedence {
        pub fn order(self) -> i8 {
            match self {
                ExprPrecedence::Closure => PREC_CLOSURE,
                ExprPrecedence::Break
                | ExprPrecedence::Continue
                | ExprPrecedence::Ret
                | ExprPrecedence::Yield
                | ExprPrecedence::Yeet
                | ExprPrecedence::Become => PREC_JUMP,
                ExprPrecedence::Range => PREC_RANGE,
                ExprPrecedence::Binary(op) => AssocOp::from_ast_binop(op).precedence() as i8,
                ExprPrecedence::Cast => AssocOp::As.precedence() as i8,
                ExprPrecedence::Assign | ExprPrecedence::AssignOp => AssocOp::Assign.precedence() as i8,
                ExprPrecedence::AddrOf | ExprPrecedence::Let | ExprPrecedence::Unary => PREC_PREFIX,
                ExprPrecedence::Await
                | ExprPrecedence::Call
                | ExprPrecedence::MethodCall
                | ExprPrecedence::Field
                | ExprPrecedence::Index
                | ExprPrecedence::Try
                | ExprPrecedence::InlineAsm
                | ExprPrecedence::Mac
                | ExprPrecedence::FormatArgs
                | ExprPrecedence::OffsetOf => PREC_POSTFIX,
                ExprPrecedence::Array
                | ExprPrecedence::Repeat
                | ExprPrecedence::Tup
                | ExprPrecedence::Lit
                | ExprPrecedence::Path
                | ExprPrecedence::Paren
                | ExprPrecedence::If
                | ExprPrecedence::While
                | ExprPrecedence::ForLoop
                | ExprPrecedence::Loop
                | ExprPrecedence::Match
                | ExprPrecedence::ConstBlock
                | ExprPrecedence::Block
                | ExprPrecedence::TryBlock
                | ExprPrecedence::Async
                | ExprPrecedence::Struct
                | ExprPrecedence::Err => PREC_PAREN,
            }
        }
    }

    pub fn prec_let_scrutinee_needs_par() -> usize {
        AssocOp::LAnd.precedence()
    }
    pub fn needs_par_as_let_scrutinee(order: i8) -> bool {
        order <= prec_let_scrutinee_needs_par() as i8
    }
    pub fn contains_exterior_struct_lit(value: &Expr) -> bool {
        match &value.kind {
            ExprKind::Struct(..) => true,
            ExprKind::Assign(lhs, rhs, _) | ExprKind::AssignOp(_, lhs, rhs) | ExprKind::Binary(_, lhs, rhs) => {
                contains_exterior_struct_lit(lhs) || contains_exterior_struct_lit(rhs)
            },
            ExprKind::Await(x, _)
            | ExprKind::Unary(_, x)
            | ExprKind::Cast(x, _)
            | ExprKind::Type(x, _)
            | ExprKind::Field(x, _)
            | ExprKind::Index(x, _) => contains_exterior_struct_lit(x),
            ExprKind::MethodCall(box MethodCall { receiver, .. }) => contains_exterior_struct_lit(receiver),
            _ => false,
        }
    }
}
pub mod unicode {
    pub const TEXT_FLOW_CONTROL_CHARS: &[char] = &[
        '\u{202A}', '\u{202B}', '\u{202D}', '\u{202E}', '\u{2066}', '\u{2067}', '\u{2068}', '\u{202C}', '\u{2069}',
    ];
    #[inline]
    pub fn contains_text_flow_control_chars(s: &str) -> bool {
        let mut bytes = s.as_bytes();
        loop {
            match memchr::memchr(0xE2, bytes) {
                Some(idx) => {
                    let ch = &bytes[idx..idx + 3];
                    match ch {
                        [_, 0x80, 0xAA..=0xAE] | [_, 0x81, 0xA6..=0xA9] => break true,
                        _ => {},
                    }
                    bytes = &bytes[idx + 3..];
                },
                None => {
                    break false;
                },
            }
        }
    }
}
