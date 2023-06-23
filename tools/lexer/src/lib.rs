use crate::errors;
use crate::make_unclosed_delims_error;
use ast::{
    self,
    lexer::{
        unescape::{self, EscapeError, Mode},
        Base, Cursor, DocStyle, RawStrError,
    },
    token::{self, stream::TokenStream, CommentKind, Delimiter, Token, TokenKind},
    util::unicode::contains_text_flow_control_chars,
    AttrStyle,
};
use rustc_errors::{error_code, Applicability, Diagnostic, DiagnosticBuilder, StashKey};
use rustc_session::{
    lint::{
        builtin::{RUST_2021_PREFIXES_INCOMPATIBLE_SYNTAX, TEXT_DIRECTION_CODEPOINT_IN_COMMENT},
        BuiltinLintDiagnostics,
    },
    parse::ParseSess,
};
use rustc_span::{
    edition::Edition,
    symbol::{sym, Symbol},
    BytePos, Pos, Span,
};
use std::ops::Range;

mod chars;
use chars::UNICODE_ARRAY;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(super::Token, 12);

#[derive(Clone, Debug)]
pub struct UnmatchedDelim {
    pub expected_delim: Delimiter,
    pub found_delim: Option<Delimiter>,
    pub found_span: Span,
    pub unclosed_span: Option<Span>,
    pub candidate_span: Option<Span>,
}

pub fn parse_token_trees<'a>(
    sess: &'a ParseSess,
    mut src: &'a str,
    mut start_pos: BytePos,
    override_span: Option<Span>,
) -> Result<TokenStream, Vec<Diagnostic>> {
    if let Some(shebang_len) = super::strip_shebang(src) {
        src = &src[shebang_len..];
        start_pos = start_pos + BytePos::from_usize(shebang_len);
    }
    let cursor = Cursor::new(src);
    let string_reader = StringReader {
        sess,
        start_pos,
        pos: start_pos,
        src,
        cursor,
        override_span,
        nbsp_is_whitespace: false,
    };
    let (token_trees, unmatched_delims) = tokentrees::TokenTreesReader::parse_all_token_trees(string_reader);
    match token_trees {
        Ok(stream) if unmatched_delims.is_empty() => Ok(stream),
        _ => {
            let mut buffer = Vec::with_capacity(1);
            for unmatched in unmatched_delims {
                if let Some(err) = make_unclosed_delims_error(unmatched, &sess) {
                    err.buffer(&mut buffer);
                }
            }
            if let Err(err) = token_trees {
                err.buffer(&mut buffer);
            }
            Err(buffer)
        },
    }
}

struct StringReader<'a> {
    sess: &'a ParseSess,
    start_pos: BytePos,
    pos: BytePos,
    src: &'a str,
    cursor: Cursor<'a>,
    override_span: Option<Span>,
    nbsp_is_whitespace: bool,
}
impl<'a> StringReader<'a> {
    fn mk_sp(&self, lo: BytePos, hi: BytePos) -> Span {
        self.override_span.unwrap_or_else(|| Span::with_root_ctxt(lo, hi))
    }
    fn next_token(&mut self) -> (Token, bool) {
        let mut preceded_by_whitespace = false;
        let mut swallow_next_invalid = 0;
        loop {
            let token = self.cursor.advance_token();
            let start = self.pos;
            self.pos = self.pos + BytePos(token.len);
            debug!("next_token: {:?}({:?})", token.kind, self.str_from(start));
            let kind = match token.kind {
                super::TokenKind::LineComment { doc_style } => {
                    let Some(doc_style) = doc_style else {
                        self.lint_unicode_text_flow(start);
                        preceded_by_whitespace = true;
                        continue;
                    };
                    let content_start = start + BytePos(3);
                    let content = self.str_from(content_start);
                    self.cook_doc_comment(content_start, content, CommentKind::Line, doc_style)
                },
                super::TokenKind::BlockComment { doc_style, terminated } => {
                    if !terminated {
                        self.report_unterminated_block_comment(start, doc_style);
                    }
                    let Some(doc_style) = doc_style else {
                        self.lint_unicode_text_flow(start);
                        preceded_by_whitespace = true;
                        continue;
                    };
                    let content_start = start + BytePos(3);
                    let content_end = self.pos - BytePos(if terminated { 2 } else { 0 });
                    let content = self.str_from_to(content_start, content_end);
                    self.cook_doc_comment(content_start, content, CommentKind::Block, doc_style)
                },
                super::TokenKind::Whitespace => {
                    preceded_by_whitespace = true;
                    continue;
                },
                super::TokenKind::Ident => {
                    let sym = nfc_normalize(self.str_from(start));
                    let span = self.mk_sp(start, self.pos);
                    self.sess.symbol_gallery.insert(sym, span);
                    token::Ident(sym, false)
                },
                super::TokenKind::RawIdent => {
                    let sym = nfc_normalize(self.str_from(start + BytePos(2)));
                    let span = self.mk_sp(start, self.pos);
                    self.sess.symbol_gallery.insert(sym, span);
                    if !sym.can_be_raw() {
                        self.sess.emit_err(errors::CannotBeRawIdent { span, ident: sym });
                    }
                    self.sess.raw_identifier_spans.push(span);
                    token::Ident(sym, true)
                },
                super::TokenKind::UnknownPrefix => {
                    self.report_unknown_prefix(start);
                    let sym = nfc_normalize(self.str_from(start));
                    let span = self.mk_sp(start, self.pos);
                    self.sess.symbol_gallery.insert(sym, span);
                    token::Ident(sym, false)
                },
                super::TokenKind::InvalidIdent
                    if !UNICODE_ARRAY.iter().any(|&(c, _, _)| {
                        let sym = self.str_from(start);
                        sym.chars().count() == 1 && c == sym.chars().next().unwrap()
                    }) =>
                {
                    let sym = nfc_normalize(self.str_from(start));
                    let span = self.mk_sp(start, self.pos);
                    self.sess
                        .bad_unicode_identifiers
                        .borrow_mut()
                        .entry(sym)
                        .or_default()
                        .push(span);
                    token::Ident(sym, false)
                },
                super::TokenKind::Literal { kind, suffix_start } => {
                    let suffix_start = start + BytePos(suffix_start);
                    let (kind, symbol) = self.cook_lexer_literal(start, suffix_start, kind);
                    if let token::LitKind::CStr | token::LitKind::CStrRaw(_) = kind {
                        self.sess
                            .gated_spans
                            .gate(sym::c_str_literals, self.mk_sp(start, self.pos));
                    }
                    let suffix = if suffix_start < self.pos {
                        let string = self.str_from(suffix_start);
                        if string == "_" {
                            self.sess.span_diagnostic.emit_err(errors::UnderscoreLiteralSuffix {
                                span: self.mk_sp(suffix_start, self.pos),
                            });
                            None
                        } else {
                            Some(Symbol::intern(string))
                        }
                    } else {
                        None
                    };
                    token::Literal(token::Lit { kind, symbol, suffix })
                },
                super::TokenKind::Lifetime { starts_with_number } => {
                    let lifetime_name = self.str_from(start);
                    if starts_with_number {
                        let span = self.mk_sp(start, self.pos);
                        let mut diag = self.sess.struct_err("lifetimes cannot start with a number");
                        diag.set_span(span);
                        diag.stash(span, StashKey::LifetimeIsChar);
                    }
                    let ident = Symbol::intern(lifetime_name);
                    token::Lifetime(ident)
                },
                super::TokenKind::Semi => token::Semi,
                super::TokenKind::Comma => token::Comma,
                super::TokenKind::Dot => token::Dot,
                super::TokenKind::OpenParen => token::OpenDelim(Delimiter::Parenthesis),
                super::TokenKind::CloseParen => token::CloseDelim(Delimiter::Parenthesis),
                super::TokenKind::OpenBrace => token::OpenDelim(Delimiter::Brace),
                super::TokenKind::CloseBrace => token::CloseDelim(Delimiter::Brace),
                super::TokenKind::OpenBracket => token::OpenDelim(Delimiter::Bracket),
                super::TokenKind::CloseBracket => token::CloseDelim(Delimiter::Bracket),
                super::TokenKind::At => token::At,
                super::TokenKind::Pound => token::Pound,
                super::TokenKind::Tilde => token::Tilde,
                super::TokenKind::Question => token::Question,
                super::TokenKind::Colon => token::Colon,
                super::TokenKind::Dollar => token::Dollar,
                super::TokenKind::Eq => token::Eq,
                super::TokenKind::Bang => token::Not,
                super::TokenKind::Lt => token::Lt,
                super::TokenKind::Gt => token::Gt,
                super::TokenKind::Minus => token::BinOp(token::Minus),
                super::TokenKind::And => token::BinOp(token::And),
                super::TokenKind::Or => token::BinOp(token::Or),
                super::TokenKind::Plus => token::BinOp(token::Plus),
                super::TokenKind::Star => token::BinOp(token::Star),
                super::TokenKind::Slash => token::BinOp(token::Slash),
                super::TokenKind::Caret => token::BinOp(token::Caret),
                super::TokenKind::Percent => token::BinOp(token::Percent),
                super::TokenKind::Unknown | super::TokenKind::InvalidIdent => {
                    if swallow_next_invalid > 0 {
                        swallow_next_invalid -= 1;
                        continue;
                    }
                    let mut it = self.str_from_to_end(start).chars();
                    let c = it.next().unwrap();
                    if c == '\u{00a0}' {
                        if self.nbsp_is_whitespace {
                            preceded_by_whitespace = true;
                            continue;
                        }
                        self.nbsp_is_whitespace = true;
                    }
                    let repeats = it.take_while(|c1| *c1 == c).count();
                    let (token, sugg) = chars::check_for_substitution(self, start, c, repeats + 1);
                    self.sess.emit_err(errors::UnknownTokenStart {
                        span: self.mk_sp(start, self.pos + Pos::from_usize(repeats * c.len_utf8())),
                        escaped: escaped_char(c),
                        sugg,
                        null: if c == '\x00' {
                            Some(errors::UnknownTokenNull)
                        } else {
                            None
                        },
                        repeat: if repeats > 0 {
                            swallow_next_invalid = repeats;
                            Some(errors::UnknownTokenRepeat { repeats })
                        } else {
                            None
                        },
                    });
                    if let Some(token) = token {
                        token
                    } else {
                        preceded_by_whitespace = true;
                        continue;
                    }
                },
                super::TokenKind::Eof => token::Eof,
            };
            let span = self.mk_sp(start, self.pos);
            return (Token::new(kind, span), preceded_by_whitespace);
        }
    }
    fn struct_fatal_span_char(&self, from_pos: BytePos, to_pos: BytePos, m: &str, c: char) -> DiagnosticBuilder<'a, !> {
        self.sess
            .span_diagnostic
            .struct_span_fatal(self.mk_sp(from_pos, to_pos), format!("{}: {}", m, escaped_char(c)))
    }
    fn lint_unicode_text_flow(&self, start: BytePos) {
        let content_start = start + BytePos(2);
        let content = self.str_from(content_start);
        if contains_text_flow_control_chars(content) {
            let span = self.mk_sp(start, self.pos);
            self.sess.buffer_lint_with_diagnostic(
                &TEXT_DIRECTION_CODEPOINT_IN_COMMENT,
                span,
                ast::CRATE_NODE_ID,
                "unicode codepoint changing visible direction of text present in comment",
                BuiltinLintDiagnostics::UnicodeTextFlow(span, content.to_string()),
            );
        }
    }
    fn cook_doc_comment(
        &self,
        content_start: BytePos,
        content: &str,
        comment_kind: CommentKind,
        doc_style: DocStyle,
    ) -> TokenKind {
        if content.contains('\r') {
            for (idx, _) in content.char_indices().filter(|&(_, c)| c == '\r') {
                let span = self.mk_sp(
                    content_start + BytePos(idx as u32),
                    content_start + BytePos(idx as u32 + 1),
                );
                let block = matches!(comment_kind, CommentKind::Block);
                self.sess.emit_err(errors::CrDocComment { span, block });
            }
        }
        let attr_style = match doc_style {
            DocStyle::Outer => AttrStyle::Outer,
            DocStyle::Inner => AttrStyle::Inner,
        };
        token::DocComment(comment_kind, attr_style, Symbol::intern(content))
    }
    fn cook_lexer_literal(&self, start: BytePos, end: BytePos, kind: super::LiteralKind) -> (token::LitKind, Symbol) {
        match kind {
            super::LiteralKind::Char { terminated } => {
                if !terminated {
                    self.sess.span_diagnostic.span_fatal_with_code(
                        self.mk_sp(start, end),
                        "unterminated character literal",
                        error_code!(E0762),
                    )
                }
                self.cook_quoted(token::Char, Mode::Char, start, end, 1, 1) // ' '
            },
            super::LiteralKind::Byte { terminated } => {
                if !terminated {
                    self.sess.span_diagnostic.span_fatal_with_code(
                        self.mk_sp(start + BytePos(1), end),
                        "unterminated byte constant",
                        error_code!(E0763),
                    )
                }
                self.cook_quoted(token::Byte, Mode::Byte, start, end, 2, 1) // b' '
            },
            super::LiteralKind::Str { terminated } => {
                if !terminated {
                    self.sess.span_diagnostic.span_fatal_with_code(
                        self.mk_sp(start, end),
                        "unterminated double quote string",
                        error_code!(E0765),
                    )
                }
                self.cook_quoted(token::Str, Mode::Str, start, end, 1, 1) // " "
            },
            super::LiteralKind::ByteStr { terminated } => {
                if !terminated {
                    self.sess.span_diagnostic.span_fatal_with_code(
                        self.mk_sp(start + BytePos(1), end),
                        "unterminated double quote byte string",
                        error_code!(E0766),
                    )
                }
                self.cook_quoted(token::ByteStr, Mode::ByteStr, start, end, 2, 1)
            },
            super::LiteralKind::CStr { terminated } => {
                if !terminated {
                    self.sess.span_diagnostic.span_fatal_with_code(
                        self.mk_sp(start + BytePos(1), end),
                        "unterminated C string",
                        error_code!(E0767),
                    )
                }
                self.cook_c_string(token::CStr, Mode::CStr, start, end, 2, 1) // c" "
            },
            super::LiteralKind::RawStr { n_hashes } => {
                if let Some(n_hashes) = n_hashes {
                    let n = u32::from(n_hashes);
                    let kind = token::StrRaw(n_hashes);
                    self.cook_quoted(kind, Mode::RawStr, start, end, 2 + n, 1 + n)
                } else {
                    self.report_raw_str_error(start, 1);
                }
            },
            super::LiteralKind::RawByteStr { n_hashes } => {
                if let Some(n_hashes) = n_hashes {
                    let n = u32::from(n_hashes);
                    let kind = token::ByteStrRaw(n_hashes);
                    self.cook_quoted(kind, Mode::RawByteStr, start, end, 3 + n, 1 + n)
                } else {
                    self.report_raw_str_error(start, 2);
                }
            },
            super::LiteralKind::RawCStr { n_hashes } => {
                if let Some(n_hashes) = n_hashes {
                    let n = u32::from(n_hashes);
                    let kind = token::CStrRaw(n_hashes);
                    self.cook_c_string(kind, Mode::RawCStr, start, end, 3 + n, 1 + n)
                } else {
                    self.report_raw_str_error(start, 2);
                }
            },
            super::LiteralKind::Int { base, empty_int } => {
                if empty_int {
                    let span = self.mk_sp(start, end);
                    self.sess.emit_err(errors::NoDigitsLiteral { span });
                    (token::Integer, sym::integer(0))
                } else {
                    if matches!(base, Base::Binary | Base::Octal) {
                        let base = base as u32;
                        let s = self.str_from_to(start + BytePos(2), end);
                        for (idx, c) in s.char_indices() {
                            let span = self.mk_sp(
                                start + BytePos::from_usize(2 + idx),
                                start + BytePos::from_usize(2 + idx + c.len_utf8()),
                            );
                            if c != '_' && c.to_digit(base).is_none() {
                                self.sess.emit_err(errors::InvalidDigitLiteral { span, base });
                            }
                        }
                    }
                    (token::Integer, self.symbol_from_to(start, end))
                }
            },
            super::LiteralKind::Float { base, empty_exponent } => {
                if empty_exponent {
                    let span = self.mk_sp(start, self.pos);
                    self.sess.emit_err(errors::EmptyExponentFloat { span });
                }
                let base = match base {
                    Base::Hexadecimal => Some("hexadecimal"),
                    Base::Octal => Some("octal"),
                    Base::Binary => Some("binary"),
                    _ => None,
                };
                if let Some(base) = base {
                    let span = self.mk_sp(start, end);
                    self.sess.emit_err(errors::FloatLiteralUnsupportedBase { span, base });
                }
                (token::Float, self.symbol_from_to(start, end))
            },
        }
    }
    #[inline]
    fn src_index(&self, pos: BytePos) -> usize {
        (pos - self.start_pos).to_usize()
    }
    fn str_from(&self, start: BytePos) -> &'a str {
        self.str_from_to(start, self.pos)
    }
    fn symbol_from_to(&self, start: BytePos, end: BytePos) -> Symbol {
        debug!("taking an ident from {:?} to {:?}", start, end);
        Symbol::intern(self.str_from_to(start, end))
    }
    fn str_from_to(&self, start: BytePos, end: BytePos) -> &'a str {
        &self.src[self.src_index(start)..self.src_index(end)]
    }
    fn str_from_to_end(&self, start: BytePos) -> &'a str {
        &self.src[self.src_index(start)..]
    }
    fn report_raw_str_error(&self, start: BytePos, prefix_len: u32) -> ! {
        match super::validate_raw_str(self.str_from(start), prefix_len) {
            Err(RawStrError::InvalidStarter { bad_char }) => self.report_non_started_raw_string(start, bad_char),
            Err(RawStrError::NoTerminator {
                expected,
                found,
                possible_terminator_offset,
            }) => self.report_unterminated_raw_string(start, expected, possible_terminator_offset, found),
            Err(RawStrError::TooManyDelimiters { found }) => self.report_too_many_hashes(start, found),
            Ok(()) => panic!("no error found for supposedly invalid raw string literal"),
        }
    }
    fn report_non_started_raw_string(&self, start: BytePos, bad_char: char) -> ! {
        self.struct_fatal_span_char(
            start,
            self.pos,
            "found invalid character; only `#` is allowed in raw string delimitation",
            bad_char,
        )
        .emit()
    }
    fn report_unterminated_raw_string(
        &self,
        start: BytePos,
        n_hashes: u32,
        possible_offset: Option<u32>,
        found_terminators: u32,
    ) -> ! {
        let mut err = self.sess.span_diagnostic.struct_span_fatal_with_code(
            self.mk_sp(start, start),
            "unterminated raw string",
            error_code!(E0748),
        );
        err.span_label(self.mk_sp(start, start), "unterminated raw string");
        if n_hashes > 0 {
            err.note(format!(
                "this raw string should be terminated with `\"{}`",
                "#".repeat(n_hashes as usize)
            ));
        }
        if let Some(possible_offset) = possible_offset {
            let lo = start + BytePos(possible_offset);
            let hi = lo + BytePos(found_terminators);
            let span = self.mk_sp(lo, hi);
            err.span_suggestion(
                span,
                "consider terminating the string here",
                "#".repeat(n_hashes as usize),
                Applicability::MaybeIncorrect,
            );
        }
        err.emit()
    }
    fn report_unterminated_block_comment(&self, start: BytePos, doc_style: Option<DocStyle>) {
        let msg = match doc_style {
            Some(_) => "unterminated block doc-comment",
            None => "unterminated block comment",
        };
        let last_bpos = self.pos;
        let mut err = self.sess.span_diagnostic.struct_span_fatal_with_code(
            self.mk_sp(start, last_bpos),
            msg,
            error_code!(E0758),
        );
        let mut nested_block_comment_open_idxs = vec![];
        let mut last_nested_block_comment_idxs = None;
        let mut content_chars = self.str_from(start).char_indices().peekable();
        while let Some((idx, current_char)) = content_chars.next() {
            match content_chars.peek() {
                Some((_, '*')) if current_char == '/' => {
                    nested_block_comment_open_idxs.push(idx);
                },
                Some((_, '/')) if current_char == '*' => {
                    last_nested_block_comment_idxs =
                        nested_block_comment_open_idxs.pop().map(|open_idx| (open_idx, idx));
                },
                _ => {},
            };
        }
        if let Some((nested_open_idx, nested_close_idx)) = last_nested_block_comment_idxs {
            err.span_label(self.mk_sp(start, start + BytePos(2)), msg)
                .span_label(
                    self.mk_sp(
                        start + BytePos(nested_open_idx as u32),
                        start + BytePos(nested_open_idx as u32 + 2),
                    ),
                    "...as last nested comment starts here, maybe you want to close this instead?",
                )
                .span_label(
                    self.mk_sp(
                        start + BytePos(nested_close_idx as u32),
                        start + BytePos(nested_close_idx as u32 + 2),
                    ),
                    "...and last nested comment terminates here.",
                );
        }
        err.emit();
    }
    fn report_unknown_prefix(&self, start: BytePos) {
        let prefix_span = self.mk_sp(start, self.pos);
        let prefix = self.str_from_to(start, self.pos);
        let expn_data = prefix_span.ctxt().outer_expn_data();
        if expn_data.edition >= Edition::Edition2021 {
            let sugg = if prefix == "rb" {
                Some(errors::UnknownPrefixSugg::UseBr(prefix_span))
            } else if expn_data.is_root() {
                Some(errors::UnknownPrefixSugg::Whitespace(prefix_span.shrink_to_hi()))
            } else {
                None
            };
            self.sess.emit_err(errors::UnknownPrefix {
                span: prefix_span,
                prefix,
                sugg,
            });
        } else {
            self.sess.buffer_lint_with_diagnostic(
                &RUST_2021_PREFIXES_INCOMPATIBLE_SYNTAX,
                prefix_span,
                ast::CRATE_NODE_ID,
                format!("prefix `{prefix}` is unknown"),
                BuiltinLintDiagnostics::ReservedPrefix(prefix_span),
            );
        }
    }
    fn report_too_many_hashes(&self, start: BytePos, num: u32) -> ! {
        self.sess.emit_fatal(errors::TooManyHashes {
            span: self.mk_sp(start, self.pos),
            num,
        });
    }
    fn cook_common(
        &self,
        kind: token::LitKind,
        mode: Mode,
        start: BytePos,
        end: BytePos,
        prefix_len: u32,
        postfix_len: u32,
        unescape: fn(&str, Mode, &mut dyn FnMut(Range<usize>, Result<(), EscapeError>)),
    ) -> (token::LitKind, Symbol) {
        let mut has_fatal_err = false;
        let content_start = start + BytePos(prefix_len);
        let content_end = end - BytePos(postfix_len);
        let lit_content = self.str_from_to(content_start, content_end);
        unescape(lit_content, mode, &mut |range, result| {
            if let Err(err) = result {
                let span_with_quotes = self.mk_sp(start, end);
                let (start, end) = (range.start as u32, range.end as u32);
                let lo = content_start + BytePos(start);
                let hi = lo + BytePos(end - start);
                let span = self.mk_sp(lo, hi);
                if err.is_fatal() {
                    has_fatal_err = true;
                }
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit_content,
                    span_with_quotes,
                    span,
                    mode,
                    range,
                    err,
                );
            }
        });
        if !has_fatal_err {
            (kind, Symbol::intern(lit_content))
        } else {
            (token::Err, self.symbol_from_to(start, end))
        }
    }
    fn cook_quoted(
        &self,
        kind: token::LitKind,
        mode: Mode,
        start: BytePos,
        end: BytePos,
        prefix_len: u32,
        postfix_len: u32,
    ) -> (token::LitKind, Symbol) {
        self.cook_common(
            kind,
            mode,
            start,
            end,
            prefix_len,
            postfix_len,
            |src, mode, callback| {
                unescape::unescape_literal(src, mode, &mut |span, result| callback(span, result.map(drop)))
            },
        )
    }
    fn cook_c_string(
        &self,
        kind: token::LitKind,
        mode: Mode,
        start: BytePos,
        end: BytePos,
        prefix_len: u32,
        postfix_len: u32,
    ) -> (token::LitKind, Symbol) {
        self.cook_common(
            kind,
            mode,
            start,
            end,
            prefix_len,
            postfix_len,
            |src, mode, callback| {
                unescape::unescape_c_string(src, mode, &mut |span, result| callback(span, result.map(drop)))
            },
        )
    }
}

pub fn nfc_normalize(string: &str) -> Symbol {
    use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};
    match is_nfc_quick(string.chars()) {
        IsNormalized::Yes => Symbol::intern(string),
        _ => {
            let normalized_str: String = string.chars().nfc().collect();
            Symbol::intern(&normalized_str)
        },
    }
}

mod tokentrees {
    use super::{
        diagnostics::{report_suspicious_mismatch_block, same_indentation_level, TokenTreeDiagInfo},
        StringReader, UnmatchedDelim,
    };
    use ast::token::{
        self,
        stream::{DelimSpan, Spacing, TokenStream, TokenTree},
        Delimiter, Token,
    };
    use ast_pretty::pprust::token_to_string;
    use rustc_errors::{PErr, PResult};

    pub struct TokenTreesReader<'a> {
        string_reader: StringReader<'a>,
        token: Token,
        diag_info: TokenTreeDiagInfo,
    }
    impl<'a> TokenTreesReader<'a> {
        pub fn parse_all_token_trees(
            string_reader: StringReader<'a>,
        ) -> (PResult<'a, TokenStream>, Vec<UnmatchedDelim>) {
            let mut tt_reader = TokenTreesReader {
                string_reader,
                token: Token::dummy(),
                diag_info: TokenTreeDiagInfo::default(),
            };
            let res = tt_reader.parse_token_trees(/* is_delimited */ false);
            (res, tt_reader.diag_info.unmatched_delims)
        }
        fn parse_token_trees(&mut self, is_delimited: bool) -> PResult<'a, TokenStream> {
            self.token = self.string_reader.next_token().0;
            let mut buf = Vec::new();
            loop {
                match self.token.kind {
                    token::OpenDelim(delim) => buf.push(self.parse_token_tree_open_delim(delim)?),
                    token::CloseDelim(delim) => {
                        return if is_delimited {
                            Ok(TokenStream::new(buf))
                        } else {
                            Err(self.close_delim_err(delim))
                        };
                    },
                    token::Eof => {
                        return if is_delimited {
                            Err(self.eof_err())
                        } else {
                            Ok(TokenStream::new(buf))
                        };
                    },
                    _ => {
                        let (this_spacing, next_tok) = loop {
                            let (next_tok, is_next_tok_preceded_by_whitespace) = self.string_reader.next_token();
                            if !is_next_tok_preceded_by_whitespace {
                                if let Some(glued) = self.token.glue(&next_tok) {
                                    self.token = glued;
                                } else {
                                    let this_spacing = if next_tok.is_op() {
                                        Spacing::Joint
                                    } else {
                                        Spacing::Alone
                                    };
                                    break (this_spacing, next_tok);
                                }
                            } else {
                                break (Spacing::Alone, next_tok);
                            }
                        };
                        let this_tok = std::mem::replace(&mut self.token, next_tok);
                        buf.push(TokenTree::Token(this_tok, this_spacing));
                    },
                }
            }
        }
        fn eof_err(&mut self) -> PErr<'a> {
            let msg = "this file contains an unclosed delimiter";
            let mut err = self
                .string_reader
                .sess
                .span_diagnostic
                .struct_span_err(self.token.span, msg);
            for &(_, sp) in &self.diag_info.open_braces {
                err.span_label(sp, "unclosed delimiter");
                self.diag_info.unmatched_delims.push(UnmatchedDelim {
                    expected_delim: Delimiter::Brace,
                    found_delim: None,
                    found_span: self.token.span,
                    unclosed_span: Some(sp),
                    candidate_span: None,
                });
            }
            if let Some((delim, _)) = self.diag_info.open_braces.last() {
                report_suspicious_mismatch_block(
                    &mut err,
                    &self.diag_info,
                    &self.string_reader.sess.source_map(),
                    *delim,
                )
            }
            err
        }
        fn parse_token_tree_open_delim(&mut self, open_delim: Delimiter) -> PResult<'a, TokenTree> {
            let pre_span = self.token.span;
            self.diag_info.open_braces.push((open_delim, self.token.span));
            let tts = self.parse_token_trees(/* is_delimited */ true)?;
            let delim_span = DelimSpan::from_pair(pre_span, self.token.span);
            let sm = self.string_reader.sess.source_map();
            match self.token.kind {
                token::CloseDelim(close_delim) if close_delim == open_delim => {
                    let (open_brace, open_brace_span) = self.diag_info.open_braces.pop().unwrap();
                    let close_brace_span = self.token.span;
                    if tts.is_empty() && close_delim == Delimiter::Brace {
                        let empty_block_span = open_brace_span.to(close_brace_span);
                        if !sm.is_multiline(empty_block_span) {
                            self.diag_info.empty_block_spans.push(empty_block_span);
                        }
                    }
                    if let (Delimiter::Brace, Delimiter::Brace) = (open_brace, open_delim) {
                        self.diag_info
                            .matching_block_spans
                            .push((open_brace_span, close_brace_span));
                    }
                    self.token = self.string_reader.next_token().0;
                },
                token::CloseDelim(close_delim) => {
                    let mut unclosed_delimiter = None;
                    let mut candidate = None;
                    if self.diag_info.last_unclosed_found_span != Some(self.token.span) {
                        self.diag_info.last_unclosed_found_span = Some(self.token.span);
                        if let Some(&(_, sp)) = self.diag_info.open_braces.last() {
                            unclosed_delimiter = Some(sp);
                        };
                        for (brace, brace_span) in &self.diag_info.open_braces {
                            if same_indentation_level(&sm, self.token.span, *brace_span) && brace == &close_delim {
                                candidate = Some(*brace_span);
                            }
                        }
                        let (tok, _) = self.diag_info.open_braces.pop().unwrap();
                        self.diag_info.unmatched_delims.push(UnmatchedDelim {
                            expected_delim: tok,
                            found_delim: Some(close_delim),
                            found_span: self.token.span,
                            unclosed_span: unclosed_delimiter,
                            candidate_span: candidate,
                        });
                    } else {
                        self.diag_info.open_braces.pop();
                    }
                    if !self.diag_info.open_braces.iter().any(|&(b, _)| b == close_delim) {
                        self.token = self.string_reader.next_token().0;
                    }
                },
                token::Eof => {},
                _ => unreachable!(),
            }
            Ok(TokenTree::Delimited(delim_span, open_delim, tts))
        }
        fn close_delim_err(&mut self, delim: Delimiter) -> PErr<'a> {
            let token_str = token_to_string(&self.token);
            let msg = format!("unexpected closing delimiter: `{}`", token_str);
            let mut err = self
                .string_reader
                .sess
                .span_diagnostic
                .struct_span_err(self.token.span, msg);
            report_suspicious_mismatch_block(&mut err, &self.diag_info, &self.string_reader.sess.source_map(), delim);
            err.span_label(self.token.span, "unexpected closing delimiter");
            err
        }
    }
}
mod diagnostics {
    use super::UnmatchedDelim;
    use ast::token::Delimiter;
    use rustc_errors::Diagnostic;
    use rustc_span::{source_map::SourceMap, Span};

    #[derive(Default)]
    pub struct TokenTreeDiagInfo {
        pub open_braces: Vec<(Delimiter, Span)>,
        pub unmatched_delims: Vec<UnmatchedDelim>,
        pub last_unclosed_found_span: Option<Span>,
        pub empty_block_spans: Vec<Span>,
        pub matching_block_spans: Vec<(Span, Span)>,
    }

    pub fn same_indentation_level(sm: &SourceMap, open_sp: Span, close_sp: Span) -> bool {
        match (sm.span_to_margin(open_sp), sm.span_to_margin(close_sp)) {
            (Some(open_padding), Some(close_padding)) => open_padding == close_padding,
            _ => false,
        }
    }
    pub fn report_missing_open_delim(err: &mut Diagnostic, unmatched_delims: &[UnmatchedDelim]) -> bool {
        let mut reported_missing_open = false;
        for unmatch_brace in unmatched_delims.iter() {
            if let Some(delim) = unmatch_brace.found_delim
                && matches!(delim, Delimiter::Parenthesis | Delimiter::Bracket)
            {
                let missed_open = match delim {
                    Delimiter::Parenthesis => "(",
                    Delimiter::Bracket => "[",
                    _ => unreachable!(),
                };
                err.span_label(
                    unmatch_brace.found_span.shrink_to_lo(),
                    format!("missing open `{}` for this delimiter", missed_open),
                );
                reported_missing_open = true;
            }
        }
        reported_missing_open
    }
    pub fn report_suspicious_mismatch_block(
        err: &mut Diagnostic,
        diag_info: &TokenTreeDiagInfo,
        sm: &SourceMap,
        delim: Delimiter,
    ) {
        if report_missing_open_delim(err, &diag_info.unmatched_delims) {
            return;
        }
        let mut matched_spans: Vec<(Span, bool)> = diag_info
            .matching_block_spans
            .iter()
            .map(|&(open, close)| (open.with_hi(close.lo()), same_indentation_level(sm, open, close)))
            .collect();
        matched_spans.sort_by_key(|(span, _)| span.lo());
        for i in 0..matched_spans.len() {
            let (block_span, same_ident) = matched_spans[i];
            if same_ident {
                for j in i + 1..matched_spans.len() {
                    let (inner_block, inner_same_ident) = matched_spans[j];
                    if block_span.contains(inner_block) && !inner_same_ident {
                        matched_spans[j] = (inner_block, true);
                    }
                }
            }
        }
        let candidate_span = matched_spans
            .into_iter()
            .rev()
            .find(|&(_, same_ident)| !same_ident)
            .map(|(span, _)| span);
        if let Some(block_span) = candidate_span {
            err.span_label(
                block_span.shrink_to_lo(),
                "this delimiter might not be properly closed...",
            );
            err.span_label(
                block_span.shrink_to_hi(),
                "...as it matches this but it has different indentation",
            );
            if delim == Delimiter::Brace {
                for span in diag_info.empty_block_spans.iter() {
                    if block_span.contains(*span) {
                        err.span_label(*span, "block is empty, you might have not meant to close it");
                        break;
                    }
                }
            }
        } else {
            if let Some(parent) = diag_info.matching_block_spans.last()
                && diag_info.open_braces.last().is_none()
                && diag_info.empty_block_spans.iter().all(|&sp| sp != parent.0.to(parent.1)) {
                    err.span_label(parent.0, "this opening brace...");
                    err.span_label(parent.1, "...matches this closing brace");
            }
        }
    }
}

mod unescape_error_reporting {
    use crate::errors::{MoreThanOneCharNote, MoreThanOneCharSugg, NoBraceUnicodeSub, UnescapeError};
    use ast::lexer::unescape::{EscapeError, Mode};
    use rustc_errors::{Applicability, Handler};
    use rustc_span::{BytePos, Span};
    use std::{iter::once, ops::Range};

    pub fn emit_unescape_error(
        handler: &Handler,
        lit: &str,
        span_with_quotes: Span,
        span: Span,
        mode: Mode,
        range: Range<usize>,
        error: EscapeError,
    ) {
        debug!(
            "emit_unescape_error: {:?}, {:?}, {:?}, {:?}, {:?}",
            lit, span_with_quotes, mode, range, error
        );
        let last_char = || {
            let c = lit[range.clone()].chars().rev().next().unwrap();
            let span = span.with_lo(span.hi() - BytePos(c.len_utf8() as u32));
            (c, span)
        };
        match error {
            EscapeError::LoneSurrogateUnicodeEscape => {
                handler.emit_err(UnescapeError::InvalidUnicodeEscape { span, surrogate: true });
            },
            EscapeError::OutOfRangeUnicodeEscape => {
                handler.emit_err(UnescapeError::InvalidUnicodeEscape { span, surrogate: false });
            },
            EscapeError::MoreThanOneChar => {
                use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};
                let mut sugg = None;
                let mut note = None;
                let lit_chars = lit.chars().collect::<Vec<_>>();
                let (first, rest) = lit_chars.split_first().unwrap();
                if rest.iter().copied().all(is_combining_mark) {
                    let normalized = lit.nfc().to_string();
                    if normalized.chars().count() == 1 {
                        let ch = normalized.chars().next().unwrap().escape_default().to_string();
                        sugg = Some(MoreThanOneCharSugg::NormalizedForm { span, ch, normalized });
                    }
                    let escaped_marks = rest.iter().map(|c| c.escape_default().to_string()).collect::<Vec<_>>();
                    note = Some(MoreThanOneCharNote::AllCombining {
                        span,
                        chr: format!("{first}"),
                        len: escaped_marks.len(),
                        escaped_marks: escaped_marks.join(""),
                    });
                } else {
                    let printable: Vec<char> = lit
                        .chars()
                        .filter(|&x| unicode_width::UnicodeWidthChar::width(x).unwrap_or(0) != 0 && !x.is_whitespace())
                        .collect();
                    if let &[ch] = printable.as_slice() {
                        sugg = Some(MoreThanOneCharSugg::RemoveNonPrinting {
                            span,
                            ch: ch.to_string(),
                        });
                        note = Some(MoreThanOneCharNote::NonPrinting {
                            span,
                            escaped: lit.escape_default().to_string(),
                        });
                    }
                };
                let sugg = sugg.unwrap_or_else(|| {
                    let prefix = mode.prefix_noraw();
                    let mut escaped = String::with_capacity(lit.len());
                    let mut chrs = lit.chars().peekable();
                    while let Some(first) = chrs.next() {
                        match (first, chrs.peek()) {
                            ('\\', Some('"')) => {
                                escaped.push('\\');
                                escaped.push('"');
                                chrs.next();
                            },
                            ('"', _) => {
                                escaped.push('\\');
                                escaped.push('"')
                            },
                            (c, _) => escaped.push(c),
                        };
                    }
                    let sugg = format!("{prefix}\"{escaped}\"");
                    MoreThanOneCharSugg::Quotes {
                        span: span_with_quotes,
                        is_byte: mode == Mode::Byte,
                        sugg,
                    }
                });
                handler.emit_err(UnescapeError::MoreThanOneChar {
                    span: span_with_quotes,
                    note,
                    suggestion: sugg,
                });
            },
            EscapeError::EscapeOnlyChar => {
                let (c, char_span) = last_char();
                handler.emit_err(UnescapeError::EscapeOnlyChar {
                    span,
                    char_span,
                    escaped_sugg: c.escape_default().to_string(),
                    escaped_msg: escaped_char(c),
                    byte: mode == Mode::Byte,
                });
            },
            EscapeError::BareCarriageReturn => {
                let double_quotes = mode.in_double_quotes();
                handler.emit_err(UnescapeError::BareCr { span, double_quotes });
            },
            EscapeError::BareCarriageReturnInRawString => {
                assert!(mode.in_double_quotes());
                handler.emit_err(UnescapeError::BareCrRawString(span));
            },
            EscapeError::InvalidEscape => {
                let (c, span) = last_char();
                let label = if mode == Mode::Byte || mode == Mode::ByteStr {
                    "unknown byte escape"
                } else {
                    "unknown character escape"
                };
                let ec = escaped_char(c);
                let mut diag = handler.struct_span_err(span, format!("{}: `{}`", label, ec));
                diag.span_label(span, label);
                if c == '{' || c == '}' && matches!(mode, Mode::Str | Mode::RawStr) {
                    diag.help("if used in a formatting string, curly braces are escaped with `{{` and `}}`");
                } else if c == '\r' {
                    diag.help(
                        "this is an isolated carriage return; consider checking your editor and \
                         version control settings",
                    );
                } else {
                    if mode == Mode::Str || mode == Mode::Char {
                        diag.span_suggestion(
                            span_with_quotes,
                            "if you meant to write a literal backslash (perhaps escaping in a regular expression), consider a raw string literal",
                            format!("r\"{}\"", lit),
                            Applicability::MaybeIncorrect,
                        );
                    }
                    diag.help(
                        "for more information, visit \
                         <https://doc.rust-lang.org/reference/tokens.html#literals>",
                    );
                }
                diag.emit();
            },
            EscapeError::TooShortHexEscape => {
                handler.emit_err(UnescapeError::TooShortHexEscape(span));
            },
            EscapeError::InvalidCharInHexEscape | EscapeError::InvalidCharInUnicodeEscape => {
                let (c, span) = last_char();
                let is_hex = error == EscapeError::InvalidCharInHexEscape;
                let ch = escaped_char(c);
                handler.emit_err(UnescapeError::InvalidCharInEscape { span, is_hex, ch });
            },
            EscapeError::NonAsciiCharInByte => {
                let (c, span) = last_char();
                let desc = match mode {
                    Mode::Byte => "byte literal",
                    Mode::ByteStr => "byte string literal",
                    Mode::RawByteStr => "raw byte string literal",
                    _ => panic!("non-is_byte literal paired with NonAsciiCharInByte"),
                };
                let mut err = handler.struct_span_err(span, format!("non-ASCII character in {}", desc));
                let postfix = if unicode_width::UnicodeWidthChar::width(c).unwrap_or(1) == 0 {
                    format!(" but is {:?}", c)
                } else {
                    String::new()
                };
                err.span_label(span, format!("must be ASCII{}", postfix));
                if (c as u32) <= 0xFF && mode != Mode::RawByteStr {
                    err.span_suggestion(
                        span,
                        format!(
                            "if you meant to use the unicode code point for {:?}, use a \\xHH escape",
                            c
                        ),
                        format!("\\x{:X}", c as u32),
                        Applicability::MaybeIncorrect,
                    );
                } else if mode == Mode::Byte {
                    err.span_label(span, "this multibyte character does not fit into a single byte");
                } else if mode != Mode::RawByteStr {
                    let mut utf8 = String::new();
                    utf8.push(c);
                    err.span_suggestion(
                        span,
                        format!("if you meant to use the UTF-8 encoding of {:?}, use \\xHH escapes", c),
                        utf8.as_bytes()
                            .iter()
                            .map(|b: &u8| format!("\\x{:X}", *b))
                            .fold("".to_string(), |a, c| a + &c),
                        Applicability::MaybeIncorrect,
                    );
                }
                err.emit();
            },
            EscapeError::OutOfRangeHexEscape => {
                handler.emit_err(UnescapeError::OutOfRangeHexEscape(span));
            },
            EscapeError::LeadingUnderscoreUnicodeEscape => {
                let (c, span) = last_char();
                handler.emit_err(UnescapeError::LeadingUnderscoreUnicodeEscape {
                    span,
                    ch: escaped_char(c),
                });
            },
            EscapeError::OverlongUnicodeEscape => {
                handler.emit_err(UnescapeError::OverlongUnicodeEscape(span));
            },
            EscapeError::UnclosedUnicodeEscape => {
                handler.emit_err(UnescapeError::UnclosedUnicodeEscape(span, span.shrink_to_hi()));
            },
            EscapeError::NoBraceInUnicodeEscape => {
                let mut suggestion = "\\u{".to_owned();
                let mut suggestion_len = 0;
                let (c, char_span) = last_char();
                let chars = once(c).chain(lit[range.end..].chars());
                for c in chars.take(6).take_while(|c| c.is_digit(16)) {
                    suggestion.push(c);
                    suggestion_len += c.len_utf8();
                }
                let (label, sub) = if suggestion_len > 0 {
                    suggestion.push('}');
                    let hi = char_span.lo() + BytePos(suggestion_len as u32);
                    (
                        None,
                        NoBraceUnicodeSub::Suggestion {
                            span: span.with_hi(hi),
                            suggestion,
                        },
                    )
                } else {
                    (Some(span), NoBraceUnicodeSub::Help)
                };
                handler.emit_err(UnescapeError::NoBraceInUnicodeEscape { span, label, sub });
            },
            EscapeError::UnicodeEscapeInByte => {
                handler.emit_err(UnescapeError::UnicodeEscapeInByte(span));
            },
            EscapeError::EmptyUnicodeEscape => {
                handler.emit_err(UnescapeError::EmptyUnicodeEscape(span));
            },
            EscapeError::ZeroChars => {
                handler.emit_err(UnescapeError::ZeroChars(span));
            },
            EscapeError::LoneSlash => {
                handler.emit_err(UnescapeError::LoneSlash(span));
            },
            EscapeError::UnskippedWhitespaceWarning => {
                let (c, char_span) = last_char();
                handler.emit_warning(UnescapeError::UnskippedWhitespace {
                    span,
                    ch: escaped_char(c),
                    char_span,
                });
            },
            EscapeError::MultipleSkippedLinesWarning => {
                handler.emit_warning(UnescapeError::MultipleSkippedLinesWarning(span));
            },
        }
    }
    pub fn escaped_char(c: char) -> String {
        match c {
            '\u{20}'..='\u{7e}' => c.to_string(),
            _ => c.escape_default().to_string(),
        }
    }
}
use unescape_error_reporting::{emit_unescape_error, escaped_char};
