use super::diagnostics::report_suspicious_mismatch_block;
use super::diagnostics::same_indentation_level;
use super::diagnostics::TokenTreeDiagInfo;
use super::{StringReader, UnmatchedDelim};
use rustc_ast::token::{self, Delimiter, Token};
use rustc_ast::tokenstream::{DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast_pretty::pprust::token_to_string;
use rustc_errors::{PErr, PResult};

pub(super) struct TokenTreesReader<'a> {
    string_reader: StringReader<'a>,
    token: Token,
    diag_info: TokenTreeDiagInfo,
}

impl<'a> TokenTreesReader<'a> {
    pub(super) fn parse_all_token_trees(
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
            report_suspicious_mismatch_block(&mut err, &self.diag_info, &self.string_reader.sess.source_map(), *delim)
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
