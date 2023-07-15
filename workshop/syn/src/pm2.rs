//! [![github]](https://github.com/dtolnay/proc-macro2)&ensp;[![crates-io]](https://crates.io/crates/proc-macro2)&ensp;[![docs-rs]](crate)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs
//!
//! <br>
//!
//! A wrapper around the procedural macro API of the compiler's [`proc_macro`]
//! crate. This library serves two purposes:
//!
//! [`proc_macro`]: https://doc.rust-lang.org/proc_macro/
//!
//! - **Bring proc-macro-like functionality to other contexts like build.rs and
//!   main.rs.** Types from `proc_macro` are entirely specific to procedural
//!   macros and cannot ever exist in code outside of a procedural macro.
//!   Meanwhile `proc_macro2` types may exist anywhere including non-macro code.
//!   By developing foundational libraries like [syn] and [quote] against
//!   `proc_macro2` rather than `proc_macro`, the procedural macro ecosystem
//!   becomes easily applicable to many other use cases and we avoid
//!   reimplementing non-macro equivalents of those libraries.
//!
//! - **Make procedural macros unit testable.** As a consequence of being
//!   specific to procedural macros, nothing that uses `proc_macro` can be
//!   executed from a unit test. In order for helper libraries or components of
//!   a macro to be testable in isolation, they must be implemented using
//!   `proc_macro2`.
//!
//! [syn]: https://github.com/dtolnay/syn
//! [quote]: https://github.com/dtolnay/quote
//!
//! # Usage
//!
//! The skeleton of a typical procedural macro typically looks like this:
//!
//! ```
//! extern crate proc_macro;
//!
//! # const IGNORE: &str = stringify! {
//! #[proc_macro_derive(MyDerive)]
//! # };
//! # #[cfg(wrap_proc_macro)]
//! pub fn my_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
//!     let input = proc_macro2::TokenStream::from(input);
//!
//!     let output: proc_macro2::TokenStream = {
//!         /* transform input */
//!         # input
//!     };
//!
//!     proc_macro::TokenStream::from(output)
//! }
//! ```
//!
//! If parsing with [Syn], you'll use [`parse_macro_input!`] instead to
//! propagate parse errors correctly back to the compiler when parsing fails.
//!
//! [`parse_macro_input!`]: https://docs.rs/syn/1.0/syn/macro.parse_macro_input.html
//!
//! # Unstable features
//!
//! The default feature set of proc-macro2 tracks the most recent stable
//! compiler API. Functionality in `proc_macro` that is not yet stable is not
//! exposed by proc-macro2 by default.
//!
//! To opt into the additional APIs available in the most recent nightly
//! compiler, the `procmacro2_semver_exempt` config flag must be passed to
//! rustc. We will polyfill those nightly-only APIs back to Rust 1.31.0. As
//! these are unstable APIs that track the nightly compiler, minor versions of
//! proc-macro2 may make breaking changes to them at any time.
//!
//! ```sh
//! RUSTFLAGS='--cfg procmacro2_semver_exempt' cargo build
//! ```
//!
//! Note that this must not only be done for your crate, but for any crate that
//! depends on your crate. This infectious nature is intentional, as it serves
//! as a reminder that you are outside of the normal semver guarantees.
//!
//! Semver exempt methods are marked as such in the proc-macro2 documentation.
//!
//! # Thread-Safety
//!
//! Most types in this crate are `!Sync` because the underlying compiler
//! types make use of thread-local memory, meaning they cannot be accessed from
//! a different thread.

// Proc-macro2 types in rustdoc of other crates get linked to here.
#![doc(html_root_url = "https://docs.rs/proc-macro2/1.0.64")]
#![cfg_attr(any(proc_macro_span, super_unstable), feature(proc_macro_span))]
#![cfg_attr(super_unstable, feature(proc_macro_def_site))]
#![cfg_attr(doc_cfg, feature(doc_cfg))]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::let_underscore_untyped,
    clippy::manual_assert,
    clippy::must_use_candidate,
    clippy::needless_doctest_main,
    clippy::new_without_default,
    clippy::return_self_not_must_use,
    clippy::shadow_unrelated,
    clippy::trivially_copy_pass_by_ref,
    clippy::unnecessary_wraps,
    clippy::unused_self,
    clippy::used_underscore_binding,
    clippy::vec_init_then_push
)]

#[cfg(all(procmacro2_semver_exempt, wrap_proc_macro, not(super_unstable)))]
compile_error! {"\
    Something is not right. If you've tried to turn on \
    procmacro2_semver_exempt, you need to ensure that it \
    is turned on for the compilation of the proc-macro2 \
    build script as well.
"}

#[cfg(feature = "proc-macro")]
extern crate proc_macro;

mod marker {
    use core::marker::PhantomData;
    use std::panic::{RefUnwindSafe, UnwindSafe};
    use std::rc::Rc;

    // Zero sized marker with the correct set of autotrait impls we want all proc
    // macro types to have.
    pub(crate) type Marker = PhantomData<ProcMacroAutoTraits>;

    pub(crate) use self::value::*;

    mod value {
        pub(crate) use core::marker::PhantomData as Marker;
    }

    pub(crate) struct ProcMacroAutoTraits(Rc<()>);

    impl UnwindSafe for ProcMacroAutoTraits {}
    impl RefUnwindSafe for ProcMacroAutoTraits {}
}
mod parse {
    use crate::fallback::{
        is_ident_continue, is_ident_start, Group, LexError, Literal, Span, TokenStream, TokenStreamBuilder,
    };
    use crate::{Delimiter, Punct, Spacing, TokenTree};
    use core::char;
    use core::str::{Bytes, CharIndices, Chars};

    #[derive(Copy, Clone, Eq, PartialEq)]
    pub(crate) struct Cursor<'a> {
        pub rest: &'a str,
        #[cfg(span_locations)]
        pub off: u32,
    }

    impl<'a> Cursor<'a> {
        pub fn advance(&self, bytes: usize) -> Cursor<'a> {
            let (_front, rest) = self.rest.split_at(bytes);
            Cursor {
                rest,
                #[cfg(span_locations)]
                off: self.off + _front.chars().count() as u32,
            }
        }

        pub fn starts_with(&self, s: &str) -> bool {
            self.rest.starts_with(s)
        }

        pub fn starts_with_char(&self, ch: char) -> bool {
            self.rest.starts_with(ch)
        }

        pub fn starts_with_fn<Pattern>(&self, f: Pattern) -> bool
        where
            Pattern: FnMut(char) -> bool,
        {
            self.rest.starts_with(f)
        }

        pub fn is_empty(&self) -> bool {
            self.rest.is_empty()
        }

        fn len(&self) -> usize {
            self.rest.len()
        }

        fn as_bytes(&self) -> &'a [u8] {
            self.rest.as_bytes()
        }

        fn bytes(&self) -> Bytes<'a> {
            self.rest.bytes()
        }

        fn chars(&self) -> Chars<'a> {
            self.rest.chars()
        }

        fn char_indices(&self) -> CharIndices<'a> {
            self.rest.char_indices()
        }

        fn parse(&self, tag: &str) -> Result<Cursor<'a>, Reject> {
            if self.starts_with(tag) {
                Ok(self.advance(tag.len()))
            } else {
                Err(Reject)
            }
        }
    }

    pub(crate) struct Reject;
    type PResult<'a, O> = Result<(Cursor<'a>, O), Reject>;

    fn skip_whitespace(input: Cursor) -> Cursor {
        let mut s = input;

        while !s.is_empty() {
            let byte = s.as_bytes()[0];
            if byte == b'/' {
                if s.starts_with("//") && (!s.starts_with("///") || s.starts_with("////")) && !s.starts_with("//!") {
                    let (cursor, _) = take_until_newline_or_eof(s);
                    s = cursor;
                    continue;
                } else if s.starts_with("/**/") {
                    s = s.advance(4);
                    continue;
                } else if s.starts_with("/*")
                    && (!s.starts_with("/**") || s.starts_with("/***"))
                    && !s.starts_with("/*!")
                {
                    match block_comment(s) {
                        Ok((rest, _)) => {
                            s = rest;
                            continue;
                        },
                        Err(Reject) => return s,
                    }
                }
            }
            match byte {
                b' ' | 0x09..=0x0d => {
                    s = s.advance(1);
                    continue;
                },
                b if b.is_ascii() => {},
                _ => {
                    let ch = s.chars().next().unwrap();
                    if is_whitespace(ch) {
                        s = s.advance(ch.len_utf8());
                        continue;
                    }
                },
            }
            return s;
        }
        s
    }

    fn block_comment(input: Cursor) -> PResult<&str> {
        if !input.starts_with("/*") {
            return Err(Reject);
        }

        let mut depth = 0usize;
        let bytes = input.as_bytes();
        let mut i = 0usize;
        let upper = bytes.len() - 1;

        while i < upper {
            if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                depth += 1;
                i += 1; // eat '*'
            } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                depth -= 1;
                if depth == 0 {
                    return Ok((input.advance(i + 2), &input.rest[..i + 2]));
                }
                i += 1; // eat '/'
            }
            i += 1;
        }

        Err(Reject)
    }

    fn is_whitespace(ch: char) -> bool {
        // Rust treats left-to-right mark and right-to-left mark as whitespace
        ch.is_whitespace() || ch == '\u{200e}' || ch == '\u{200f}'
    }

    fn word_break(input: Cursor) -> Result<Cursor, Reject> {
        match input.chars().next() {
            Some(ch) if is_ident_continue(ch) => Err(Reject),
            Some(_) | None => Ok(input),
        }
    }

    pub(crate) fn token_stream(mut input: Cursor) -> Result<TokenStream, LexError> {
        let mut trees = TokenStreamBuilder::new();
        let mut stack = Vec::new();

        loop {
            input = skip_whitespace(input);

            if let Ok((rest, ())) = doc_comment(input, &mut trees) {
                input = rest;
                continue;
            }

            #[cfg(span_locations)]
            let lo = input.off;

            let first = match input.bytes().next() {
                Some(first) => first,
                None => match stack.last() {
                    None => return Ok(trees.build()),
                    #[cfg(span_locations)]
                    Some((lo, _frame)) => {
                        return Err(LexError {
                            span: Span { lo: *lo, hi: *lo },
                        })
                    },
                    #[cfg(not(span_locations))]
                    Some(_frame) => return Err(LexError { span: Span {} }),
                },
            };

            if let Some(open_delimiter) = match first {
                b'(' => Some(Delimiter::Parenthesis),
                b'[' => Some(Delimiter::Bracket),
                b'{' => Some(Delimiter::Brace),
                _ => None,
            } {
                input = input.advance(1);
                let frame = (open_delimiter, trees);
                #[cfg(span_locations)]
                let frame = (lo, frame);
                stack.push(frame);
                trees = TokenStreamBuilder::new();
            } else if let Some(close_delimiter) = match first {
                b')' => Some(Delimiter::Parenthesis),
                b']' => Some(Delimiter::Bracket),
                b'}' => Some(Delimiter::Brace),
                _ => None,
            } {
                let frame = match stack.pop() {
                    Some(frame) => frame,
                    None => return Err(lex_error(input)),
                };
                #[cfg(span_locations)]
                let (lo, frame) = frame;
                let (open_delimiter, outer) = frame;
                if open_delimiter != close_delimiter {
                    return Err(lex_error(input));
                }
                input = input.advance(1);
                let mut g = Group::new(open_delimiter, trees.build());
                g.set_span(Span {
                    #[cfg(span_locations)]
                    lo,
                    #[cfg(span_locations)]
                    hi: input.off,
                });
                trees = outer;
                trees.push_token_from_parser(TokenTree::Group(crate::Group::_new_fallback(g)));
            } else {
                let (rest, mut tt) = match leaf_token(input) {
                    Ok((rest, tt)) => (rest, tt),
                    Err(Reject) => return Err(lex_error(input)),
                };
                tt.set_span(crate::Span::_new_fallback(Span {
                    #[cfg(span_locations)]
                    lo,
                    #[cfg(span_locations)]
                    hi: rest.off,
                }));
                trees.push_token_from_parser(tt);
                input = rest;
            }
        }
    }

    fn lex_error(cursor: Cursor) -> LexError {
        #[cfg(not(span_locations))]
        let _ = cursor;
        LexError {
            span: Span {
                #[cfg(span_locations)]
                lo: cursor.off,
                #[cfg(span_locations)]
                hi: cursor.off,
            },
        }
    }

    fn leaf_token(input: Cursor) -> PResult<TokenTree> {
        if let Ok((input, l)) = literal(input) {
            // must be parsed before ident
            Ok((input, TokenTree::Literal(crate::Literal::_new_fallback(l))))
        } else if let Ok((input, p)) = punct(input) {
            Ok((input, TokenTree::Punct(p)))
        } else if let Ok((input, i)) = ident(input) {
            Ok((input, TokenTree::Ident(i)))
        } else {
            Err(Reject)
        }
    }

    fn ident(input: Cursor) -> PResult<crate::Ident> {
        if ["r\"", "r#\"", "r##", "b\"", "b\'", "br\"", "br#", "c\"", "cr\"", "cr#"]
            .iter()
            .any(|prefix| input.starts_with(prefix))
        {
            Err(Reject)
        } else {
            ident_any(input)
        }
    }

    fn ident_any(input: Cursor) -> PResult<crate::Ident> {
        let raw = input.starts_with("r#");
        let rest = input.advance((raw as usize) << 1);

        let (rest, sym) = ident_not_raw(rest)?;

        if !raw {
            let ident = crate::Ident::new(sym, crate::Span::call_site());
            return Ok((rest, ident));
        }

        match sym {
            "_" | "super" | "self" | "Self" | "crate" => return Err(Reject),
            _ => {},
        }

        let ident = crate::Ident::_new_raw(sym, crate::Span::call_site());
        Ok((rest, ident))
    }

    fn ident_not_raw(input: Cursor) -> PResult<&str> {
        let mut chars = input.char_indices();

        match chars.next() {
            Some((_, ch)) if is_ident_start(ch) => {},
            _ => return Err(Reject),
        }

        let mut end = input.len();
        for (i, ch) in chars {
            if !is_ident_continue(ch) {
                end = i;
                break;
            }
        }

        Ok((input.advance(end), &input.rest[..end]))
    }

    pub(crate) fn literal(input: Cursor) -> PResult<Literal> {
        let rest = literal_nocapture(input)?;
        let end = input.len() - rest.len();
        Ok((rest, Literal::_new(input.rest[..end].to_string())))
    }

    fn literal_nocapture(input: Cursor) -> Result<Cursor, Reject> {
        if let Ok(ok) = string(input) {
            Ok(ok)
        } else if let Ok(ok) = byte_string(input) {
            Ok(ok)
        } else if let Ok(ok) = c_string(input) {
            Ok(ok)
        } else if let Ok(ok) = byte(input) {
            Ok(ok)
        } else if let Ok(ok) = character(input) {
            Ok(ok)
        } else if let Ok(ok) = float(input) {
            Ok(ok)
        } else if let Ok(ok) = int(input) {
            Ok(ok)
        } else {
            Err(Reject)
        }
    }

    fn literal_suffix(input: Cursor) -> Cursor {
        match ident_not_raw(input) {
            Ok((input, _)) => input,
            Err(Reject) => input,
        }
    }

    fn string(input: Cursor) -> Result<Cursor, Reject> {
        if let Ok(input) = input.parse("\"") {
            cooked_string(input)
        } else if let Ok(input) = input.parse("r") {
            raw_string(input)
        } else {
            Err(Reject)
        }
    }

    fn cooked_string(mut input: Cursor) -> Result<Cursor, Reject> {
        let mut chars = input.char_indices();

        while let Some((i, ch)) = chars.next() {
            match ch {
                '"' => {
                    let input = input.advance(i + 1);
                    return Ok(literal_suffix(input));
                },
                '\r' => match chars.next() {
                    Some((_, '\n')) => {},
                    _ => break,
                },
                '\\' => match chars.next() {
                    Some((_, 'x')) => {
                        backslash_x_char(&mut chars)?;
                    },
                    Some((_, 'n')) | Some((_, 'r')) | Some((_, 't')) | Some((_, '\\')) | Some((_, '\''))
                    | Some((_, '"')) | Some((_, '0')) => {},
                    Some((_, 'u')) => {
                        backslash_u(&mut chars)?;
                    },
                    Some((newline, ch @ '\n')) | Some((newline, ch @ '\r')) => {
                        input = input.advance(newline + 1);
                        trailing_backslash(&mut input, ch as u8)?;
                        chars = input.char_indices();
                    },
                    _ => break,
                },
                _ch => {},
            }
        }
        Err(Reject)
    }

    fn raw_string(input: Cursor) -> Result<Cursor, Reject> {
        let (input, delimiter) = delimiter_of_raw_string(input)?;
        let mut bytes = input.bytes().enumerate();
        while let Some((i, byte)) = bytes.next() {
            match byte {
                b'"' if input.rest[i + 1..].starts_with(delimiter) => {
                    let rest = input.advance(i + 1 + delimiter.len());
                    return Ok(literal_suffix(rest));
                },
                b'\r' => match bytes.next() {
                    Some((_, b'\n')) => {},
                    _ => break,
                },
                _ => {},
            }
        }
        Err(Reject)
    }

    fn byte_string(input: Cursor) -> Result<Cursor, Reject> {
        if let Ok(input) = input.parse("b\"") {
            cooked_byte_string(input)
        } else if let Ok(input) = input.parse("br") {
            raw_byte_string(input)
        } else {
            Err(Reject)
        }
    }

    fn cooked_byte_string(mut input: Cursor) -> Result<Cursor, Reject> {
        let mut bytes = input.bytes().enumerate();
        while let Some((offset, b)) = bytes.next() {
            match b {
                b'"' => {
                    let input = input.advance(offset + 1);
                    return Ok(literal_suffix(input));
                },
                b'\r' => match bytes.next() {
                    Some((_, b'\n')) => {},
                    _ => break,
                },
                b'\\' => match bytes.next() {
                    Some((_, b'x')) => {
                        backslash_x_byte(&mut bytes)?;
                    },
                    Some((_, b'n')) | Some((_, b'r')) | Some((_, b't')) | Some((_, b'\\')) | Some((_, b'0'))
                    | Some((_, b'\'')) | Some((_, b'"')) => {},
                    Some((newline, b @ b'\n')) | Some((newline, b @ b'\r')) => {
                        input = input.advance(newline + 1);
                        trailing_backslash(&mut input, b)?;
                        bytes = input.bytes().enumerate();
                    },
                    _ => break,
                },
                b if b.is_ascii() => {},
                _ => break,
            }
        }
        Err(Reject)
    }

    fn delimiter_of_raw_string(input: Cursor) -> PResult<&str> {
        for (i, byte) in input.bytes().enumerate() {
            match byte {
                b'"' => {
                    if i > 255 {
                        // https://github.com/rust-lang/rust/pull/95251
                        return Err(Reject);
                    }
                    return Ok((input.advance(i + 1), &input.rest[..i]));
                },
                b'#' => {},
                _ => break,
            }
        }
        Err(Reject)
    }

    fn raw_byte_string(input: Cursor) -> Result<Cursor, Reject> {
        let (input, delimiter) = delimiter_of_raw_string(input)?;
        let mut bytes = input.bytes().enumerate();
        while let Some((i, byte)) = bytes.next() {
            match byte {
                b'"' if input.rest[i + 1..].starts_with(delimiter) => {
                    let rest = input.advance(i + 1 + delimiter.len());
                    return Ok(literal_suffix(rest));
                },
                b'\r' => match bytes.next() {
                    Some((_, b'\n')) => {},
                    _ => break,
                },
                other => {
                    if !other.is_ascii() {
                        break;
                    }
                },
            }
        }
        Err(Reject)
    }

    fn c_string(input: Cursor) -> Result<Cursor, Reject> {
        if let Ok(input) = input.parse("c\"") {
            cooked_c_string(input)
        } else if let Ok(input) = input.parse("cr") {
            raw_c_string(input)
        } else {
            Err(Reject)
        }
    }

    fn raw_c_string(input: Cursor) -> Result<Cursor, Reject> {
        let (input, delimiter) = delimiter_of_raw_string(input)?;
        let mut bytes = input.bytes().enumerate();
        while let Some((i, byte)) = bytes.next() {
            match byte {
                b'"' if input.rest[i + 1..].starts_with(delimiter) => {
                    let rest = input.advance(i + 1 + delimiter.len());
                    return Ok(literal_suffix(rest));
                },
                b'\r' => match bytes.next() {
                    Some((_, b'\n')) => {},
                    _ => break,
                },
                b'\0' => break,
                _ => {},
            }
        }
        Err(Reject)
    }

    fn cooked_c_string(mut input: Cursor) -> Result<Cursor, Reject> {
        let mut chars = input.char_indices();

        while let Some((i, ch)) = chars.next() {
            match ch {
                '"' => {
                    let input = input.advance(i + 1);
                    return Ok(literal_suffix(input));
                },
                '\r' => match chars.next() {
                    Some((_, '\n')) => {},
                    _ => break,
                },
                '\\' => match chars.next() {
                    Some((_, 'x')) => {
                        backslash_x_nonzero(&mut chars)?;
                    },
                    Some((_, 'n')) | Some((_, 'r')) | Some((_, 't')) | Some((_, '\\')) | Some((_, '\''))
                    | Some((_, '"')) => {},
                    Some((_, 'u')) => {
                        if backslash_u(&mut chars)? == '\0' {
                            break;
                        }
                    },
                    Some((newline, ch @ '\n')) | Some((newline, ch @ '\r')) => {
                        input = input.advance(newline + 1);
                        trailing_backslash(&mut input, ch as u8)?;
                        chars = input.char_indices();
                    },
                    _ => break,
                },
                '\0' => break,
                _ch => {},
            }
        }
        Err(Reject)
    }

    fn byte(input: Cursor) -> Result<Cursor, Reject> {
        let input = input.parse("b'")?;
        let mut bytes = input.bytes().enumerate();
        let ok = match bytes.next().map(|(_, b)| b) {
            Some(b'\\') => match bytes.next().map(|(_, b)| b) {
                Some(b'x') => backslash_x_byte(&mut bytes).is_ok(),
                Some(b'n') | Some(b'r') | Some(b't') | Some(b'\\') | Some(b'0') | Some(b'\'') | Some(b'"') => true,
                _ => false,
            },
            b => b.is_some(),
        };
        if !ok {
            return Err(Reject);
        }
        let (offset, _) = bytes.next().ok_or(Reject)?;
        if !input.chars().as_str().is_char_boundary(offset) {
            return Err(Reject);
        }
        let input = input.advance(offset).parse("'")?;
        Ok(literal_suffix(input))
    }

    fn character(input: Cursor) -> Result<Cursor, Reject> {
        let input = input.parse("'")?;
        let mut chars = input.char_indices();
        let ok = match chars.next().map(|(_, ch)| ch) {
            Some('\\') => match chars.next().map(|(_, ch)| ch) {
                Some('x') => backslash_x_char(&mut chars).is_ok(),
                Some('u') => backslash_u(&mut chars).is_ok(),
                Some('n') | Some('r') | Some('t') | Some('\\') | Some('0') | Some('\'') | Some('"') => true,
                _ => false,
            },
            ch => ch.is_some(),
        };
        if !ok {
            return Err(Reject);
        }
        let (idx, _) = chars.next().ok_or(Reject)?;
        let input = input.advance(idx).parse("'")?;
        Ok(literal_suffix(input))
    }

    macro_rules! next_ch {
        ($chars:ident @ $pat:pat $(| $rest:pat)*) => {
            match $chars.next() {
                Some((_, ch)) => match ch {
                    $pat $(| $rest)* => ch,
                    _ => return Err(Reject),
                },
                None => return Err(Reject),
            }
        };
    }

    fn backslash_x_char<I>(chars: &mut I) -> Result<(), Reject>
    where
        I: Iterator<Item = (usize, char)>,
    {
        next_ch!(chars @ '0'..='7');
        next_ch!(chars @ '0'..='9' | 'a'..='f' | 'A'..='F');
        Ok(())
    }

    fn backslash_x_byte<I>(chars: &mut I) -> Result<(), Reject>
    where
        I: Iterator<Item = (usize, u8)>,
    {
        next_ch!(chars @ b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F');
        next_ch!(chars @ b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F');
        Ok(())
    }

    fn backslash_x_nonzero<I>(chars: &mut I) -> Result<(), Reject>
    where
        I: Iterator<Item = (usize, char)>,
    {
        let first = next_ch!(chars @ '0'..='9' | 'a'..='f' | 'A'..='F');
        let second = next_ch!(chars @ '0'..='9' | 'a'..='f' | 'A'..='F');
        if first == '0' && second == '0' {
            Err(Reject)
        } else {
            Ok(())
        }
    }

    fn backslash_u<I>(chars: &mut I) -> Result<char, Reject>
    where
        I: Iterator<Item = (usize, char)>,
    {
        next_ch!(chars @ '{');
        let mut value = 0;
        let mut len = 0;
        for (_, ch) in chars {
            let digit = match ch {
                '0'..='9' => ch as u8 - b'0',
                'a'..='f' => 10 + ch as u8 - b'a',
                'A'..='F' => 10 + ch as u8 - b'A',
                '_' if len > 0 => continue,
                '}' if len > 0 => return char::from_u32(value).ok_or(Reject),
                _ => break,
            };
            if len == 6 {
                break;
            }
            value *= 0x10;
            value += u32::from(digit);
            len += 1;
        }
        Err(Reject)
    }

    fn trailing_backslash(input: &mut Cursor, mut last: u8) -> Result<(), Reject> {
        let mut whitespace = input.bytes().enumerate();
        loop {
            if last == b'\r' && whitespace.next().map_or(true, |(_, b)| b != b'\n') {
                return Err(Reject);
            }
            match whitespace.next() {
                Some((_, b @ b' ')) | Some((_, b @ b'\t')) | Some((_, b @ b'\n')) | Some((_, b @ b'\r')) => {
                    last = b;
                },
                Some((offset, _)) => {
                    *input = input.advance(offset);
                    return Ok(());
                },
                None => return Err(Reject),
            }
        }
    }

    fn float(input: Cursor) -> Result<Cursor, Reject> {
        let mut rest = float_digits(input)?;
        if let Some(ch) = rest.chars().next() {
            if is_ident_start(ch) {
                rest = ident_not_raw(rest)?.0;
            }
        }
        word_break(rest)
    }

    fn float_digits(input: Cursor) -> Result<Cursor, Reject> {
        let mut chars = input.chars().peekable();
        match chars.next() {
            Some(ch) if ch >= '0' && ch <= '9' => {},
            _ => return Err(Reject),
        }

        let mut len = 1;
        let mut has_dot = false;
        let mut has_exp = false;
        while let Some(&ch) = chars.peek() {
            match ch {
                '0'..='9' | '_' => {
                    chars.next();
                    len += 1;
                },
                '.' => {
                    if has_dot {
                        break;
                    }
                    chars.next();
                    if chars.peek().map_or(false, |&ch| ch == '.' || is_ident_start(ch)) {
                        return Err(Reject);
                    }
                    len += 1;
                    has_dot = true;
                },
                'e' | 'E' => {
                    chars.next();
                    len += 1;
                    has_exp = true;
                    break;
                },
                _ => break,
            }
        }

        if !(has_dot || has_exp) {
            return Err(Reject);
        }

        if has_exp {
            let token_before_exp = if has_dot {
                Ok(input.advance(len - 1))
            } else {
                Err(Reject)
            };
            let mut has_sign = false;
            let mut has_exp_value = false;
            while let Some(&ch) = chars.peek() {
                match ch {
                    '+' | '-' => {
                        if has_exp_value {
                            break;
                        }
                        if has_sign {
                            return token_before_exp;
                        }
                        chars.next();
                        len += 1;
                        has_sign = true;
                    },
                    '0'..='9' => {
                        chars.next();
                        len += 1;
                        has_exp_value = true;
                    },
                    '_' => {
                        chars.next();
                        len += 1;
                    },
                    _ => break,
                }
            }
            if !has_exp_value {
                return token_before_exp;
            }
        }

        Ok(input.advance(len))
    }

    fn int(input: Cursor) -> Result<Cursor, Reject> {
        let mut rest = digits(input)?;
        if let Some(ch) = rest.chars().next() {
            if is_ident_start(ch) {
                rest = ident_not_raw(rest)?.0;
            }
        }
        word_break(rest)
    }

    fn digits(mut input: Cursor) -> Result<Cursor, Reject> {
        let base = if input.starts_with("0x") {
            input = input.advance(2);
            16
        } else if input.starts_with("0o") {
            input = input.advance(2);
            8
        } else if input.starts_with("0b") {
            input = input.advance(2);
            2
        } else {
            10
        };

        let mut len = 0;
        let mut empty = true;
        for b in input.bytes() {
            match b {
                b'0'..=b'9' => {
                    let digit = (b - b'0') as u64;
                    if digit >= base {
                        return Err(Reject);
                    }
                },
                b'a'..=b'f' => {
                    let digit = 10 + (b - b'a') as u64;
                    if digit >= base {
                        break;
                    }
                },
                b'A'..=b'F' => {
                    let digit = 10 + (b - b'A') as u64;
                    if digit >= base {
                        break;
                    }
                },
                b'_' => {
                    if empty && base == 10 {
                        return Err(Reject);
                    }
                    len += 1;
                    continue;
                },
                _ => break,
            };
            len += 1;
            empty = false;
        }
        if empty {
            Err(Reject)
        } else {
            Ok(input.advance(len))
        }
    }

    fn punct(input: Cursor) -> PResult<Punct> {
        let (rest, ch) = punct_char(input)?;
        if ch == '\'' {
            if ident_any(rest)?.0.starts_with_char('\'') {
                Err(Reject)
            } else {
                Ok((rest, Punct::new('\'', Spacing::Joint)))
            }
        } else {
            let kind = match punct_char(rest) {
                Ok(_) => Spacing::Joint,
                Err(Reject) => Spacing::Alone,
            };
            Ok((rest, Punct::new(ch, kind)))
        }
    }

    fn punct_char(input: Cursor) -> PResult<char> {
        if input.starts_with("//") || input.starts_with("/*") {
            // Do not accept `/` of a comment as a punct.
            return Err(Reject);
        }

        let mut chars = input.chars();
        let first = match chars.next() {
            Some(ch) => ch,
            None => {
                return Err(Reject);
            },
        };
        let recognized = "~!@#$%^&*-=+|;:,<.>/?'";
        if recognized.contains(first) {
            Ok((input.advance(first.len_utf8()), first))
        } else {
            Err(Reject)
        }
    }

    fn doc_comment<'a>(input: Cursor<'a>, trees: &mut TokenStreamBuilder) -> PResult<'a, ()> {
        #[cfg(span_locations)]
        let lo = input.off;
        let (rest, (comment, inner)) = doc_comment_contents(input)?;
        let span = crate::Span::_new_fallback(Span {
            #[cfg(span_locations)]
            lo,
            #[cfg(span_locations)]
            hi: rest.off,
        });

        let mut scan_for_bare_cr = comment;
        while let Some(cr) = scan_for_bare_cr.find('\r') {
            let rest = &scan_for_bare_cr[cr + 1..];
            if !rest.starts_with('\n') {
                return Err(Reject);
            }
            scan_for_bare_cr = rest;
        }

        let mut pound = Punct::new('#', Spacing::Alone);
        pound.set_span(span);
        trees.push_token_from_parser(TokenTree::Punct(pound));

        if inner {
            let mut bang = Punct::new('!', Spacing::Alone);
            bang.set_span(span);
            trees.push_token_from_parser(TokenTree::Punct(bang));
        }

        let doc_ident = crate::Ident::new("doc", span);
        let mut equal = Punct::new('=', Spacing::Alone);
        equal.set_span(span);
        let mut literal = crate::Literal::string(comment);
        literal.set_span(span);
        let mut bracketed = TokenStreamBuilder::with_capacity(3);
        bracketed.push_token_from_parser(TokenTree::Ident(doc_ident));
        bracketed.push_token_from_parser(TokenTree::Punct(equal));
        bracketed.push_token_from_parser(TokenTree::Literal(literal));
        let group = Group::new(Delimiter::Bracket, bracketed.build());
        let mut group = crate::Group::_new_fallback(group);
        group.set_span(span);
        trees.push_token_from_parser(TokenTree::Group(group));

        Ok((rest, ()))
    }

    fn doc_comment_contents(input: Cursor) -> PResult<(&str, bool)> {
        if input.starts_with("//!") {
            let input = input.advance(3);
            let (input, s) = take_until_newline_or_eof(input);
            Ok((input, (s, true)))
        } else if input.starts_with("/*!") {
            let (input, s) = block_comment(input)?;
            Ok((input, (&s[3..s.len() - 2], true)))
        } else if input.starts_with("///") {
            let input = input.advance(3);
            if input.starts_with_char('/') {
                return Err(Reject);
            }
            let (input, s) = take_until_newline_or_eof(input);
            Ok((input, (s, false)))
        } else if input.starts_with("/**") && !input.rest[3..].starts_with('*') {
            let (input, s) = block_comment(input)?;
            Ok((input, (&s[3..s.len() - 2], false)))
        } else {
            Err(Reject)
        }
    }

    fn take_until_newline_or_eof(input: Cursor) -> (Cursor, &str) {
        let chars = input.char_indices();

        for (i, ch) in chars {
            if ch == '\n' {
                return (input.advance(i), &input.rest[..i]);
            } else if ch == '\r' && input.rest[i + 1..].starts_with('\n') {
                return (input.advance(i + 1), &input.rest[..i]);
            }
        }

        (input.advance(input.len()), input.rest)
    }
}
mod rcvec {
    use core::mem;
    use core::slice;
    use std::panic::RefUnwindSafe;
    use std::rc::Rc;
    use std::vec;

    pub(crate) struct RcVec<T> {
        inner: Rc<Vec<T>>,
    }

    pub(crate) struct RcVecBuilder<T> {
        inner: Vec<T>,
    }

    pub(crate) struct RcVecMut<'a, T> {
        inner: &'a mut Vec<T>,
    }

    #[derive(Clone)]
    pub(crate) struct RcVecIntoIter<T> {
        inner: vec::IntoIter<T>,
    }

    impl<T> RcVec<T> {
        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        pub fn len(&self) -> usize {
            self.inner.len()
        }

        pub fn iter(&self) -> slice::Iter<T> {
            self.inner.iter()
        }

        pub fn make_mut(&mut self) -> RcVecMut<T>
        where
            T: Clone,
        {
            RcVecMut {
                inner: Rc::make_mut(&mut self.inner),
            }
        }

        pub fn get_mut(&mut self) -> Option<RcVecMut<T>> {
            let inner = Rc::get_mut(&mut self.inner)?;
            Some(RcVecMut { inner })
        }

        pub fn make_owned(mut self) -> RcVecBuilder<T>
        where
            T: Clone,
        {
            let vec = if let Some(owned) = Rc::get_mut(&mut self.inner) {
                mem::replace(owned, Vec::new())
            } else {
                Vec::clone(&self.inner)
            };
            RcVecBuilder { inner: vec }
        }
    }

    impl<T> RcVecBuilder<T> {
        pub fn new() -> Self {
            RcVecBuilder { inner: Vec::new() }
        }

        pub fn with_capacity(cap: usize) -> Self {
            RcVecBuilder {
                inner: Vec::with_capacity(cap),
            }
        }

        pub fn push(&mut self, element: T) {
            self.inner.push(element);
        }

        pub fn extend(&mut self, iter: impl IntoIterator<Item = T>) {
            self.inner.extend(iter);
        }

        pub fn as_mut(&mut self) -> RcVecMut<T> {
            RcVecMut { inner: &mut self.inner }
        }

        pub fn build(self) -> RcVec<T> {
            RcVec {
                inner: Rc::new(self.inner),
            }
        }
    }

    impl<'a, T> RcVecMut<'a, T> {
        pub fn push(&mut self, element: T) {
            self.inner.push(element);
        }

        pub fn extend(&mut self, iter: impl IntoIterator<Item = T>) {
            self.inner.extend(iter);
        }

        pub fn pop(&mut self) -> Option<T> {
            self.inner.pop()
        }

        pub fn as_mut(&mut self) -> RcVecMut<T> {
            RcVecMut { inner: self.inner }
        }
    }

    impl<T> Clone for RcVec<T> {
        fn clone(&self) -> Self {
            RcVec {
                inner: Rc::clone(&self.inner),
            }
        }
    }

    impl<T> IntoIterator for RcVecBuilder<T> {
        type Item = T;
        type IntoIter = RcVecIntoIter<T>;

        fn into_iter(self) -> Self::IntoIter {
            RcVecIntoIter {
                inner: self.inner.into_iter(),
            }
        }
    }

    impl<T> Iterator for RcVecIntoIter<T> {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }

    impl<T> RefUnwindSafe for RcVec<T> where T: RefUnwindSafe {}
}

#[cfg(wrap_proc_macro)]
mod detection {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Once;

    static WORKS: AtomicUsize = AtomicUsize::new(0);
    static INIT: Once = Once::new();

    pub(crate) fn inside_proc_macro() -> bool {
        match WORKS.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {},
        }

        INIT.call_once(initialize);
        inside_proc_macro()
    }

    pub(crate) fn force_fallback() {
        WORKS.store(1, Ordering::Relaxed);
    }

    pub(crate) fn unforce_fallback() {
        initialize();
    }

    #[cfg(not(no_is_available))]
    fn initialize() {
        let available = proc_macro::is_available();
        WORKS.store(available as usize + 1, Ordering::Relaxed);
    }

    // Swap in a null panic hook to avoid printing "thread panicked" to stderr,
    // then use catch_unwind to determine whether the compiler's proc_macro is
    // working. When proc-macro2 is used from outside of a procedural macro all
    // of the proc_macro crate's APIs currently panic.
    //
    // The Once is to prevent the possibility of this ordering:
    //
    //     thread 1 calls take_hook, gets the user's original hook
    //     thread 1 calls set_hook with the null hook
    //     thread 2 calls take_hook, thinks null hook is the original hook
    //     thread 2 calls set_hook with the null hook
    //     thread 1 calls set_hook with the actual original hook
    //     thread 2 calls set_hook with what it thinks is the original hook
    //
    // in which the user's hook has been lost.
    //
    // There is still a race condition where a panic in a different thread can
    // happen during the interval that the user's original panic hook is
    // unregistered such that their hook is incorrectly not called. This is
    // sufficiently unlikely and less bad than printing panic messages to stderr
    // on correct use of this crate. Maybe there is a libstd feature request
    // here. For now, if a user needs to guarantee that this failure mode does
    // not occur, they need to call e.g. `proc_macro2::Span::call_site()` from
    // the main thread before launching any other threads.
    #[cfg(no_is_available)]
    fn initialize() {
        use std::panic::{self, PanicInfo};

        type PanicHook = dyn Fn(&PanicInfo) + Sync + Send + 'static;

        let null_hook: Box<PanicHook> = Box::new(|_panic_info| { /* ignore */ });
        let sanity_check = &*null_hook as *const PanicHook;
        let original_hook = panic::take_hook();
        panic::set_hook(null_hook);

        let works = panic::catch_unwind(proc_macro::Span::call_site).is_ok();
        WORKS.store(works as usize + 1, Ordering::Relaxed);

        let hopefully_null_hook = panic::take_hook();
        panic::set_hook(original_hook);
        if sanity_check != &*hopefully_null_hook {
            panic!("observed race condition in proc_macro2::inside_proc_macro");
        }
    }
}

// Public for proc_macro2::fallback::force() and unforce(), but those are quite
// a niche use case so we omit it from rustdoc.
#[doc(hidden)]
pub mod fallback {
    #[cfg(span_locations)]
    use crate::location::LineColumn;
    use crate::parse::{self, Cursor};
    use crate::rcvec::{RcVec, RcVecBuilder, RcVecIntoIter, RcVecMut};
    use crate::{Delimiter, Spacing, TokenTree};
    #[cfg(all(span_locations, not(fuzzing)))]
    use core::cell::RefCell;
    #[cfg(span_locations)]
    use core::cmp;
    use core::fmt::{self, Debug, Display, Write};
    use core::iter::FromIterator;
    use core::mem::ManuallyDrop;
    use core::ops::RangeBounds;
    use core::ptr;
    use core::str::FromStr;
    use std::path::PathBuf;

    /// Force use of proc-macro2's fallback implementation of the API for now, even
    /// if the compiler's implementation is available.
    pub fn force() {
        #[cfg(wrap_proc_macro)]
        crate::detection::force_fallback();
    }

    /// Resume using the compiler's implementation of the proc macro API if it is
    /// available.
    pub fn unforce() {
        #[cfg(wrap_proc_macro)]
        crate::detection::unforce_fallback();
    }

    #[derive(Clone)]
    pub(crate) struct TokenStream {
        inner: RcVec<TokenTree>,
    }

    #[derive(Debug)]
    pub(crate) struct LexError {
        pub(crate) span: Span,
    }

    impl LexError {
        pub(crate) fn span(&self) -> Span {
            self.span
        }

        fn call_site() -> Self {
            LexError {
                span: Span::call_site(),
            }
        }
    }

    impl TokenStream {
        pub fn new() -> Self {
            TokenStream {
                inner: RcVecBuilder::new().build(),
            }
        }

        pub fn is_empty(&self) -> bool {
            self.inner.len() == 0
        }

        fn take_inner(self) -> RcVecBuilder<TokenTree> {
            let nodrop = ManuallyDrop::new(self);
            unsafe { ptr::read(&nodrop.inner) }.make_owned()
        }
    }

    fn push_token_from_proc_macro(mut vec: RcVecMut<TokenTree>, token: TokenTree) {
        // https://github.com/dtolnay/proc-macro2/issues/235
        match token {
            #[cfg(not(no_bind_by_move_pattern_guard))]
            TokenTree::Literal(crate::Literal {
                #[cfg(wrap_proc_macro)]
                    inner: crate::imp::Literal::Fallback(literal),
                #[cfg(not(wrap_proc_macro))]
                    inner: literal,
                ..
            }) if literal.repr.starts_with('-') => {
                push_negative_literal(vec, literal);
            },
            #[cfg(no_bind_by_move_pattern_guard)]
            TokenTree::Literal(crate::Literal {
                #[cfg(wrap_proc_macro)]
                    inner: crate::imp::Literal::Fallback(literal),
                #[cfg(not(wrap_proc_macro))]
                    inner: literal,
                ..
            }) => {
                if literal.repr.starts_with('-') {
                    push_negative_literal(vec, literal);
                } else {
                    vec.push(TokenTree::Literal(crate::Literal::_new_fallback(literal)));
                }
            },
            _ => vec.push(token),
        }

        #[cold]
        fn push_negative_literal(mut vec: RcVecMut<TokenTree>, mut literal: Literal) {
            literal.repr.remove(0);
            let mut punct = crate::Punct::new('-', Spacing::Alone);
            punct.set_span(crate::Span::_new_fallback(literal.span));
            vec.push(TokenTree::Punct(punct));
            vec.push(TokenTree::Literal(crate::Literal::_new_fallback(literal)));
        }
    }

    // Nonrecursive to prevent stack overflow.
    impl Drop for TokenStream {
        fn drop(&mut self) {
            let mut inner = match self.inner.get_mut() {
                Some(inner) => inner,
                None => return,
            };
            while let Some(token) = inner.pop() {
                let group = match token {
                    TokenTree::Group(group) => group.inner,
                    _ => continue,
                };
                #[cfg(wrap_proc_macro)]
                let group = match group {
                    crate::imp::Group::Fallback(group) => group,
                    crate::imp::Group::Compiler(_) => continue,
                };
                inner.extend(group.stream.take_inner());
            }
        }
    }

    pub(crate) struct TokenStreamBuilder {
        inner: RcVecBuilder<TokenTree>,
    }

    impl TokenStreamBuilder {
        pub fn new() -> Self {
            TokenStreamBuilder {
                inner: RcVecBuilder::new(),
            }
        }

        pub fn with_capacity(cap: usize) -> Self {
            TokenStreamBuilder {
                inner: RcVecBuilder::with_capacity(cap),
            }
        }

        pub fn push_token_from_parser(&mut self, tt: TokenTree) {
            self.inner.push(tt);
        }

        pub fn build(self) -> TokenStream {
            TokenStream {
                inner: self.inner.build(),
            }
        }
    }

    #[cfg(span_locations)]
    fn get_cursor(src: &str) -> Cursor {
        #[cfg(fuzzing)]
        return Cursor { rest: src, off: 1 };

        // Create a dummy file & add it to the source map
        #[cfg(not(fuzzing))]
        SOURCE_MAP.with(|cm| {
            let mut cm = cm.borrow_mut();
            let span = cm.add_file(src);
            Cursor {
                rest: src,
                off: span.lo,
            }
        })
    }

    #[cfg(not(span_locations))]
    fn get_cursor(src: &str) -> Cursor {
        Cursor { rest: src }
    }

    impl FromStr for TokenStream {
        type Err = LexError;

        fn from_str(src: &str) -> Result<TokenStream, LexError> {
            // Create a dummy file & add it to the source map
            let mut cursor = get_cursor(src);

            // Strip a byte order mark if present
            const BYTE_ORDER_MARK: &str = "\u{feff}";
            if cursor.starts_with(BYTE_ORDER_MARK) {
                cursor = cursor.advance(BYTE_ORDER_MARK.len());
            }

            parse::token_stream(cursor)
        }
    }

    impl Display for LexError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("cannot parse string into token stream")
        }
    }

    impl Display for TokenStream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut joint = false;
            for (i, tt) in self.inner.iter().enumerate() {
                if i != 0 && !joint {
                    write!(f, " ")?;
                }
                joint = false;
                match tt {
                    TokenTree::Group(tt) => Display::fmt(tt, f),
                    TokenTree::Ident(tt) => Display::fmt(tt, f),
                    TokenTree::Punct(tt) => {
                        joint = tt.spacing() == Spacing::Joint;
                        Display::fmt(tt, f)
                    },
                    TokenTree::Literal(tt) => Display::fmt(tt, f),
                }?;
            }

            Ok(())
        }
    }

    impl Debug for TokenStream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("TokenStream ")?;
            f.debug_list().entries(self.clone()).finish()
        }
    }

    #[cfg(feature = "proc-macro")]
    impl From<proc_macro::TokenStream> for TokenStream {
        fn from(inner: proc_macro::TokenStream) -> Self {
            inner.to_string().parse().expect("compiler token stream parse failed")
        }
    }

    #[cfg(feature = "proc-macro")]
    impl From<TokenStream> for proc_macro::TokenStream {
        fn from(inner: TokenStream) -> Self {
            inner.to_string().parse().expect("failed to parse to compiler tokens")
        }
    }

    impl From<TokenTree> for TokenStream {
        fn from(tree: TokenTree) -> Self {
            let mut stream = RcVecBuilder::new();
            push_token_from_proc_macro(stream.as_mut(), tree);
            TokenStream { inner: stream.build() }
        }
    }

    impl FromIterator<TokenTree> for TokenStream {
        fn from_iter<I: IntoIterator<Item = TokenTree>>(tokens: I) -> Self {
            let mut stream = TokenStream::new();
            stream.extend(tokens);
            stream
        }
    }

    impl FromIterator<TokenStream> for TokenStream {
        fn from_iter<I: IntoIterator<Item = TokenStream>>(streams: I) -> Self {
            let mut v = RcVecBuilder::new();

            for stream in streams {
                v.extend(stream.take_inner());
            }

            TokenStream { inner: v.build() }
        }
    }

    impl Extend<TokenTree> for TokenStream {
        fn extend<I: IntoIterator<Item = TokenTree>>(&mut self, tokens: I) {
            let mut vec = self.inner.make_mut();
            tokens
                .into_iter()
                .for_each(|token| push_token_from_proc_macro(vec.as_mut(), token));
        }
    }

    impl Extend<TokenStream> for TokenStream {
        fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, streams: I) {
            self.inner.make_mut().extend(streams.into_iter().flatten());
        }
    }

    pub(crate) type TokenTreeIter = RcVecIntoIter<TokenTree>;

    impl IntoIterator for TokenStream {
        type Item = TokenTree;
        type IntoIter = TokenTreeIter;

        fn into_iter(self) -> TokenTreeIter {
            self.take_inner().into_iter()
        }
    }

    #[derive(Clone, PartialEq, Eq)]
    pub(crate) struct SourceFile {
        path: PathBuf,
    }

    impl SourceFile {
        /// Get the path to this source file as a string.
        pub fn path(&self) -> PathBuf {
            self.path.clone()
        }

        pub fn is_real(&self) -> bool {
            // XXX(nika): Support real files in the future?
            false
        }
    }

    impl Debug for SourceFile {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.debug_struct("SourceFile")
                .field("path", &self.path())
                .field("is_real", &self.is_real())
                .finish()
        }
    }

    #[cfg(all(span_locations, not(fuzzing)))]
    thread_local! {
        static SOURCE_MAP: RefCell<SourceMap> = RefCell::new(SourceMap {
            // NOTE: We start with a single dummy file which all call_site() and
            // def_site() spans reference.
            files: vec![FileInfo {
                source_text: String::new(),
                span: Span { lo: 0, hi: 0 },
                lines: vec![0],
            }],
        });
    }

    #[cfg(all(span_locations, not(fuzzing)))]
    struct FileInfo {
        source_text: String,
        span: Span,
        lines: Vec<usize>,
    }

    #[cfg(all(span_locations, not(fuzzing)))]
    impl FileInfo {
        fn offset_line_column(&self, offset: usize) -> LineColumn {
            assert!(self.span_within(Span {
                lo: offset as u32,
                hi: offset as u32
            }));
            let offset = offset - self.span.lo as usize;
            match self.lines.binary_search(&offset) {
                Ok(found) => LineColumn {
                    line: found + 1,
                    column: 0,
                },
                Err(idx) => LineColumn {
                    line: idx,
                    column: offset - self.lines[idx - 1],
                },
            }
        }

        fn span_within(&self, span: Span) -> bool {
            span.lo >= self.span.lo && span.hi <= self.span.hi
        }

        fn source_text(&self, span: Span) -> String {
            let lo = (span.lo - self.span.lo) as usize;
            let hi = (span.hi - self.span.lo) as usize;
            self.source_text[lo..hi].to_owned()
        }
    }

    /// Computes the offsets of each line in the given source string
    /// and the total number of characters
    #[cfg(all(span_locations, not(fuzzing)))]
    fn lines_offsets(s: &str) -> (usize, Vec<usize>) {
        let mut lines = vec![0];
        let mut total = 0;

        for ch in s.chars() {
            total += 1;
            if ch == '\n' {
                lines.push(total);
            }
        }

        (total, lines)
    }

    #[cfg(all(span_locations, not(fuzzing)))]
    struct SourceMap {
        files: Vec<FileInfo>,
    }

    #[cfg(all(span_locations, not(fuzzing)))]
    impl SourceMap {
        fn next_start_pos(&self) -> u32 {
            // Add 1 so there's always space between files.
            //
            // We'll always have at least 1 file, as we initialize our files list
            // with a dummy file.
            self.files.last().unwrap().span.hi + 1
        }

        fn add_file(&mut self, src: &str) -> Span {
            let (len, lines) = lines_offsets(src);
            let lo = self.next_start_pos();
            // XXX(nika): Should we bother doing a checked cast or checked add here?
            let span = Span {
                lo,
                hi: lo + (len as u32),
            };

            self.files.push(FileInfo {
                source_text: src.to_owned(),
                span,
                lines,
            });

            span
        }

        #[cfg(procmacro2_semver_exempt)]
        fn filepath(&self, span: Span) -> PathBuf {
            for (i, file) in self.files.iter().enumerate() {
                if file.span_within(span) {
                    return PathBuf::from(if i == 0 {
                        "<unspecified>".to_owned()
                    } else {
                        format!("<parsed string {}>", i)
                    });
                }
            }
            unreachable!("Invalid span with no related FileInfo!");
        }

        fn fileinfo(&self, span: Span) -> &FileInfo {
            for file in &self.files {
                if file.span_within(span) {
                    return file;
                }
            }
            unreachable!("Invalid span with no related FileInfo!");
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(crate) struct Span {
        #[cfg(span_locations)]
        pub(crate) lo: u32,
        #[cfg(span_locations)]
        pub(crate) hi: u32,
    }

    impl Span {
        #[cfg(not(span_locations))]
        pub fn call_site() -> Self {
            Span {}
        }

        #[cfg(span_locations)]
        pub fn call_site() -> Self {
            Span { lo: 0, hi: 0 }
        }

        #[cfg(not(no_hygiene))]
        pub fn mixed_site() -> Self {
            Span::call_site()
        }

        #[cfg(procmacro2_semver_exempt)]
        pub fn def_site() -> Self {
            Span::call_site()
        }

        pub fn resolved_at(&self, _other: Span) -> Span {
            // Stable spans consist only of line/column information, so
            // `resolved_at` and `located_at` only select which span the
            // caller wants line/column information from.
            *self
        }

        pub fn located_at(&self, other: Span) -> Span {
            other
        }

        #[cfg(procmacro2_semver_exempt)]
        pub fn source_file(&self) -> SourceFile {
            #[cfg(fuzzing)]
            return SourceFile {
                path: PathBuf::from("<unspecified>"),
            };

            #[cfg(not(fuzzing))]
            SOURCE_MAP.with(|cm| {
                let cm = cm.borrow();
                let path = cm.filepath(*self);
                SourceFile { path }
            })
        }

        #[cfg(span_locations)]
        pub fn start(&self) -> LineColumn {
            #[cfg(fuzzing)]
            return LineColumn { line: 0, column: 0 };

            #[cfg(not(fuzzing))]
            SOURCE_MAP.with(|cm| {
                let cm = cm.borrow();
                let fi = cm.fileinfo(*self);
                fi.offset_line_column(self.lo as usize)
            })
        }

        #[cfg(span_locations)]
        pub fn end(&self) -> LineColumn {
            #[cfg(fuzzing)]
            return LineColumn { line: 0, column: 0 };

            #[cfg(not(fuzzing))]
            SOURCE_MAP.with(|cm| {
                let cm = cm.borrow();
                let fi = cm.fileinfo(*self);
                fi.offset_line_column(self.hi as usize)
            })
        }

        #[cfg(not(span_locations))]
        pub fn join(&self, _other: Span) -> Option<Span> {
            Some(Span {})
        }

        #[cfg(span_locations)]
        pub fn join(&self, other: Span) -> Option<Span> {
            #[cfg(fuzzing)]
            return {
                let _ = other;
                None
            };

            #[cfg(not(fuzzing))]
            SOURCE_MAP.with(|cm| {
                let cm = cm.borrow();
                // If `other` is not within the same FileInfo as us, return None.
                if !cm.fileinfo(*self).span_within(other) {
                    return None;
                }
                Some(Span {
                    lo: cmp::min(self.lo, other.lo),
                    hi: cmp::max(self.hi, other.hi),
                })
            })
        }

        #[cfg(not(span_locations))]
        pub fn source_text(&self) -> Option<String> {
            None
        }

        #[cfg(span_locations)]
        pub fn source_text(&self) -> Option<String> {
            #[cfg(fuzzing)]
            return None;

            #[cfg(not(fuzzing))]
            {
                if self.is_call_site() {
                    None
                } else {
                    Some(SOURCE_MAP.with(|cm| cm.borrow().fileinfo(*self).source_text(*self)))
                }
            }
        }

        #[cfg(not(span_locations))]
        pub(crate) fn first_byte(self) -> Self {
            self
        }

        #[cfg(span_locations)]
        pub(crate) fn first_byte(self) -> Self {
            Span {
                lo: self.lo,
                hi: cmp::min(self.lo.saturating_add(1), self.hi),
            }
        }

        #[cfg(not(span_locations))]
        pub(crate) fn last_byte(self) -> Self {
            self
        }

        #[cfg(span_locations)]
        pub(crate) fn last_byte(self) -> Self {
            Span {
                lo: cmp::max(self.hi.saturating_sub(1), self.lo),
                hi: self.hi,
            }
        }

        #[cfg(span_locations)]
        fn is_call_site(&self) -> bool {
            self.lo == 0 && self.hi == 0
        }
    }

    impl Debug for Span {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            #[cfg(span_locations)]
            return write!(f, "bytes({}..{})", self.lo, self.hi);

            #[cfg(not(span_locations))]
            write!(f, "Span")
        }
    }

    pub(crate) fn debug_span_field_if_nontrivial(debug: &mut fmt::DebugStruct, span: Span) {
        #[cfg(span_locations)]
        {
            if span.is_call_site() {
                return;
            }
        }

        if cfg!(span_locations) {
            debug.field("span", &span);
        }
    }

    #[derive(Clone)]
    pub(crate) struct Group {
        delimiter: Delimiter,
        stream: TokenStream,
        span: Span,
    }

    impl Group {
        pub fn new(delimiter: Delimiter, stream: TokenStream) -> Self {
            Group {
                delimiter,
                stream,
                span: Span::call_site(),
            }
        }

        pub fn delimiter(&self) -> Delimiter {
            self.delimiter
        }

        pub fn stream(&self) -> TokenStream {
            self.stream.clone()
        }

        pub fn span(&self) -> Span {
            self.span
        }

        pub fn span_open(&self) -> Span {
            self.span.first_byte()
        }

        pub fn span_close(&self) -> Span {
            self.span.last_byte()
        }

        pub fn set_span(&mut self, span: Span) {
            self.span = span;
        }
    }

    impl Display for Group {
        // We attempt to match libproc_macro's formatting.
        // Empty parens: ()
        // Nonempty parens: (...)
        // Empty brackets: []
        // Nonempty brackets: [...]
        // Empty braces: { }
        // Nonempty braces: { ... }
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let (open, close) = match self.delimiter {
                Delimiter::Parenthesis => ("(", ")"),
                Delimiter::Brace => ("{ ", "}"),
                Delimiter::Bracket => ("[", "]"),
                Delimiter::None => ("", ""),
            };

            f.write_str(open)?;
            Display::fmt(&self.stream, f)?;
            if self.delimiter == Delimiter::Brace && !self.stream.inner.is_empty() {
                f.write_str(" ")?;
            }
            f.write_str(close)?;

            Ok(())
        }
    }

    impl Debug for Group {
        fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
            let mut debug = fmt.debug_struct("Group");
            debug.field("delimiter", &self.delimiter);
            debug.field("stream", &self.stream);
            debug_span_field_if_nontrivial(&mut debug, self.span);
            debug.finish()
        }
    }

    #[derive(Clone)]
    pub(crate) struct Ident {
        sym: String,
        span: Span,
        raw: bool,
    }

    impl Ident {
        fn _new(string: &str, raw: bool, span: Span) -> Self {
            validate_ident(string, raw);

            Ident {
                sym: string.to_owned(),
                span,
                raw,
            }
        }

        pub fn new(string: &str, span: Span) -> Self {
            Ident::_new(string, false, span)
        }

        pub fn new_raw(string: &str, span: Span) -> Self {
            Ident::_new(string, true, span)
        }

        pub fn span(&self) -> Span {
            self.span
        }

        pub fn set_span(&mut self, span: Span) {
            self.span = span;
        }
    }

    pub(crate) fn is_ident_start(c: char) -> bool {
        c == '_' || unicode_ident::is_xid_start(c)
    }

    pub(crate) fn is_ident_continue(c: char) -> bool {
        unicode_ident::is_xid_continue(c)
    }

    fn validate_ident(string: &str, raw: bool) {
        if string.is_empty() {
            panic!("Ident is not allowed to be empty; use Option<Ident>");
        }

        if string.bytes().all(|digit| digit >= b'0' && digit <= b'9') {
            panic!("Ident cannot be a number; use Literal instead");
        }

        fn ident_ok(string: &str) -> bool {
            let mut chars = string.chars();
            let first = chars.next().unwrap();
            if !is_ident_start(first) {
                return false;
            }
            for ch in chars {
                if !is_ident_continue(ch) {
                    return false;
                }
            }
            true
        }

        if !ident_ok(string) {
            panic!("{:?} is not a valid Ident", string);
        }

        if raw {
            match string {
                "_" | "super" | "self" | "Self" | "crate" => {
                    panic!("`r#{}` cannot be a raw identifier", string);
                },
                _ => {},
            }
        }
    }

    impl PartialEq for Ident {
        fn eq(&self, other: &Ident) -> bool {
            self.sym == other.sym && self.raw == other.raw
        }
    }

    impl<T> PartialEq<T> for Ident
    where
        T: ?Sized + AsRef<str>,
    {
        fn eq(&self, other: &T) -> bool {
            let other = other.as_ref();
            if self.raw {
                other.starts_with("r#") && self.sym == other[2..]
            } else {
                self.sym == other
            }
        }
    }

    impl Display for Ident {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if self.raw {
                f.write_str("r#")?;
            }
            Display::fmt(&self.sym, f)
        }
    }

    #[allow(clippy::missing_fields_in_debug)]
    impl Debug for Ident {
        // Ident(proc_macro), Ident(r#union)
        #[cfg(not(span_locations))]
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut debug = f.debug_tuple("Ident");
            debug.field(&format_args!("{}", self));
            debug.finish()
        }

        // Ident {
        //     sym: proc_macro,
        //     span: bytes(128..138)
        // }
        #[cfg(span_locations)]
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut debug = f.debug_struct("Ident");
            debug.field("sym", &format_args!("{}", self));
            debug_span_field_if_nontrivial(&mut debug, self.span);
            debug.finish()
        }
    }

    #[derive(Clone)]
    pub(crate) struct Literal {
        repr: String,
        span: Span,
    }

    macro_rules! suffixed_numbers {
    ($($name:ident => $kind:ident,)*) => ($(
        pub fn $name(n: $kind) -> Literal {
            Literal::_new(format!(concat!("{}", stringify!($kind)), n))
        }
    )*)
}

    macro_rules! unsuffixed_numbers {
    ($($name:ident => $kind:ident,)*) => ($(
        pub fn $name(n: $kind) -> Literal {
            Literal::_new(n.to_string())
        }
    )*)
}

    impl Literal {
        pub(crate) fn _new(repr: String) -> Self {
            Literal {
                repr,
                span: Span::call_site(),
            }
        }

        pub(crate) unsafe fn from_str_unchecked(repr: &str) -> Self {
            Literal::_new(repr.to_owned())
        }

        suffixed_numbers! {
            u8_suffixed => u8,
            u16_suffixed => u16,
            u32_suffixed => u32,
            u64_suffixed => u64,
            u128_suffixed => u128,
            usize_suffixed => usize,
            i8_suffixed => i8,
            i16_suffixed => i16,
            i32_suffixed => i32,
            i64_suffixed => i64,
            i128_suffixed => i128,
            isize_suffixed => isize,

            f32_suffixed => f32,
            f64_suffixed => f64,
        }

        unsuffixed_numbers! {
            u8_unsuffixed => u8,
            u16_unsuffixed => u16,
            u32_unsuffixed => u32,
            u64_unsuffixed => u64,
            u128_unsuffixed => u128,
            usize_unsuffixed => usize,
            i8_unsuffixed => i8,
            i16_unsuffixed => i16,
            i32_unsuffixed => i32,
            i64_unsuffixed => i64,
            i128_unsuffixed => i128,
            isize_unsuffixed => isize,
        }

        pub fn f32_unsuffixed(f: f32) -> Literal {
            let mut s = f.to_string();
            if !s.contains('.') {
                s.push_str(".0");
            }
            Literal::_new(s)
        }

        pub fn f64_unsuffixed(f: f64) -> Literal {
            let mut s = f.to_string();
            if !s.contains('.') {
                s.push_str(".0");
            }
            Literal::_new(s)
        }

        pub fn string(t: &str) -> Literal {
            let mut repr = String::with_capacity(t.len() + 2);
            repr.push('"');
            let mut chars = t.chars();
            while let Some(ch) = chars.next() {
                if ch == '\0' {
                    repr.push_str(if chars.as_str().starts_with(|next| '0' <= next && next <= '7') {
                        // circumvent clippy::octal_escapes lint
                        "\\x00"
                    } else {
                        "\\0"
                    });
                } else if ch == '\'' {
                    // escape_debug turns this into "\'" which is unnecessary.
                    repr.push(ch);
                } else {
                    repr.extend(ch.escape_debug());
                }
            }
            repr.push('"');
            Literal::_new(repr)
        }

        pub fn character(t: char) -> Literal {
            let mut repr = String::new();
            repr.push('\'');
            if t == '"' {
                // escape_debug turns this into '\"' which is unnecessary.
                repr.push(t);
            } else {
                repr.extend(t.escape_debug());
            }
            repr.push('\'');
            Literal::_new(repr)
        }

        pub fn byte_string(bytes: &[u8]) -> Literal {
            let mut escaped = "b\"".to_string();
            let mut bytes = bytes.iter();
            while let Some(&b) = bytes.next() {
                #[allow(clippy::match_overlapping_arm)]
                match b {
                    b'\0' => escaped.push_str(match bytes.as_slice().first() {
                        // circumvent clippy::octal_escapes lint
                        Some(b'0'..=b'7') => r"\x00",
                        _ => r"\0",
                    }),
                    b'\t' => escaped.push_str(r"\t"),
                    b'\n' => escaped.push_str(r"\n"),
                    b'\r' => escaped.push_str(r"\r"),
                    b'"' => escaped.push_str("\\\""),
                    b'\\' => escaped.push_str("\\\\"),
                    b'\x20'..=b'\x7E' => escaped.push(b as char),
                    _ => {
                        let _ = write!(escaped, "\\x{:02X}", b);
                    },
                }
            }
            escaped.push('"');
            Literal::_new(escaped)
        }

        pub fn span(&self) -> Span {
            self.span
        }

        pub fn set_span(&mut self, span: Span) {
            self.span = span;
        }

        pub fn subspan<R: RangeBounds<usize>>(&self, range: R) -> Option<Span> {
            #[cfg(not(span_locations))]
            {
                let _ = range;
                None
            }

            #[cfg(span_locations)]
            {
                use crate::convert::usize_to_u32;
                use core::ops::Bound;

                let lo = match range.start_bound() {
                    Bound::Included(start) => {
                        let start = usize_to_u32(*start)?;
                        self.span.lo.checked_add(start)?
                    },
                    Bound::Excluded(start) => {
                        let start = usize_to_u32(*start)?;
                        self.span.lo.checked_add(start)?.checked_add(1)?
                    },
                    Bound::Unbounded => self.span.lo,
                };
                let hi = match range.end_bound() {
                    Bound::Included(end) => {
                        let end = usize_to_u32(*end)?;
                        self.span.lo.checked_add(end)?.checked_add(1)?
                    },
                    Bound::Excluded(end) => {
                        let end = usize_to_u32(*end)?;
                        self.span.lo.checked_add(end)?
                    },
                    Bound::Unbounded => self.span.hi,
                };
                if lo <= hi && hi <= self.span.hi {
                    Some(Span { lo, hi })
                } else {
                    None
                }
            }
        }
    }

    impl FromStr for Literal {
        type Err = LexError;

        fn from_str(repr: &str) -> Result<Self, Self::Err> {
            let mut cursor = get_cursor(repr);
            #[cfg(span_locations)]
            let lo = cursor.off;

            let negative = cursor.starts_with_char('-');
            if negative {
                cursor = cursor.advance(1);
                if !cursor.starts_with_fn(|ch| ch.is_ascii_digit()) {
                    return Err(LexError::call_site());
                }
            }

            if let Ok((rest, mut literal)) = parse::literal(cursor) {
                if rest.is_empty() {
                    if negative {
                        literal.repr.insert(0, '-');
                    }
                    literal.span = Span {
                        #[cfg(span_locations)]
                        lo,
                        #[cfg(span_locations)]
                        hi: rest.off,
                    };
                    return Ok(literal);
                }
            }
            Err(LexError::call_site())
        }
    }

    impl Display for Literal {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Display::fmt(&self.repr, f)
        }
    }

    impl Debug for Literal {
        fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
            let mut debug = fmt.debug_struct("Literal");
            debug.field("lit", &format_args!("{}", self.repr));
            debug_span_field_if_nontrivial(&mut debug, self.span);
            debug.finish()
        }
    }
}

pub mod extra {
    //! Items which do not have a correspondence to any API in the proc_macro crate,
    //! but are necessary to include in proc-macro2.

    use crate::fallback;
    use crate::imp;
    use crate::marker::Marker;
    use crate::Span;
    use core::fmt::{self, Debug};

    /// An object that holds a [`Group`]'s `span_open()` and `span_close()` together
    /// (in a more compact representation than holding those 2 spans individually.
    ///
    /// [`Group`]: crate::Group
    #[derive(Copy, Clone)]
    pub struct DelimSpan {
        inner: DelimSpanEnum,
        _marker: Marker,
    }

    #[derive(Copy, Clone)]
    enum DelimSpanEnum {
        #[cfg(wrap_proc_macro)]
        Compiler {
            join: proc_macro::Span,
            #[cfg(not(no_group_open_close))]
            open: proc_macro::Span,
            #[cfg(not(no_group_open_close))]
            close: proc_macro::Span,
        },
        Fallback(fallback::Span),
    }

    impl DelimSpan {
        pub(crate) fn new(group: &imp::Group) -> Self {
            #[cfg(wrap_proc_macro)]
            let inner = match group {
                imp::Group::Compiler(group) => DelimSpanEnum::Compiler {
                    join: group.span(),
                    #[cfg(not(no_group_open_close))]
                    open: group.span_open(),
                    #[cfg(not(no_group_open_close))]
                    close: group.span_close(),
                },
                imp::Group::Fallback(group) => DelimSpanEnum::Fallback(group.span()),
            };

            #[cfg(not(wrap_proc_macro))]
            let inner = DelimSpanEnum::Fallback(group.span());

            DelimSpan { inner, _marker: Marker }
        }

        /// Returns a span covering the entire delimited group.
        pub fn join(&self) -> Span {
            match &self.inner {
                #[cfg(wrap_proc_macro)]
                DelimSpanEnum::Compiler { join, .. } => Span::_new(imp::Span::Compiler(*join)),
                DelimSpanEnum::Fallback(span) => Span::_new_fallback(*span),
            }
        }

        /// Returns a span for the opening punctuation of the group only.
        pub fn open(&self) -> Span {
            match &self.inner {
                #[cfg(wrap_proc_macro)]
                DelimSpanEnum::Compiler {
                    #[cfg(not(no_group_open_close))]
                    open,
                    #[cfg(no_group_open_close)]
                        join: open,
                    ..
                } => Span::_new(imp::Span::Compiler(*open)),
                DelimSpanEnum::Fallback(span) => Span::_new_fallback(span.first_byte()),
            }
        }

        /// Returns a span for the closing punctuation of the group only.
        pub fn close(&self) -> Span {
            match &self.inner {
                #[cfg(wrap_proc_macro)]
                DelimSpanEnum::Compiler {
                    #[cfg(not(no_group_open_close))]
                    close,
                    #[cfg(no_group_open_close)]
                        join: close,
                    ..
                } => Span::_new(imp::Span::Compiler(*close)),
                DelimSpanEnum::Fallback(span) => Span::_new_fallback(span.last_byte()),
            }
        }
    }

    impl Debug for DelimSpan {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(&self.join(), f)
        }
    }
}

#[cfg(not(wrap_proc_macro))]
use crate::fallback as imp;
#[cfg(wrap_proc_macro)]
mod imp {
    use crate::detection::inside_proc_macro;
    #[cfg(span_locations)]
    use crate::location::LineColumn;
    use crate::{fallback, Delimiter, Punct, Spacing, TokenTree};
    use core::fmt::{self, Debug, Display};
    use core::iter::FromIterator;
    use core::ops::RangeBounds;
    use core::str::FromStr;
    use std::panic;
    #[cfg(super_unstable)]
    use std::path::PathBuf;

    #[derive(Clone)]
    pub(crate) enum TokenStream {
        Compiler(DeferredTokenStream),
        Fallback(fallback::TokenStream),
    }

    // Work around https://github.com/rust-lang/rust/issues/65080.
    // In `impl Extend<TokenTree> for TokenStream` which is used heavily by quote,
    // we hold on to the appended tokens and do proc_macro::TokenStream::extend as
    // late as possible to batch together consecutive uses of the Extend impl.
    #[derive(Clone)]
    pub(crate) struct DeferredTokenStream {
        stream: proc_macro::TokenStream,
        extra: Vec<proc_macro::TokenTree>,
    }

    pub(crate) enum LexError {
        Compiler(proc_macro::LexError),
        Fallback(fallback::LexError),
    }

    impl LexError {
        fn call_site() -> Self {
            LexError::Fallback(fallback::LexError {
                span: fallback::Span::call_site(),
            })
        }
    }

    fn mismatch() -> ! {
        panic!("compiler/fallback mismatch")
    }

    impl DeferredTokenStream {
        fn new(stream: proc_macro::TokenStream) -> Self {
            DeferredTokenStream {
                stream,
                extra: Vec::new(),
            }
        }

        fn is_empty(&self) -> bool {
            self.stream.is_empty() && self.extra.is_empty()
        }

        fn evaluate_now(&mut self) {
            // If-check provides a fast short circuit for the common case of `extra`
            // being empty, which saves a round trip over the proc macro bridge.
            // Improves macro expansion time in winrt by 6% in debug mode.
            if !self.extra.is_empty() {
                self.stream.extend(self.extra.drain(..));
            }
        }

        fn into_token_stream(mut self) -> proc_macro::TokenStream {
            self.evaluate_now();
            self.stream
        }
    }

    impl TokenStream {
        pub fn new() -> Self {
            if inside_proc_macro() {
                TokenStream::Compiler(DeferredTokenStream::new(proc_macro::TokenStream::new()))
            } else {
                TokenStream::Fallback(fallback::TokenStream::new())
            }
        }

        pub fn is_empty(&self) -> bool {
            match self {
                TokenStream::Compiler(tts) => tts.is_empty(),
                TokenStream::Fallback(tts) => tts.is_empty(),
            }
        }

        fn unwrap_nightly(self) -> proc_macro::TokenStream {
            match self {
                TokenStream::Compiler(s) => s.into_token_stream(),
                TokenStream::Fallback(_) => mismatch(),
            }
        }

        fn unwrap_stable(self) -> fallback::TokenStream {
            match self {
                TokenStream::Compiler(_) => mismatch(),
                TokenStream::Fallback(s) => s,
            }
        }
    }

    impl FromStr for TokenStream {
        type Err = LexError;

        fn from_str(src: &str) -> Result<TokenStream, LexError> {
            if inside_proc_macro() {
                Ok(TokenStream::Compiler(DeferredTokenStream::new(proc_macro_parse(src)?)))
            } else {
                Ok(TokenStream::Fallback(src.parse()?))
            }
        }
    }

    // Work around https://github.com/rust-lang/rust/issues/58736.
    fn proc_macro_parse(src: &str) -> Result<proc_macro::TokenStream, LexError> {
        let result = panic::catch_unwind(|| src.parse().map_err(LexError::Compiler));
        result.unwrap_or_else(|_| Err(LexError::call_site()))
    }

    impl Display for TokenStream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                TokenStream::Compiler(tts) => Display::fmt(&tts.clone().into_token_stream(), f),
                TokenStream::Fallback(tts) => Display::fmt(tts, f),
            }
        }
    }

    impl From<proc_macro::TokenStream> for TokenStream {
        fn from(inner: proc_macro::TokenStream) -> Self {
            TokenStream::Compiler(DeferredTokenStream::new(inner))
        }
    }

    impl From<TokenStream> for proc_macro::TokenStream {
        fn from(inner: TokenStream) -> Self {
            match inner {
                TokenStream::Compiler(inner) => inner.into_token_stream(),
                TokenStream::Fallback(inner) => inner.to_string().parse().unwrap(),
            }
        }
    }

    impl From<fallback::TokenStream> for TokenStream {
        fn from(inner: fallback::TokenStream) -> Self {
            TokenStream::Fallback(inner)
        }
    }

    // Assumes inside_proc_macro().
    fn into_compiler_token(token: TokenTree) -> proc_macro::TokenTree {
        match token {
            TokenTree::Group(tt) => tt.inner.unwrap_nightly().into(),
            TokenTree::Punct(tt) => {
                let spacing = match tt.spacing() {
                    Spacing::Joint => proc_macro::Spacing::Joint,
                    Spacing::Alone => proc_macro::Spacing::Alone,
                };
                let mut punct = proc_macro::Punct::new(tt.as_char(), spacing);
                punct.set_span(tt.span().inner.unwrap_nightly());
                punct.into()
            },
            TokenTree::Ident(tt) => tt.inner.unwrap_nightly().into(),
            TokenTree::Literal(tt) => tt.inner.unwrap_nightly().into(),
        }
    }

    impl From<TokenTree> for TokenStream {
        fn from(token: TokenTree) -> Self {
            if inside_proc_macro() {
                TokenStream::Compiler(DeferredTokenStream::new(into_compiler_token(token).into()))
            } else {
                TokenStream::Fallback(token.into())
            }
        }
    }

    impl FromIterator<TokenTree> for TokenStream {
        fn from_iter<I: IntoIterator<Item = TokenTree>>(trees: I) -> Self {
            if inside_proc_macro() {
                TokenStream::Compiler(DeferredTokenStream::new(
                    trees.into_iter().map(into_compiler_token).collect(),
                ))
            } else {
                TokenStream::Fallback(trees.into_iter().collect())
            }
        }
    }

    impl FromIterator<TokenStream> for TokenStream {
        fn from_iter<I: IntoIterator<Item = TokenStream>>(streams: I) -> Self {
            let mut streams = streams.into_iter();
            match streams.next() {
                Some(TokenStream::Compiler(mut first)) => {
                    first.evaluate_now();
                    first.stream.extend(streams.map(|s| match s {
                        TokenStream::Compiler(s) => s.into_token_stream(),
                        TokenStream::Fallback(_) => mismatch(),
                    }));
                    TokenStream::Compiler(first)
                },
                Some(TokenStream::Fallback(mut first)) => {
                    first.extend(streams.map(|s| match s {
                        TokenStream::Fallback(s) => s,
                        TokenStream::Compiler(_) => mismatch(),
                    }));
                    TokenStream::Fallback(first)
                },
                None => TokenStream::new(),
            }
        }
    }

    impl Extend<TokenTree> for TokenStream {
        fn extend<I: IntoIterator<Item = TokenTree>>(&mut self, stream: I) {
            match self {
                TokenStream::Compiler(tts) => {
                    // Here is the reason for DeferredTokenStream.
                    for token in stream {
                        tts.extra.push(into_compiler_token(token));
                    }
                },
                TokenStream::Fallback(tts) => tts.extend(stream),
            }
        }
    }

    impl Extend<TokenStream> for TokenStream {
        fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, streams: I) {
            match self {
                TokenStream::Compiler(tts) => {
                    tts.evaluate_now();
                    tts.stream.extend(streams.into_iter().map(TokenStream::unwrap_nightly));
                },
                TokenStream::Fallback(tts) => {
                    tts.extend(streams.into_iter().map(TokenStream::unwrap_stable));
                },
            }
        }
    }

    impl Debug for TokenStream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                TokenStream::Compiler(tts) => Debug::fmt(&tts.clone().into_token_stream(), f),
                TokenStream::Fallback(tts) => Debug::fmt(tts, f),
            }
        }
    }

    impl LexError {
        pub(crate) fn span(&self) -> Span {
            match self {
                LexError::Compiler(_) => Span::call_site(),
                LexError::Fallback(e) => Span::Fallback(e.span()),
            }
        }
    }

    impl From<proc_macro::LexError> for LexError {
        fn from(e: proc_macro::LexError) -> Self {
            LexError::Compiler(e)
        }
    }

    impl From<fallback::LexError> for LexError {
        fn from(e: fallback::LexError) -> Self {
            LexError::Fallback(e)
        }
    }

    impl Debug for LexError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                LexError::Compiler(e) => Debug::fmt(e, f),
                LexError::Fallback(e) => Debug::fmt(e, f),
            }
        }
    }

    impl Display for LexError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                #[cfg(not(no_lexerror_display))]
                LexError::Compiler(e) => Display::fmt(e, f),
                #[cfg(no_lexerror_display)]
                LexError::Compiler(_e) => Display::fmt(
                    &fallback::LexError {
                        span: fallback::Span::call_site(),
                    },
                    f,
                ),
                LexError::Fallback(e) => Display::fmt(e, f),
            }
        }
    }

    #[derive(Clone)]
    pub(crate) enum TokenTreeIter {
        Compiler(proc_macro::token_stream::IntoIter),
        Fallback(fallback::TokenTreeIter),
    }

    impl IntoIterator for TokenStream {
        type Item = TokenTree;
        type IntoIter = TokenTreeIter;

        fn into_iter(self) -> TokenTreeIter {
            match self {
                TokenStream::Compiler(tts) => TokenTreeIter::Compiler(tts.into_token_stream().into_iter()),
                TokenStream::Fallback(tts) => TokenTreeIter::Fallback(tts.into_iter()),
            }
        }
    }

    impl Iterator for TokenTreeIter {
        type Item = TokenTree;

        fn next(&mut self) -> Option<TokenTree> {
            let token = match self {
                TokenTreeIter::Compiler(iter) => iter.next()?,
                TokenTreeIter::Fallback(iter) => return iter.next(),
            };
            Some(match token {
                proc_macro::TokenTree::Group(tt) => crate::Group::_new(Group::Compiler(tt)).into(),
                proc_macro::TokenTree::Punct(tt) => {
                    let spacing = match tt.spacing() {
                        proc_macro::Spacing::Joint => Spacing::Joint,
                        proc_macro::Spacing::Alone => Spacing::Alone,
                    };
                    let mut o = Punct::new(tt.as_char(), spacing);
                    o.set_span(crate::Span::_new(Span::Compiler(tt.span())));
                    o.into()
                },
                proc_macro::TokenTree::Ident(s) => crate::Ident::_new(Ident::Compiler(s)).into(),
                proc_macro::TokenTree::Literal(l) => crate::Literal::_new(Literal::Compiler(l)).into(),
            })
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            match self {
                TokenTreeIter::Compiler(tts) => tts.size_hint(),
                TokenTreeIter::Fallback(tts) => tts.size_hint(),
            }
        }
    }

    #[derive(Clone, PartialEq, Eq)]
    #[cfg(super_unstable)]
    pub(crate) enum SourceFile {
        Compiler(proc_macro::SourceFile),
        Fallback(fallback::SourceFile),
    }

    #[cfg(super_unstable)]
    impl SourceFile {
        fn nightly(sf: proc_macro::SourceFile) -> Self {
            SourceFile::Compiler(sf)
        }

        /// Get the path to this source file as a string.
        pub fn path(&self) -> PathBuf {
            match self {
                SourceFile::Compiler(a) => a.path(),
                SourceFile::Fallback(a) => a.path(),
            }
        }

        pub fn is_real(&self) -> bool {
            match self {
                SourceFile::Compiler(a) => a.is_real(),
                SourceFile::Fallback(a) => a.is_real(),
            }
        }
    }

    #[cfg(super_unstable)]
    impl Debug for SourceFile {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                SourceFile::Compiler(a) => Debug::fmt(a, f),
                SourceFile::Fallback(a) => Debug::fmt(a, f),
            }
        }
    }

    #[derive(Copy, Clone)]
    pub(crate) enum Span {
        Compiler(proc_macro::Span),
        Fallback(fallback::Span),
    }

    impl Span {
        pub fn call_site() -> Self {
            if inside_proc_macro() {
                Span::Compiler(proc_macro::Span::call_site())
            } else {
                Span::Fallback(fallback::Span::call_site())
            }
        }

        #[cfg(not(no_hygiene))]
        pub fn mixed_site() -> Self {
            if inside_proc_macro() {
                Span::Compiler(proc_macro::Span::mixed_site())
            } else {
                Span::Fallback(fallback::Span::mixed_site())
            }
        }

        #[cfg(super_unstable)]
        pub fn def_site() -> Self {
            if inside_proc_macro() {
                Span::Compiler(proc_macro::Span::def_site())
            } else {
                Span::Fallback(fallback::Span::def_site())
            }
        }

        pub fn resolved_at(&self, other: Span) -> Span {
            match (self, other) {
                #[cfg(not(no_hygiene))]
                (Span::Compiler(a), Span::Compiler(b)) => Span::Compiler(a.resolved_at(b)),

                // Name resolution affects semantics, but location is only cosmetic
                #[cfg(no_hygiene)]
                (Span::Compiler(_), Span::Compiler(_)) => other,

                (Span::Fallback(a), Span::Fallback(b)) => Span::Fallback(a.resolved_at(b)),
                _ => mismatch(),
            }
        }

        pub fn located_at(&self, other: Span) -> Span {
            match (self, other) {
                #[cfg(not(no_hygiene))]
                (Span::Compiler(a), Span::Compiler(b)) => Span::Compiler(a.located_at(b)),

                // Name resolution affects semantics, but location is only cosmetic
                #[cfg(no_hygiene)]
                (Span::Compiler(_), Span::Compiler(_)) => *self,

                (Span::Fallback(a), Span::Fallback(b)) => Span::Fallback(a.located_at(b)),
                _ => mismatch(),
            }
        }

        pub fn unwrap(self) -> proc_macro::Span {
            match self {
                Span::Compiler(s) => s,
                Span::Fallback(_) => panic!("proc_macro::Span is only available in procedural macros"),
            }
        }

        #[cfg(super_unstable)]
        pub fn source_file(&self) -> SourceFile {
            match self {
                Span::Compiler(s) => SourceFile::nightly(s.source_file()),
                Span::Fallback(s) => SourceFile::Fallback(s.source_file()),
            }
        }

        #[cfg(span_locations)]
        pub fn start(&self) -> LineColumn {
            match self {
                Span::Compiler(_) => LineColumn { line: 0, column: 0 },
                Span::Fallback(s) => s.start(),
            }
        }

        #[cfg(span_locations)]
        pub fn end(&self) -> LineColumn {
            match self {
                Span::Compiler(_) => LineColumn { line: 0, column: 0 },
                Span::Fallback(s) => s.end(),
            }
        }

        pub fn join(&self, other: Span) -> Option<Span> {
            let ret = match (self, other) {
                #[cfg(proc_macro_span)]
                (Span::Compiler(a), Span::Compiler(b)) => Span::Compiler(a.join(b)?),
                (Span::Fallback(a), Span::Fallback(b)) => Span::Fallback(a.join(b)?),
                _ => return None,
            };
            Some(ret)
        }

        #[cfg(super_unstable)]
        pub fn eq(&self, other: &Span) -> bool {
            match (self, other) {
                (Span::Compiler(a), Span::Compiler(b)) => a.eq(b),
                (Span::Fallback(a), Span::Fallback(b)) => a.eq(b),
                _ => false,
            }
        }

        pub fn source_text(&self) -> Option<String> {
            match self {
                #[cfg(not(no_source_text))]
                Span::Compiler(s) => s.source_text(),
                #[cfg(no_source_text)]
                Span::Compiler(_) => None,
                Span::Fallback(s) => s.source_text(),
            }
        }

        fn unwrap_nightly(self) -> proc_macro::Span {
            match self {
                Span::Compiler(s) => s,
                Span::Fallback(_) => mismatch(),
            }
        }
    }

    impl From<proc_macro::Span> for crate::Span {
        fn from(proc_span: proc_macro::Span) -> Self {
            crate::Span::_new(Span::Compiler(proc_span))
        }
    }

    impl From<fallback::Span> for Span {
        fn from(inner: fallback::Span) -> Self {
            Span::Fallback(inner)
        }
    }

    impl Debug for Span {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Span::Compiler(s) => Debug::fmt(s, f),
                Span::Fallback(s) => Debug::fmt(s, f),
            }
        }
    }

    pub(crate) fn debug_span_field_if_nontrivial(debug: &mut fmt::DebugStruct, span: Span) {
        match span {
            Span::Compiler(s) => {
                debug.field("span", &s);
            },
            Span::Fallback(s) => fallback::debug_span_field_if_nontrivial(debug, s),
        }
    }

    #[derive(Clone)]
    pub(crate) enum Group {
        Compiler(proc_macro::Group),
        Fallback(fallback::Group),
    }

    impl Group {
        pub fn new(delimiter: Delimiter, stream: TokenStream) -> Self {
            match stream {
                TokenStream::Compiler(tts) => {
                    let delimiter = match delimiter {
                        Delimiter::Parenthesis => proc_macro::Delimiter::Parenthesis,
                        Delimiter::Bracket => proc_macro::Delimiter::Bracket,
                        Delimiter::Brace => proc_macro::Delimiter::Brace,
                        Delimiter::None => proc_macro::Delimiter::None,
                    };
                    Group::Compiler(proc_macro::Group::new(delimiter, tts.into_token_stream()))
                },
                TokenStream::Fallback(stream) => Group::Fallback(fallback::Group::new(delimiter, stream)),
            }
        }

        pub fn delimiter(&self) -> Delimiter {
            match self {
                Group::Compiler(g) => match g.delimiter() {
                    proc_macro::Delimiter::Parenthesis => Delimiter::Parenthesis,
                    proc_macro::Delimiter::Bracket => Delimiter::Bracket,
                    proc_macro::Delimiter::Brace => Delimiter::Brace,
                    proc_macro::Delimiter::None => Delimiter::None,
                },
                Group::Fallback(g) => g.delimiter(),
            }
        }

        pub fn stream(&self) -> TokenStream {
            match self {
                Group::Compiler(g) => TokenStream::Compiler(DeferredTokenStream::new(g.stream())),
                Group::Fallback(g) => TokenStream::Fallback(g.stream()),
            }
        }

        pub fn span(&self) -> Span {
            match self {
                Group::Compiler(g) => Span::Compiler(g.span()),
                Group::Fallback(g) => Span::Fallback(g.span()),
            }
        }

        pub fn span_open(&self) -> Span {
            match self {
                #[cfg(not(no_group_open_close))]
                Group::Compiler(g) => Span::Compiler(g.span_open()),
                #[cfg(no_group_open_close)]
                Group::Compiler(g) => Span::Compiler(g.span()),
                Group::Fallback(g) => Span::Fallback(g.span_open()),
            }
        }

        pub fn span_close(&self) -> Span {
            match self {
                #[cfg(not(no_group_open_close))]
                Group::Compiler(g) => Span::Compiler(g.span_close()),
                #[cfg(no_group_open_close)]
                Group::Compiler(g) => Span::Compiler(g.span()),
                Group::Fallback(g) => Span::Fallback(g.span_close()),
            }
        }

        pub fn set_span(&mut self, span: Span) {
            match (self, span) {
                (Group::Compiler(g), Span::Compiler(s)) => g.set_span(s),
                (Group::Fallback(g), Span::Fallback(s)) => g.set_span(s),
                _ => mismatch(),
            }
        }

        fn unwrap_nightly(self) -> proc_macro::Group {
            match self {
                Group::Compiler(g) => g,
                Group::Fallback(_) => mismatch(),
            }
        }
    }

    impl From<fallback::Group> for Group {
        fn from(g: fallback::Group) -> Self {
            Group::Fallback(g)
        }
    }

    impl Display for Group {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Group::Compiler(group) => Display::fmt(group, formatter),
                Group::Fallback(group) => Display::fmt(group, formatter),
            }
        }
    }

    impl Debug for Group {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Group::Compiler(group) => Debug::fmt(group, formatter),
                Group::Fallback(group) => Debug::fmt(group, formatter),
            }
        }
    }

    #[derive(Clone)]
    pub(crate) enum Ident {
        Compiler(proc_macro::Ident),
        Fallback(fallback::Ident),
    }

    impl Ident {
        pub fn new(string: &str, span: Span) -> Self {
            match span {
                Span::Compiler(s) => Ident::Compiler(proc_macro::Ident::new(string, s)),
                Span::Fallback(s) => Ident::Fallback(fallback::Ident::new(string, s)),
            }
        }

        pub fn new_raw(string: &str, span: Span) -> Self {
            match span {
                #[cfg(not(no_ident_new_raw))]
                Span::Compiler(s) => Ident::Compiler(proc_macro::Ident::new_raw(string, s)),
                #[cfg(no_ident_new_raw)]
                Span::Compiler(s) => {
                    let _ = proc_macro::Ident::new(string, s);
                    // At this point the un-r#-prefixed string is known to be a
                    // valid identifier. Try to produce a valid raw identifier by
                    // running the `TokenStream` parser, and unwrapping the first
                    // token as an `Ident`.
                    let raw_prefixed = format!("r#{}", string);
                    if let Ok(ts) = raw_prefixed.parse::<proc_macro::TokenStream>() {
                        let mut iter = ts.into_iter();
                        if let (Some(proc_macro::TokenTree::Ident(mut id)), None) = (iter.next(), iter.next()) {
                            id.set_span(s);
                            return Ident::Compiler(id);
                        }
                    }
                    panic!("not allowed as a raw identifier: `{}`", raw_prefixed)
                },
                Span::Fallback(s) => Ident::Fallback(fallback::Ident::new_raw(string, s)),
            }
        }

        pub fn span(&self) -> Span {
            match self {
                Ident::Compiler(t) => Span::Compiler(t.span()),
                Ident::Fallback(t) => Span::Fallback(t.span()),
            }
        }

        pub fn set_span(&mut self, span: Span) {
            match (self, span) {
                (Ident::Compiler(t), Span::Compiler(s)) => t.set_span(s),
                (Ident::Fallback(t), Span::Fallback(s)) => t.set_span(s),
                _ => mismatch(),
            }
        }

        fn unwrap_nightly(self) -> proc_macro::Ident {
            match self {
                Ident::Compiler(s) => s,
                Ident::Fallback(_) => mismatch(),
            }
        }
    }

    impl PartialEq for Ident {
        fn eq(&self, other: &Ident) -> bool {
            match (self, other) {
                (Ident::Compiler(t), Ident::Compiler(o)) => t.to_string() == o.to_string(),
                (Ident::Fallback(t), Ident::Fallback(o)) => t == o,
                _ => mismatch(),
            }
        }
    }

    impl<T> PartialEq<T> for Ident
    where
        T: ?Sized + AsRef<str>,
    {
        fn eq(&self, other: &T) -> bool {
            let other = other.as_ref();
            match self {
                Ident::Compiler(t) => t.to_string() == other,
                Ident::Fallback(t) => t == other,
            }
        }
    }

    impl Display for Ident {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Ident::Compiler(t) => Display::fmt(t, f),
                Ident::Fallback(t) => Display::fmt(t, f),
            }
        }
    }

    impl Debug for Ident {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Ident::Compiler(t) => Debug::fmt(t, f),
                Ident::Fallback(t) => Debug::fmt(t, f),
            }
        }
    }

    #[derive(Clone)]
    pub(crate) enum Literal {
        Compiler(proc_macro::Literal),
        Fallback(fallback::Literal),
    }

    macro_rules! suffixed_numbers {
    ($($name:ident => $kind:ident,)*) => ($(
        pub fn $name(n: $kind) -> Literal {
            if inside_proc_macro() {
                Literal::Compiler(proc_macro::Literal::$name(n))
            } else {
                Literal::Fallback(fallback::Literal::$name(n))
            }
        }
    )*)
}

    macro_rules! unsuffixed_integers {
    ($($name:ident => $kind:ident,)*) => ($(
        pub fn $name(n: $kind) -> Literal {
            if inside_proc_macro() {
                Literal::Compiler(proc_macro::Literal::$name(n))
            } else {
                Literal::Fallback(fallback::Literal::$name(n))
            }
        }
    )*)
}

    impl Literal {
        pub unsafe fn from_str_unchecked(repr: &str) -> Self {
            if inside_proc_macro() {
                Literal::Compiler(compiler_literal_from_str(repr).expect("invalid literal"))
            } else {
                Literal::Fallback(fallback::Literal::from_str_unchecked(repr))
            }
        }

        suffixed_numbers! {
            u8_suffixed => u8,
            u16_suffixed => u16,
            u32_suffixed => u32,
            u64_suffixed => u64,
            u128_suffixed => u128,
            usize_suffixed => usize,
            i8_suffixed => i8,
            i16_suffixed => i16,
            i32_suffixed => i32,
            i64_suffixed => i64,
            i128_suffixed => i128,
            isize_suffixed => isize,

            f32_suffixed => f32,
            f64_suffixed => f64,
        }

        unsuffixed_integers! {
            u8_unsuffixed => u8,
            u16_unsuffixed => u16,
            u32_unsuffixed => u32,
            u64_unsuffixed => u64,
            u128_unsuffixed => u128,
            usize_unsuffixed => usize,
            i8_unsuffixed => i8,
            i16_unsuffixed => i16,
            i32_unsuffixed => i32,
            i64_unsuffixed => i64,
            i128_unsuffixed => i128,
            isize_unsuffixed => isize,
        }

        pub fn f32_unsuffixed(f: f32) -> Literal {
            if inside_proc_macro() {
                Literal::Compiler(proc_macro::Literal::f32_unsuffixed(f))
            } else {
                Literal::Fallback(fallback::Literal::f32_unsuffixed(f))
            }
        }

        pub fn f64_unsuffixed(f: f64) -> Literal {
            if inside_proc_macro() {
                Literal::Compiler(proc_macro::Literal::f64_unsuffixed(f))
            } else {
                Literal::Fallback(fallback::Literal::f64_unsuffixed(f))
            }
        }

        pub fn string(t: &str) -> Literal {
            if inside_proc_macro() {
                Literal::Compiler(proc_macro::Literal::string(t))
            } else {
                Literal::Fallback(fallback::Literal::string(t))
            }
        }

        pub fn character(t: char) -> Literal {
            if inside_proc_macro() {
                Literal::Compiler(proc_macro::Literal::character(t))
            } else {
                Literal::Fallback(fallback::Literal::character(t))
            }
        }

        pub fn byte_string(bytes: &[u8]) -> Literal {
            if inside_proc_macro() {
                Literal::Compiler(proc_macro::Literal::byte_string(bytes))
            } else {
                Literal::Fallback(fallback::Literal::byte_string(bytes))
            }
        }

        pub fn span(&self) -> Span {
            match self {
                Literal::Compiler(lit) => Span::Compiler(lit.span()),
                Literal::Fallback(lit) => Span::Fallback(lit.span()),
            }
        }

        pub fn set_span(&mut self, span: Span) {
            match (self, span) {
                (Literal::Compiler(lit), Span::Compiler(s)) => lit.set_span(s),
                (Literal::Fallback(lit), Span::Fallback(s)) => lit.set_span(s),
                _ => mismatch(),
            }
        }

        pub fn subspan<R: RangeBounds<usize>>(&self, range: R) -> Option<Span> {
            match self {
                #[cfg(proc_macro_span)]
                Literal::Compiler(lit) => lit.subspan(range).map(Span::Compiler),
                #[cfg(not(proc_macro_span))]
                Literal::Compiler(_lit) => None,
                Literal::Fallback(lit) => lit.subspan(range).map(Span::Fallback),
            }
        }

        fn unwrap_nightly(self) -> proc_macro::Literal {
            match self {
                Literal::Compiler(s) => s,
                Literal::Fallback(_) => mismatch(),
            }
        }
    }

    impl From<fallback::Literal> for Literal {
        fn from(s: fallback::Literal) -> Self {
            Literal::Fallback(s)
        }
    }

    impl FromStr for Literal {
        type Err = LexError;

        fn from_str(repr: &str) -> Result<Self, Self::Err> {
            if inside_proc_macro() {
                compiler_literal_from_str(repr).map(Literal::Compiler)
            } else {
                let literal = fallback::Literal::from_str(repr)?;
                Ok(Literal::Fallback(literal))
            }
        }
    }

    fn compiler_literal_from_str(repr: &str) -> Result<proc_macro::Literal, LexError> {
        #[cfg(not(no_literal_from_str))]
        {
            proc_macro::Literal::from_str(repr).map_err(LexError::Compiler)
        }
        #[cfg(no_literal_from_str)]
        {
            let tokens = proc_macro_parse(repr)?;
            let mut iter = tokens.into_iter();
            if let (Some(proc_macro::TokenTree::Literal(literal)), None) = (iter.next(), iter.next()) {
                if literal.to_string().len() == repr.len() {
                    return Ok(literal);
                }
            }
            Err(LexError::call_site())
        }
    }

    impl Display for Literal {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Literal::Compiler(t) => Display::fmt(t, f),
                Literal::Fallback(t) => Display::fmt(t, f),
            }
        }
    }

    impl Debug for Literal {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Literal::Compiler(t) => Debug::fmt(t, f),
                Literal::Fallback(t) => Debug::fmt(t, f),
            }
        }
    }
}

#[cfg(span_locations)]
mod convert {
    pub(crate) fn usize_to_u32(u: usize) -> Option<u32> {
        #[cfg(not(no_try_from))]
        {
            use core::convert::TryFrom;

            u32::try_from(u).ok()
        }

        #[cfg(no_try_from)]
        {
            use core::mem;

            if mem::size_of::<usize>() <= mem::size_of::<u32>() || u <= u32::max_value() as usize {
                Some(u as u32)
            } else {
                None
            }
        }
    }
}
#[cfg(span_locations)]
mod location {
    use core::cmp::Ordering;

    /// A line-column pair representing the start or end of a `Span`.
    ///
    /// This type is semver exempt and not exposed by default.
    #[cfg_attr(doc_cfg, doc(cfg(feature = "span-locations")))]
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct LineColumn {
        /// The 1-indexed line in the source file on which the span starts or ends
        /// (inclusive).
        pub line: usize,
        /// The 0-indexed column (in UTF-8 characters) in the source file on which
        /// the span starts or ends (inclusive).
        pub column: usize,
    }

    impl Ord for LineColumn {
        fn cmp(&self, other: &Self) -> Ordering {
            self.line.cmp(&other.line).then(self.column.cmp(&other.column))
        }
    }

    impl PartialOrd for LineColumn {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
}

use crate::extra::DelimSpan;
use crate::marker::Marker;
use core::cmp::Ordering;
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::iter::FromIterator;
use core::ops::RangeBounds;
use core::str::FromStr;
use std::error::Error;
#[cfg(procmacro2_semver_exempt)]
use std::path::PathBuf;

#[cfg(span_locations)]
pub use crate::location::LineColumn;

/// An abstract stream of tokens, or more concretely a sequence of token trees.
///
/// This type provides interfaces for iterating over token trees and for
/// collecting token trees into one stream.
///
/// Token stream is both the input and output of `#[proc_macro]`,
/// `#[proc_macro_attribute]` and `#[proc_macro_derive]` definitions.
#[derive(Clone)]
pub struct TokenStream {
    inner: imp::TokenStream,
    _marker: Marker,
}

/// Error returned from `TokenStream::from_str`.
pub struct LexError {
    inner: imp::LexError,
    _marker: Marker,
}

impl TokenStream {
    fn _new(inner: imp::TokenStream) -> Self {
        TokenStream { inner, _marker: Marker }
    }

    fn _new_fallback(inner: fallback::TokenStream) -> Self {
        TokenStream {
            inner: inner.into(),
            _marker: Marker,
        }
    }

    /// Returns an empty `TokenStream` containing no token trees.
    pub fn new() -> Self {
        TokenStream::_new(imp::TokenStream::new())
    }

    /// Checks if this `TokenStream` is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// `TokenStream::default()` returns an empty stream,
/// i.e. this is equivalent with `TokenStream::new()`.
impl Default for TokenStream {
    fn default() -> Self {
        TokenStream::new()
    }
}

/// Attempts to break the string into tokens and parse those tokens into a token
/// stream.
///
/// May fail for a number of reasons, for example, if the string contains
/// unbalanced delimiters or characters not existing in the language.
///
/// NOTE: Some errors may cause panics instead of returning `LexError`. We
/// reserve the right to change these errors into `LexError`s later.
impl FromStr for TokenStream {
    type Err = LexError;

    fn from_str(src: &str) -> Result<TokenStream, LexError> {
        let e = src.parse().map_err(|e| LexError {
            inner: e,
            _marker: Marker,
        })?;
        Ok(TokenStream::_new(e))
    }
}

#[cfg(feature = "proc-macro")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "proc-macro")))]
impl From<proc_macro::TokenStream> for TokenStream {
    fn from(inner: proc_macro::TokenStream) -> Self {
        TokenStream::_new(inner.into())
    }
}

#[cfg(feature = "proc-macro")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "proc-macro")))]
impl From<TokenStream> for proc_macro::TokenStream {
    fn from(inner: TokenStream) -> Self {
        inner.inner.into()
    }
}

impl From<TokenTree> for TokenStream {
    fn from(token: TokenTree) -> Self {
        TokenStream::_new(imp::TokenStream::from(token))
    }
}

impl Extend<TokenTree> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenTree>>(&mut self, streams: I) {
        self.inner.extend(streams);
    }
}

impl Extend<TokenStream> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, streams: I) {
        self.inner.extend(streams.into_iter().map(|stream| stream.inner));
    }
}

/// Collects a number of token trees into a single stream.
impl FromIterator<TokenTree> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenTree>>(streams: I) -> Self {
        TokenStream::_new(streams.into_iter().collect())
    }
}
impl FromIterator<TokenStream> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenStream>>(streams: I) -> Self {
        TokenStream::_new(streams.into_iter().map(|i| i.inner).collect())
    }
}

/// Prints the token stream as a string that is supposed to be losslessly
/// convertible back into the same token stream (modulo spans), except for
/// possibly `TokenTree::Group`s with `Delimiter::None` delimiters and negative
/// numeric literals.
impl Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.inner, f)
    }
}

/// Prints token in a form convenient for debugging.
impl Debug for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}

impl LexError {
    pub fn span(&self) -> Span {
        Span::_new(self.inner.span())
    }
}

impl Debug for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}

impl Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.inner, f)
    }
}

impl Error for LexError {}

/// The source file of a given `Span`.
///
/// This type is semver exempt and not exposed by default.
#[cfg(all(procmacro2_semver_exempt, any(not(wrap_proc_macro), super_unstable)))]
#[cfg_attr(doc_cfg, doc(cfg(procmacro2_semver_exempt)))]
#[derive(Clone, PartialEq, Eq)]
pub struct SourceFile {
    inner: imp::SourceFile,
    _marker: Marker,
}

#[cfg(all(procmacro2_semver_exempt, any(not(wrap_proc_macro), super_unstable)))]
impl SourceFile {
    fn _new(inner: imp::SourceFile) -> Self {
        SourceFile { inner, _marker: Marker }
    }

    /// Get the path to this source file.
    ///
    /// ### Note
    ///
    /// If the code span associated with this `SourceFile` was generated by an
    /// external macro, this may not be an actual path on the filesystem. Use
    /// [`is_real`] to check.
    ///
    /// Also note that even if `is_real` returns `true`, if
    /// `--remap-path-prefix` was passed on the command line, the path as given
    /// may not actually be valid.
    ///
    /// [`is_real`]: #method.is_real
    pub fn path(&self) -> PathBuf {
        self.inner.path()
    }

    /// Returns `true` if this source file is a real source file, and not
    /// generated by an external macro's expansion.
    pub fn is_real(&self) -> bool {
        self.inner.is_real()
    }
}

#[cfg(all(procmacro2_semver_exempt, any(not(wrap_proc_macro), super_unstable)))]
impl Debug for SourceFile {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}

/// A region of source code, along with macro expansion information.
#[derive(Copy, Clone)]
pub struct Span {
    inner: imp::Span,
    _marker: Marker,
}

impl Span {
    fn _new(inner: imp::Span) -> Self {
        Span { inner, _marker: Marker }
    }

    fn _new_fallback(inner: fallback::Span) -> Self {
        Span {
            inner: inner.into(),
            _marker: Marker,
        }
    }

    /// The span of the invocation of the current procedural macro.
    ///
    /// Identifiers created with this span will be resolved as if they were
    /// written directly at the macro call location (call-site hygiene) and
    /// other code at the macro call site will be able to refer to them as well.
    pub fn call_site() -> Self {
        Span::_new(imp::Span::call_site())
    }

    /// The span located at the invocation of the procedural macro, but with
    /// local variables, labels, and `$crate` resolved at the definition site
    /// of the macro. This is the same hygiene behavior as `macro_rules`.
    ///
    /// This function requires Rust 1.45 or later.
    #[cfg(not(no_hygiene))]
    pub fn mixed_site() -> Self {
        Span::_new(imp::Span::mixed_site())
    }

    /// A span that resolves at the macro definition site.
    ///
    /// This method is semver exempt and not exposed by default.
    #[cfg(procmacro2_semver_exempt)]
    #[cfg_attr(doc_cfg, doc(cfg(procmacro2_semver_exempt)))]
    pub fn def_site() -> Self {
        Span::_new(imp::Span::def_site())
    }

    /// Creates a new span with the same line/column information as `self` but
    /// that resolves symbols as though it were at `other`.
    pub fn resolved_at(&self, other: Span) -> Span {
        Span::_new(self.inner.resolved_at(other.inner))
    }

    /// Creates a new span with the same name resolution behavior as `self` but
    /// with the line/column information of `other`.
    pub fn located_at(&self, other: Span) -> Span {
        Span::_new(self.inner.located_at(other.inner))
    }

    /// Convert `proc_macro2::Span` to `proc_macro::Span`.
    ///
    /// This method is available when building with a nightly compiler, or when
    /// building with rustc 1.29+ *without* semver exempt features.
    ///
    /// # Panics
    ///
    /// Panics if called from outside of a procedural macro. Unlike
    /// `proc_macro2::Span`, the `proc_macro::Span` type can only exist within
    /// the context of a procedural macro invocation.
    #[cfg(wrap_proc_macro)]
    pub fn unwrap(self) -> proc_macro::Span {
        self.inner.unwrap()
    }

    // Soft deprecated. Please use Span::unwrap.
    #[cfg(wrap_proc_macro)]
    #[doc(hidden)]
    pub fn unstable(self) -> proc_macro::Span {
        self.unwrap()
    }

    /// The original source file into which this span points.
    ///
    /// This method is semver exempt and not exposed by default.
    #[cfg(all(procmacro2_semver_exempt, any(not(wrap_proc_macro), super_unstable)))]
    #[cfg_attr(doc_cfg, doc(cfg(procmacro2_semver_exempt)))]
    pub fn source_file(&self) -> SourceFile {
        SourceFile::_new(self.inner.source_file())
    }

    /// Get the starting line/column in the source file for this span.
    ///
    /// This method requires the `"span-locations"` feature to be enabled.
    ///
    /// When executing in a procedural macro context, the returned line/column
    /// are only meaningful if compiled with a nightly toolchain. The stable
    /// toolchain does not have this information available. When executing
    /// outside of a procedural macro, such as main.rs or build.rs, the
    /// line/column are always meaningful regardless of toolchain.
    #[cfg(span_locations)]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "span-locations")))]
    pub fn start(&self) -> LineColumn {
        self.inner.start()
    }

    /// Get the ending line/column in the source file for this span.
    ///
    /// This method requires the `"span-locations"` feature to be enabled.
    ///
    /// When executing in a procedural macro context, the returned line/column
    /// are only meaningful if compiled with a nightly toolchain. The stable
    /// toolchain does not have this information available. When executing
    /// outside of a procedural macro, such as main.rs or build.rs, the
    /// line/column are always meaningful regardless of toolchain.
    #[cfg(span_locations)]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "span-locations")))]
    pub fn end(&self) -> LineColumn {
        self.inner.end()
    }

    /// Create a new span encompassing `self` and `other`.
    ///
    /// Returns `None` if `self` and `other` are from different files.
    ///
    /// Warning: the underlying [`proc_macro::Span::join`] method is
    /// nightly-only. When called from within a procedural macro not using a
    /// nightly compiler, this method will always return `None`.
    ///
    /// [`proc_macro::Span::join`]: https://doc.rust-lang.org/proc_macro/struct.Span.html#method.join
    pub fn join(&self, other: Span) -> Option<Span> {
        self.inner.join(other.inner).map(Span::_new)
    }

    /// Compares two spans to see if they're equal.
    ///
    /// This method is semver exempt and not exposed by default.
    #[cfg(procmacro2_semver_exempt)]
    #[cfg_attr(doc_cfg, doc(cfg(procmacro2_semver_exempt)))]
    pub fn eq(&self, other: &Span) -> bool {
        self.inner.eq(&other.inner)
    }

    /// Returns the source text behind a span. This preserves the original
    /// source code, including spaces and comments. It only returns a result if
    /// the span corresponds to real source code.
    ///
    /// Note: The observable result of a macro should only rely on the tokens
    /// and not on this source text. The result of this function is a best
    /// effort to be used for diagnostics only.
    pub fn source_text(&self) -> Option<String> {
        self.inner.source_text()
    }
}

/// Prints a span in a form convenient for debugging.
impl Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}

/// A single token or a delimited sequence of token trees (e.g. `[1, (), ..]`).
#[derive(Clone)]
pub enum TokenTree {
    /// A token stream surrounded by bracket delimiters.
    Group(Group),
    /// An identifier.
    Ident(Ident),
    /// A single punctuation character (`+`, `,`, `$`, etc.).
    Punct(Punct),
    /// A literal character (`'a'`), string (`"hello"`), number (`2.3`), etc.
    Literal(Literal),
}

impl TokenTree {
    /// Returns the span of this tree, delegating to the `span` method of
    /// the contained token or a delimited stream.
    pub fn span(&self) -> Span {
        match self {
            TokenTree::Group(t) => t.span(),
            TokenTree::Ident(t) => t.span(),
            TokenTree::Punct(t) => t.span(),
            TokenTree::Literal(t) => t.span(),
        }
    }

    /// Configures the span for *only this token*.
    ///
    /// Note that if this token is a `Group` then this method will not configure
    /// the span of each of the internal tokens, this will simply delegate to
    /// the `set_span` method of each variant.
    pub fn set_span(&mut self, span: Span) {
        match self {
            TokenTree::Group(t) => t.set_span(span),
            TokenTree::Ident(t) => t.set_span(span),
            TokenTree::Punct(t) => t.set_span(span),
            TokenTree::Literal(t) => t.set_span(span),
        }
    }
}

impl From<Group> for TokenTree {
    fn from(g: Group) -> Self {
        TokenTree::Group(g)
    }
}

impl From<Ident> for TokenTree {
    fn from(g: Ident) -> Self {
        TokenTree::Ident(g)
    }
}

impl From<Punct> for TokenTree {
    fn from(g: Punct) -> Self {
        TokenTree::Punct(g)
    }
}

impl From<Literal> for TokenTree {
    fn from(g: Literal) -> Self {
        TokenTree::Literal(g)
    }
}

/// Prints the token tree as a string that is supposed to be losslessly
/// convertible back into the same token tree (modulo spans), except for
/// possibly `TokenTree::Group`s with `Delimiter::None` delimiters and negative
/// numeric literals.
impl Display for TokenTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TokenTree::Group(t) => Display::fmt(t, f),
            TokenTree::Ident(t) => Display::fmt(t, f),
            TokenTree::Punct(t) => Display::fmt(t, f),
            TokenTree::Literal(t) => Display::fmt(t, f),
        }
    }
}

/// Prints token tree in a form convenient for debugging.
impl Debug for TokenTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Each of these has the name in the struct type in the derived debug,
        // so don't bother with an extra layer of indirection
        match self {
            TokenTree::Group(t) => Debug::fmt(t, f),
            TokenTree::Ident(t) => {
                let mut debug = f.debug_struct("Ident");
                debug.field("sym", &format_args!("{}", t));
                imp::debug_span_field_if_nontrivial(&mut debug, t.span().inner);
                debug.finish()
            },
            TokenTree::Punct(t) => Debug::fmt(t, f),
            TokenTree::Literal(t) => Debug::fmt(t, f),
        }
    }
}

/// A delimited token stream.
///
/// A `Group` internally contains a `TokenStream` which is surrounded by
/// `Delimiter`s.
#[derive(Clone)]
pub struct Group {
    inner: imp::Group,
}

/// Describes how a sequence of token trees is delimited.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Delimiter {
    /// `( ... )`
    Parenthesis,
    /// `{ ... }`
    Brace,
    /// `[ ... ]`
    Bracket,
    /// `Ø ... Ø`
    ///
    /// An implicit delimiter, that may, for example, appear around tokens
    /// coming from a "macro variable" `$var`. It is important to preserve
    /// operator priorities in cases like `$var * 3` where `$var` is `1 + 2`.
    /// Implicit delimiters may not survive roundtrip of a token stream through
    /// a string.
    None,
}

impl Group {
    fn _new(inner: imp::Group) -> Self {
        Group { inner }
    }

    fn _new_fallback(inner: fallback::Group) -> Self {
        Group { inner: inner.into() }
    }

    /// Creates a new `Group` with the given delimiter and token stream.
    ///
    /// This constructor will set the span for this group to
    /// `Span::call_site()`. To change the span you can use the `set_span`
    /// method below.
    pub fn new(delimiter: Delimiter, stream: TokenStream) -> Self {
        Group {
            inner: imp::Group::new(delimiter, stream.inner),
        }
    }

    /// Returns the punctuation used as the delimiter for this group: a set of
    /// parentheses, square brackets, or curly braces.
    pub fn delimiter(&self) -> Delimiter {
        self.inner.delimiter()
    }

    /// Returns the `TokenStream` of tokens that are delimited in this `Group`.
    ///
    /// Note that the returned token stream does not include the delimiter
    /// returned above.
    pub fn stream(&self) -> TokenStream {
        TokenStream::_new(self.inner.stream())
    }

    /// Returns the span for the delimiters of this token stream, spanning the
    /// entire `Group`.
    ///
    /// ```text
    /// pub fn span(&self) -> Span {
    ///            ^^^^^^^
    /// ```
    pub fn span(&self) -> Span {
        Span::_new(self.inner.span())
    }

    /// Returns the span pointing to the opening delimiter of this group.
    ///
    /// ```text
    /// pub fn span_open(&self) -> Span {
    ///                 ^
    /// ```
    pub fn span_open(&self) -> Span {
        Span::_new(self.inner.span_open())
    }

    /// Returns the span pointing to the closing delimiter of this group.
    ///
    /// ```text
    /// pub fn span_close(&self) -> Span {
    ///                        ^
    /// ```
    pub fn span_close(&self) -> Span {
        Span::_new(self.inner.span_close())
    }

    /// Returns an object that holds this group's `span_open()` and
    /// `span_close()` together (in a more compact representation than holding
    /// those 2 spans individually).
    pub fn delim_span(&self) -> DelimSpan {
        DelimSpan::new(&self.inner)
    }

    /// Configures the span for this `Group`'s delimiters, but not its internal
    /// tokens.
    ///
    /// This method will **not** set the span of all the internal tokens spanned
    /// by this group, but rather it will only set the span of the delimiter
    /// tokens at the level of the `Group`.
    pub fn set_span(&mut self, span: Span) {
        self.inner.set_span(span.inner);
    }
}

/// Prints the group as a string that should be losslessly convertible back
/// into the same group (modulo spans), except for possibly `TokenTree::Group`s
/// with `Delimiter::None` delimiters.
impl Display for Group {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.inner, formatter)
    }
}

impl Debug for Group {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.inner, formatter)
    }
}

/// A `Punct` is a single punctuation character like `+`, `-` or `#`.
///
/// Multicharacter operators like `+=` are represented as two instances of
/// `Punct` with different forms of `Spacing` returned.
#[derive(Clone)]
pub struct Punct {
    ch: char,
    spacing: Spacing,
    span: Span,
}

/// Whether a `Punct` is followed immediately by another `Punct` or followed by
/// another token or whitespace.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Spacing {
    /// E.g. `+` is `Alone` in `+ =`, `+ident` or `+()`.
    Alone,
    /// E.g. `+` is `Joint` in `+=` or `'` is `Joint` in `'#`.
    ///
    /// Additionally, single quote `'` can join with identifiers to form
    /// lifetimes `'ident`.
    Joint,
}

impl Punct {
    /// Creates a new `Punct` from the given character and spacing.
    ///
    /// The `ch` argument must be a valid punctuation character permitted by the
    /// language, otherwise the function will panic.
    ///
    /// The returned `Punct` will have the default span of `Span::call_site()`
    /// which can be further configured with the `set_span` method below.
    pub fn new(ch: char, spacing: Spacing) -> Self {
        Punct {
            ch,
            spacing,
            span: Span::call_site(),
        }
    }

    /// Returns the value of this punctuation character as `char`.
    pub fn as_char(&self) -> char {
        self.ch
    }

    /// Returns the spacing of this punctuation character, indicating whether
    /// it's immediately followed by another `Punct` in the token stream, so
    /// they can potentially be combined into a multicharacter operator
    /// (`Joint`), or it's followed by some other token or whitespace (`Alone`)
    /// so the operator has certainly ended.
    pub fn spacing(&self) -> Spacing {
        self.spacing
    }

    /// Returns the span for this punctuation character.
    pub fn span(&self) -> Span {
        self.span
    }

    /// Configure the span for this punctuation character.
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

/// Prints the punctuation character as a string that should be losslessly
/// convertible back into the same character.
impl Display for Punct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.ch, f)
    }
}

impl Debug for Punct {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut debug = fmt.debug_struct("Punct");
        debug.field("char", &self.ch);
        debug.field("spacing", &self.spacing);
        imp::debug_span_field_if_nontrivial(&mut debug, self.span.inner);
        debug.finish()
    }
}

/// A word of Rust code, which may be a keyword or legal variable name.
///
/// An identifier consists of at least one Unicode code point, the first of
/// which has the XID_Start property and the rest of which have the XID_Continue
/// property.
///
/// - The empty string is not an identifier. Use `Option<Ident>`.
/// - A lifetime is not an identifier. Use `syn::Lifetime` instead.
///
/// An identifier constructed with `Ident::new` is permitted to be a Rust
/// keyword, though parsing one through its [`Parse`] implementation rejects
/// Rust keywords. Use `input.call(Ident::parse_any)` when parsing to match the
/// behaviour of `Ident::new`.
///
/// [`Parse`]: https://docs.rs/syn/1.0/syn/parse/trait.Parse.html
///
/// # Examples
///
/// A new ident can be created from a string using the `Ident::new` function.
/// A span must be provided explicitly which governs the name resolution
/// behavior of the resulting identifier.
///
/// ```
/// use proc_macro2::{Ident, Span};
///
/// fn main() {
///     let call_ident = Ident::new("calligraphy", Span::call_site());
///
///     println!("{}", call_ident);
/// }
/// ```
///
/// An ident can be interpolated into a token stream using the `quote!` macro.
///
/// ```
/// use proc_macro2::{Ident, Span};
/// use quote::quote;
///
/// fn main() {
///     let ident = Ident::new("demo", Span::call_site());
///
///     // Create a variable binding whose name is this ident.
///     let expanded = quote! { let #ident = 10; };
///
///     // Create a variable binding with a slightly different name.
///     let temp_ident = Ident::new(&format!("new_{}", ident), Span::call_site());
///     let expanded = quote! { let #temp_ident = 10; };
/// }
/// ```
///
/// A string representation of the ident is available through the `to_string()`
/// method.
///
/// ```
/// # use proc_macro2::{Ident, Span};
/// #
/// # let ident = Ident::new("another_identifier", Span::call_site());
/// #
/// // Examine the ident as a string.
/// let ident_string = ident.to_string();
/// if ident_string.len() > 60 {
///     println!("Very long identifier: {}", ident_string)
/// }
/// ```
#[derive(Clone)]
pub struct Ident {
    inner: imp::Ident,
    _marker: Marker,
}

impl Ident {
    fn _new(inner: imp::Ident) -> Self {
        Ident { inner, _marker: Marker }
    }

    /// Creates a new `Ident` with the given `string` as well as the specified
    /// `span`.
    ///
    /// The `string` argument must be a valid identifier permitted by the
    /// language, otherwise the function will panic.
    ///
    /// Note that `span`, currently in rustc, configures the hygiene information
    /// for this identifier.
    ///
    /// As of this time `Span::call_site()` explicitly opts-in to "call-site"
    /// hygiene meaning that identifiers created with this span will be resolved
    /// as if they were written directly at the location of the macro call, and
    /// other code at the macro call site will be able to refer to them as well.
    ///
    /// Later spans like `Span::def_site()` will allow to opt-in to
    /// "definition-site" hygiene meaning that identifiers created with this
    /// span will be resolved at the location of the macro definition and other
    /// code at the macro call site will not be able to refer to them.
    ///
    /// Due to the current importance of hygiene this constructor, unlike other
    /// tokens, requires a `Span` to be specified at construction.
    ///
    /// # Panics
    ///
    /// Panics if the input string is neither a keyword nor a legal variable
    /// name. If you are not sure whether the string contains an identifier and
    /// need to handle an error case, use
    /// <a href="https://docs.rs/syn/1.0/syn/fn.parse_str.html"><code
    ///   style="padding-right:0;">syn::parse_str</code></a><code
    ///   style="padding-left:0;">::&lt;Ident&gt;</code>
    /// rather than `Ident::new`.
    pub fn new(string: &str, span: Span) -> Self {
        Ident::_new(imp::Ident::new(string, span.inner))
    }

    /// Same as `Ident::new`, but creates a raw identifier (`r#ident`). The
    /// `string` argument must be a valid identifier permitted by the language
    /// (including keywords, e.g. `fn`). Keywords which are usable in path
    /// segments (e.g. `self`, `super`) are not supported, and will cause a
    /// panic.
    pub fn new_raw(string: &str, span: Span) -> Self {
        Ident::_new_raw(string, span)
    }

    fn _new_raw(string: &str, span: Span) -> Self {
        Ident::_new(imp::Ident::new_raw(string, span.inner))
    }

    /// Returns the span of this `Ident`.
    pub fn span(&self) -> Span {
        Span::_new(self.inner.span())
    }

    /// Configures the span of this `Ident`, possibly changing its hygiene
    /// context.
    pub fn set_span(&mut self, span: Span) {
        self.inner.set_span(span.inner);
    }
}

impl PartialEq for Ident {
    fn eq(&self, other: &Ident) -> bool {
        self.inner == other.inner
    }
}

impl<T> PartialEq<T> for Ident
where
    T: ?Sized + AsRef<str>,
{
    fn eq(&self, other: &T) -> bool {
        self.inner == other
    }
}

impl Eq for Ident {}

impl PartialOrd for Ident {
    fn partial_cmp(&self, other: &Ident) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ident {
    fn cmp(&self, other: &Ident) -> Ordering {
        self.to_string().cmp(&other.to_string())
    }
}

impl Hash for Ident {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.to_string().hash(hasher);
    }
}

/// Prints the identifier as a string that should be losslessly convertible back
/// into the same identifier.
impl Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.inner, f)
    }
}

impl Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}

/// A literal string (`"hello"`), byte string (`b"hello"`), character (`'a'`),
/// byte character (`b'a'`), an integer or floating point number with or without
/// a suffix (`1`, `1u8`, `2.3`, `2.3f32`).
///
/// Boolean literals like `true` and `false` do not belong here, they are
/// `Ident`s.
#[derive(Clone)]
pub struct Literal {
    inner: imp::Literal,
    _marker: Marker,
}

macro_rules! suffixed_int_literals {
    ($($name:ident => $kind:ident,)*) => ($(
        /// Creates a new suffixed integer literal with the specified value.
        ///
        /// This function will create an integer like `1u32` where the integer
        /// value specified is the first part of the token and the integral is
        /// also suffixed at the end. Literals created from negative numbers may
        /// not survive roundtrips through `TokenStream` or strings and may be
        /// broken into two tokens (`-` and positive literal).
        ///
        /// Literals created through this method have the `Span::call_site()`
        /// span by default, which can be configured with the `set_span` method
        /// below.
        pub fn $name(n: $kind) -> Literal {
            Literal::_new(imp::Literal::$name(n))
        }
    )*)
}

macro_rules! unsuffixed_int_literals {
    ($($name:ident => $kind:ident,)*) => ($(
        /// Creates a new unsuffixed integer literal with the specified value.
        ///
        /// This function will create an integer like `1` where the integer
        /// value specified is the first part of the token. No suffix is
        /// specified on this token, meaning that invocations like
        /// `Literal::i8_unsuffixed(1)` are equivalent to
        /// `Literal::u32_unsuffixed(1)`. Literals created from negative numbers
        /// may not survive roundtrips through `TokenStream` or strings and may
        /// be broken into two tokens (`-` and positive literal).
        ///
        /// Literals created through this method have the `Span::call_site()`
        /// span by default, which can be configured with the `set_span` method
        /// below.
        pub fn $name(n: $kind) -> Literal {
            Literal::_new(imp::Literal::$name(n))
        }
    )*)
}

impl Literal {
    fn _new(inner: imp::Literal) -> Self {
        Literal { inner, _marker: Marker }
    }

    fn _new_fallback(inner: fallback::Literal) -> Self {
        Literal {
            inner: inner.into(),
            _marker: Marker,
        }
    }

    suffixed_int_literals! {
        u8_suffixed => u8,
        u16_suffixed => u16,
        u32_suffixed => u32,
        u64_suffixed => u64,
        u128_suffixed => u128,
        usize_suffixed => usize,
        i8_suffixed => i8,
        i16_suffixed => i16,
        i32_suffixed => i32,
        i64_suffixed => i64,
        i128_suffixed => i128,
        isize_suffixed => isize,
    }

    unsuffixed_int_literals! {
        u8_unsuffixed => u8,
        u16_unsuffixed => u16,
        u32_unsuffixed => u32,
        u64_unsuffixed => u64,
        u128_unsuffixed => u128,
        usize_unsuffixed => usize,
        i8_unsuffixed => i8,
        i16_unsuffixed => i16,
        i32_unsuffixed => i32,
        i64_unsuffixed => i64,
        i128_unsuffixed => i128,
        isize_unsuffixed => isize,
    }

    /// Creates a new unsuffixed floating-point literal.
    ///
    /// This constructor is similar to those like `Literal::i8_unsuffixed` where
    /// the float's value is emitted directly into the token but no suffix is
    /// used, so it may be inferred to be a `f64` later in the compiler.
    /// Literals created from negative numbers may not survive round-trips
    /// through `TokenStream` or strings and may be broken into two tokens (`-`
    /// and positive literal).
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for example
    /// if it is infinity or NaN this function will panic.
    pub fn f64_unsuffixed(f: f64) -> Literal {
        assert!(f.is_finite());
        Literal::_new(imp::Literal::f64_unsuffixed(f))
    }

    /// Creates a new suffixed floating-point literal.
    ///
    /// This constructor will create a literal like `1.0f64` where the value
    /// specified is the preceding part of the token and `f64` is the suffix of
    /// the token. This token will always be inferred to be an `f64` in the
    /// compiler. Literals created from negative numbers may not survive
    /// round-trips through `TokenStream` or strings and may be broken into two
    /// tokens (`-` and positive literal).
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for example
    /// if it is infinity or NaN this function will panic.
    pub fn f64_suffixed(f: f64) -> Literal {
        assert!(f.is_finite());
        Literal::_new(imp::Literal::f64_suffixed(f))
    }

    /// Creates a new unsuffixed floating-point literal.
    ///
    /// This constructor is similar to those like `Literal::i8_unsuffixed` where
    /// the float's value is emitted directly into the token but no suffix is
    /// used, so it may be inferred to be a `f64` later in the compiler.
    /// Literals created from negative numbers may not survive round-trips
    /// through `TokenStream` or strings and may be broken into two tokens (`-`
    /// and positive literal).
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for example
    /// if it is infinity or NaN this function will panic.
    pub fn f32_unsuffixed(f: f32) -> Literal {
        assert!(f.is_finite());
        Literal::_new(imp::Literal::f32_unsuffixed(f))
    }

    /// Creates a new suffixed floating-point literal.
    ///
    /// This constructor will create a literal like `1.0f32` where the value
    /// specified is the preceding part of the token and `f32` is the suffix of
    /// the token. This token will always be inferred to be an `f32` in the
    /// compiler. Literals created from negative numbers may not survive
    /// round-trips through `TokenStream` or strings and may be broken into two
    /// tokens (`-` and positive literal).
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for example
    /// if it is infinity or NaN this function will panic.
    pub fn f32_suffixed(f: f32) -> Literal {
        assert!(f.is_finite());
        Literal::_new(imp::Literal::f32_suffixed(f))
    }

    /// String literal.
    pub fn string(string: &str) -> Literal {
        Literal::_new(imp::Literal::string(string))
    }

    /// Character literal.
    pub fn character(ch: char) -> Literal {
        Literal::_new(imp::Literal::character(ch))
    }

    /// Byte string literal.
    pub fn byte_string(s: &[u8]) -> Literal {
        Literal::_new(imp::Literal::byte_string(s))
    }

    /// Returns the span encompassing this literal.
    pub fn span(&self) -> Span {
        Span::_new(self.inner.span())
    }

    /// Configures the span associated for this literal.
    pub fn set_span(&mut self, span: Span) {
        self.inner.set_span(span.inner);
    }

    /// Returns a `Span` that is a subset of `self.span()` containing only
    /// the source bytes in range `range`. Returns `None` if the would-be
    /// trimmed span is outside the bounds of `self`.
    ///
    /// Warning: the underlying [`proc_macro::Literal::subspan`] method is
    /// nightly-only. When called from within a procedural macro not using a
    /// nightly compiler, this method will always return `None`.
    ///
    /// [`proc_macro::Literal::subspan`]: https://doc.rust-lang.org/proc_macro/struct.Literal.html#method.subspan
    pub fn subspan<R: RangeBounds<usize>>(&self, range: R) -> Option<Span> {
        self.inner.subspan(range).map(Span::_new)
    }

    // Intended for the `quote!` macro to use when constructing a proc-macro2
    // token out of a macro_rules $:literal token, which is already known to be
    // a valid literal. This avoids reparsing/validating the literal's string
    // representation. This is not public API other than for quote.
    #[doc(hidden)]
    pub unsafe fn from_str_unchecked(repr: &str) -> Self {
        Literal::_new(imp::Literal::from_str_unchecked(repr))
    }
}

impl FromStr for Literal {
    type Err = LexError;

    fn from_str(repr: &str) -> Result<Self, LexError> {
        repr.parse()
            .map(Literal::_new)
            .map_err(|inner| LexError { inner, _marker: Marker })
    }
}

impl Debug for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.inner, f)
    }
}

/// Public implementation details for the `TokenStream` type, such as iterators.
pub mod token_stream {
    use crate::marker::Marker;
    use crate::{imp, TokenTree};
    use core::fmt::{self, Debug};

    pub use crate::TokenStream;

    /// An iterator over `TokenStream`'s `TokenTree`s.
    ///
    /// The iteration is "shallow", e.g. the iterator doesn't recurse into
    /// delimited groups, and returns whole groups as token trees.
    #[derive(Clone)]
    pub struct IntoIter {
        inner: imp::TokenTreeIter,
        _marker: Marker,
    }

    impl Iterator for IntoIter {
        type Item = TokenTree;

        fn next(&mut self) -> Option<TokenTree> {
            self.inner.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }

    impl Debug for IntoIter {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("TokenStream ")?;
            f.debug_list().entries(self.clone()).finish()
        }
    }

    impl IntoIterator for TokenStream {
        type Item = TokenTree;
        type IntoIter = IntoIter;

        fn into_iter(self) -> IntoIter {
            IntoIter {
                inner: self.inner.into_iter(),
                _marker: Marker,
            }
        }
    }
}

pub trait IntoSpans<S> {
    fn into_spans(self) -> S;
}
impl IntoSpans<Span> for Span {
    fn into_spans(self) -> Span {
        self
    }
}
impl IntoSpans<[Span; 1]> for Span {
    fn into_spans(self) -> [Span; 1] {
        [self]
    }
}
impl IntoSpans<[Span; 2]> for Span {
    fn into_spans(self) -> [Span; 2] {
        [self, self]
    }
}
impl IntoSpans<[Span; 3]> for Span {
    fn into_spans(self) -> [Span; 3] {
        [self, self, self]
    }
}
impl IntoSpans<[Span; 1]> for [Span; 1] {
    fn into_spans(self) -> [Span; 1] {
        self
    }
}
impl IntoSpans<[Span; 2]> for [Span; 2] {
    fn into_spans(self) -> [Span; 2] {
        self
    }
}
impl IntoSpans<[Span; 3]> for [Span; 3] {
    fn into_spans(self) -> [Span; 3] {
        self
    }
}
impl IntoSpans<DelimSpan> for Span {
    fn into_spans(self) -> DelimSpan {
        let mut y = Group::new(Delim::None, Stream::new());
        y.set_span(self);
        y.delim_span()
    }
}
impl IntoSpans<DelimSpan> for DelimSpan {
    fn into_spans(self) -> DelimSpan {
        self
    }
}

pub use self::{
    extra::DelimSpan, Delimiter as Delim, Group, Ident, LexError, Literal as Lit, Punct, Spacing, Span,
    TokenStream as Stream, TokenTree as Tree,
};
