use crate::ext::IdentExt;
use crate::lit::Lit;
use crate::parse::{Error, ParseStream, Parser, Result};
use crate::path::{Path, PathSegment};
use crate::punctuated::Punctuated;
use proc_macro2::Ident;
use std::fmt::Display;
pub fn parser(logic: impl FnMut(ParseNestedMeta) -> Result<()>) -> impl Parser<Output = ()> {
    |input: ParseStream| {
        if input.is_empty() {
            Ok(())
        } else {
            parse_nested_meta(input, logic)
        }
    }
}
#[non_exhaustive]
pub struct ParseNestedMeta<'a> {
    pub path: Path,
    pub input: ParseStream<'a>,
}
impl<'a> ParseNestedMeta<'a> {
    pub fn value(&self) -> Result<ParseStream<'a>> {
        self.input.parse::<Token![=]>()?;
        Ok(self.input)
    }
    pub fn parse_nested_meta(&self, logic: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
        let content;
        parenthesized!(content in self.input);
        parse_nested_meta(&content, logic)
    }
    pub fn error(&self, msg: impl Display) -> Error {
        let start_span = self.path.segments[0].ident.span();
        let end_span = self.input.cursor().prev_span();
        crate::error::new2(start_span, end_span, msg)
    }
}
pub(crate) fn parse_nested_meta(
    input: ParseStream,
    mut logic: impl FnMut(ParseNestedMeta) -> Result<()>,
) -> Result<()> {
    loop {
        let path = input.call(parse_meta_path)?;
        logic(ParseNestedMeta { path, input })?;
        if input.is_empty() {
            return Ok(());
        }
        input.parse::<Token![,]>()?;
        if input.is_empty() {
            return Ok(());
        }
    }
}
fn parse_meta_path(input: ParseStream) -> Result<Path> {
    Ok(Path {
        leading_colon: input.parse()?,
        segments: {
            let mut segments = Punctuated::new();
            if input.peek(Ident::peek_any) {
                let ident = Ident::parse_any(input)?;
                segments.push_value(PathSegment::from(ident));
            } else if input.is_empty() {
                return Err(input.error("expected nested attribute"));
            } else if input.peek(Lit) {
                return Err(input.error("unexpected literal in nested attribute, expected ident"));
            } else {
                return Err(input.error("unexpected token in nested attribute, expected ident"));
            }
            while input.peek(Token![::]) {
                let punct = input.parse()?;
                segments.push_punct(punct);
                let ident = Ident::parse_any(input)?;
                segments.push_value(PathSegment::from(ident));
            }
            segments
        },
    })
}
