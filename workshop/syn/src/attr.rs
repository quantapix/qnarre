use super::{
    parse::{Parse, ParseStream, Parser, Result},
    *,
};
use crate::ext::IdentExt;
use crate::lit::Lit;
use crate::parse::{Err, ParseStream, Parser, Result};
use crate::path::{Path, PathSegment};
use crate::punctuated::Punctuated;
use proc_macro2::{Ident, TokenStream};
use std::{fmt::Display, iter, slice};

ast_struct! {
    pub struct Attribute {
        pub pound_token: Token![#],
        pub style: AttrStyle,
        pub bracket_token: token::Bracket,
        pub meta: Meta,
    }
}
impl Attribute {
    pub fn path(&self) -> &Path {
        self.meta.path()
    }
    pub fn parse_args<T: Parse>(&self) -> Result<T> {
        self.parse_args_with(T::parse)
    }
    pub fn parse_args_with<T: Parser>(&self, parser: T) -> Result<T::Output> {
        match &self.meta {
            Meta::Path(x) => Err(crate::err::new2(
                x.segments.first().unwrap().ident.span(),
                x.segments.last().unwrap().ident.span(),
                format!(
                    "expected attribute arguments in parentheses: {}[{}(...)]",
                    parsing::DisplayAttrStyle(&self.style),
                    parsing::DisplayPath(x),
                ),
            )),
            Meta::NameValue(x) => Err(Err::new(
                x.eq_token.span,
                format_args!(
                    "expected parentheses: {}[{}(...)]",
                    parsing::DisplayAttrStyle(&self.style),
                    parsing::DisplayPath(&meta.path),
                ),
            )),
            Meta::List(x) => x.parse_args_with(parser),
        }
    }
    pub fn parse_nested_meta(&self, x: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
        self.parse_args_with(meta_parser(x))
    }
    pub fn parse_outer(x: ParseStream) -> Result<Vec<Self>> {
        let mut y = Vec::new();
        while x.peek(Token![#]) {
            y.push(x.call(parsing::single_parse_outer)?);
        }
        Ok(y)
    }
    pub fn parse_inner(x: ParseStream) -> Result<Vec<Self>> {
        let mut y = Vec::new();
        parsing::parse_inner(x, &mut y)?;
        Ok(y)
    }
}
ast_enum! {
    pub enum AttrStyle {
        Outer,
        Inner(Token![!]),
    }
}
ast_enum_of_structs! {
    pub enum Meta {
        Path(Path),
        List(MetaList),
        NameValue(MetaNameValue),
    }
}
ast_struct! {
    pub struct MetaList {
        pub path: Path,
        pub delimiter: MacroDelimiter,
        pub tokens: TokenStream,
    }
}
ast_struct! {
    pub struct MetaNameValue {
        pub path: Path,
        pub eq_token: Token![=],
        pub value: Expr,
    }
}
impl Meta {
    pub fn path(&self) -> &Path {
        match self {
            Meta::Path(x) => x,
            Meta::List(x) => &x.path,
            Meta::NameValue(x) => &x.path,
        }
    }
    pub fn require_path_only(&self) -> Result<&Path> {
        let y = match self {
            Meta::Path(x) => return Ok(x),
            Meta::List(x) => x.delimiter.span().open(),
            Meta::NameValue(x) => x.eq_token.span,
        };
        Err(Err::new(y, "unexpected token in attribute"))
    }
    pub fn require_list(&self) -> Result<&MetaList> {
        match self {
            Meta::List(x) => Ok(x),
            Meta::Path(x) => Err(crate::err::new2(
                x.segments.first().unwrap().ident.span(),
                x.segments.last().unwrap().ident.span(),
                format!(
                    "expected attribute arguments in parentheses: `{}(...)`",
                    parsing::DisplayPath(x),
                ),
            )),
            Meta::NameValue(x) => Err(Err::new(x.eq_token.span, "expected `(`")),
        }
    }
    pub fn require_name_value(&self) -> Result<&MetaNameValue> {
        match self {
            Meta::NameValue(x) => Ok(x),
            Meta::Path(x) => Err(crate::err::new2(
                x.segments.first().unwrap().ident.span(),
                x.segments.last().unwrap().ident.span(),
                format!(
                    "expected a value for this attribute: `{} = ...`",
                    parsing::DisplayPath(x),
                ),
            )),
            Meta::List(x) => Err(Err::new(x.delimiter.span().open(), "expected `=`")),
        }
    }
}
impl MetaList {
    pub fn parse_args<T: Parse>(&self) -> Result<T> {
        self.parse_args_with(T::parse)
    }
    pub fn parse_args_with<T: Parser>(&self, parser: T) -> Result<T::Output> {
        let y = self.delimiter.span().close();
        crate::parse::parse_scoped(parser, y, self.tokens.clone())
    }
    pub fn parse_nested_meta(&self, x: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
        self.parse_args_with(meta_parser(x))
    }
}

pub(crate) trait FilterAttrs<'a> {
    type Ret: Iterator<Item = &'a Attribute>;
    fn outer(self) -> Self::Ret;
    fn inner(self) -> Self::Ret;
}
impl<'a> FilterAttrs<'a> for &'a [Attribute] {
    type Ret = iter::Filter<slice::Iter<'a, Attribute>, fn(&&Attribute) -> bool>;
    fn outer(self) -> Self::Ret {
        fn is_outer(x: &&Attribute) -> bool {
            match x.style {
                AttrStyle::Outer => true,
                AttrStyle::Inner(_) => false,
            }
        }
        self.iter().filter(is_outer)
    }
    fn inner(self) -> Self::Ret {
        fn is_inner(x: &&Attribute) -> bool {
            match x.style {
                AttrStyle::Inner(_) => true,
                AttrStyle::Outer => false,
            }
        }
        self.iter().filter(is_inner)
    }
}

pub fn meta_parser(logic: impl FnMut(ParseNestedMeta) -> Result<()>) -> impl Parser<Output = ()> {
    |input: ParseStream| {
        if input.is_empty() {
            Ok(())
        } else {
            parse_nested_meta(input, logic)
        }
    }
}

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
    pub fn error(&self, msg: impl Display) -> Err {
        let start_span = self.path.segments[0].ident.span();
        let end_span = self.input.cursor().prev_span();
        crate::err::new2(start_span, end_span, msg)
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
