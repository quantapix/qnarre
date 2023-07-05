use super::*;
use crate::meta::{self, ParseNestedMeta};
use crate::parse::{Parse, ParseStream, Parser, Result};
use proc_macro2::TokenStream;
use std::iter;
use std::slice;
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
    pub fn parse_args_with<F: Parser>(&self, parser: F) -> Result<F::Output> {
        match &self.meta {
            Meta::Path(path) => Err(crate::error::new2(
                path.segments.first().unwrap().ident.span(),
                path.segments.last().unwrap().ident.span(),
                format!(
                    "expected attribute arguments in parentheses: {}[{}(...)]",
                    parsing::DisplayAttrStyle(&self.style),
                    parsing::DisplayPath(path),
                ),
            )),
            Meta::NameValue(meta) => Err(Error::new(
                meta.eq_token.span,
                format_args!(
                    "expected parentheses: {}[{}(...)]",
                    parsing::DisplayAttrStyle(&self.style),
                    parsing::DisplayPath(&meta.path),
                ),
            )),
            Meta::List(meta) => meta.parse_args_with(parser),
        }
    }
    pub fn parse_nested_meta(&self, logic: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
        self.parse_args_with(meta::parser(logic))
    }
    pub fn parse_outer(input: ParseStream) -> Result<Vec<Self>> {
        let mut attrs = Vec::new();
        while input.peek(Token![#]) {
            attrs.push(input.call(parsing::single_parse_outer)?);
        }
        Ok(attrs)
    }
    pub fn parse_inner(input: ParseStream) -> Result<Vec<Self>> {
        let mut attrs = Vec::new();
        parsing::parse_inner(input, &mut attrs)?;
        Ok(attrs)
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
            Meta::Path(path) => path,
            Meta::List(meta) => &meta.path,
            Meta::NameValue(meta) => &meta.path,
        }
    }
    pub fn require_path_only(&self) -> Result<&Path> {
        let error_span = match self {
            Meta::Path(path) => return Ok(path),
            Meta::List(meta) => meta.delimiter.span().open(),
            Meta::NameValue(meta) => meta.eq_token.span,
        };
        Err(Error::new(error_span, "unexpected token in attribute"))
    }
    pub fn require_list(&self) -> Result<&MetaList> {
        match self {
            Meta::List(meta) => Ok(meta),
            Meta::Path(path) => Err(crate::error::new2(
                path.segments.first().unwrap().ident.span(),
                path.segments.last().unwrap().ident.span(),
                format!(
                    "expected attribute arguments in parentheses: `{}(...)`",
                    parsing::DisplayPath(path),
                ),
            )),
            Meta::NameValue(meta) => Err(Error::new(meta.eq_token.span, "expected `(`")),
        }
    }
    pub fn require_name_value(&self) -> Result<&MetaNameValue> {
        match self {
            Meta::NameValue(meta) => Ok(meta),
            Meta::Path(path) => Err(crate::error::new2(
                path.segments.first().unwrap().ident.span(),
                path.segments.last().unwrap().ident.span(),
                format!(
                    "expected a value for this attribute: `{} = ...`",
                    parsing::DisplayPath(path),
                ),
            )),
            Meta::List(meta) => Err(Error::new(meta.delimiter.span().open(), "expected `=`")),
        }
    }
}
impl MetaList {
    pub fn parse_args<T: Parse>(&self) -> Result<T> {
        self.parse_args_with(T::parse)
    }
    pub fn parse_args_with<F: Parser>(&self, parser: F) -> Result<F::Output> {
        let scope = self.delimiter.span().close();
        crate::parse::parse_scoped(parser, scope, self.tokens.clone())
    }
    pub fn parse_nested_meta(&self, logic: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
        self.parse_args_with(meta::parser(logic))
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
        fn is_outer(attr: &&Attribute) -> bool {
            match attr.style {
                AttrStyle::Outer => true,
                AttrStyle::Inner(_) => false,
            }
        }
        self.iter().filter(is_outer)
    }
    fn inner(self) -> Self::Ret {
        fn is_inner(attr: &&Attribute) -> bool {
            match attr.style {
                AttrStyle::Inner(_) => true,
                AttrStyle::Outer => false,
            }
        }
        self.iter().filter(is_inner)
    }
}
pub(crate) mod parsing {
    use super::*;
    use crate::parse::discouraged::Speculative;
    use crate::parse::{Parse, ParseStream, Result};
    use std::fmt::{self, Display};
    pub(crate) fn parse_inner(input: ParseStream, attrs: &mut Vec<Attribute>) -> Result<()> {
        while input.peek(Token![#]) && input.peek2(Token![!]) {
            attrs.push(input.call(parsing::single_parse_inner)?);
        }
        Ok(())
    }
    pub(crate) fn single_parse_inner(input: ParseStream) -> Result<Attribute> {
        let content;
        Ok(Attribute {
            pound_token: input.parse()?,
            style: AttrStyle::Inner(input.parse()?),
            bracket_token: bracketed!(content in input),
            meta: content.parse()?,
        })
    }
    pub(crate) fn single_parse_outer(input: ParseStream) -> Result<Attribute> {
        let content;
        Ok(Attribute {
            pound_token: input.parse()?,
            style: AttrStyle::Outer,
            bracket_token: bracketed!(content in input),
            meta: content.parse()?,
        })
    }
    impl Parse for Meta {
        fn parse(input: ParseStream) -> Result<Self> {
            let path = input.call(Path::parse_mod_style)?;
            parse_meta_after_path(path, input)
        }
    }
    impl Parse for MetaList {
        fn parse(input: ParseStream) -> Result<Self> {
            let path = input.call(Path::parse_mod_style)?;
            parse_meta_list_after_path(path, input)
        }
    }
    impl Parse for MetaNameValue {
        fn parse(input: ParseStream) -> Result<Self> {
            let path = input.call(Path::parse_mod_style)?;
            parse_meta_name_value_after_path(path, input)
        }
    }
    pub(crate) fn parse_meta_after_path(path: Path, input: ParseStream) -> Result<Meta> {
        if input.peek(token::Paren) || input.peek(token::Bracket) || input.peek(token::Brace) {
            parse_meta_list_after_path(path, input).map(Meta::List)
        } else if input.peek(Token![=]) {
            parse_meta_name_value_after_path(path, input).map(Meta::NameValue)
        } else {
            Ok(Meta::Path(path))
        }
    }
    fn parse_meta_list_after_path(path: Path, input: ParseStream) -> Result<MetaList> {
        let (delimiter, tokens) = mac::parse_delimiter(input)?;
        Ok(MetaList {
            path,
            delimiter,
            tokens,
        })
    }
    fn parse_meta_name_value_after_path(path: Path, input: ParseStream) -> Result<MetaNameValue> {
        let eq_token: Token![=] = input.parse()?;
        let ahead = input.fork();
        let lit: Option<Lit> = ahead.parse()?;
        let value = if let (Some(lit), true) = (lit, ahead.is_empty()) {
            input.advance_to(&ahead);
            Expr::Lit(ExprLit { attrs: Vec::new(), lit })
        } else if input.peek(Token![#]) && input.peek2(token::Bracket) {
            return Err(input.error("unexpected attribute inside of attribute"));
        } else {
            input.parse()?
        };
        Ok(MetaNameValue { path, eq_token, value })
    }
    pub(super) struct DisplayAttrStyle<'a>(pub &'a AttrStyle);
    impl<'a> Display for DisplayAttrStyle<'a> {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str(match self.0 {
                AttrStyle::Outer => "#",
                AttrStyle::Inner(_) => "#!",
            })
        }
    }
    pub(super) struct DisplayPath<'a>(pub &'a Path);
    impl<'a> Display for DisplayPath<'a> {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            for (i, segment) in self.0.segments.iter().enumerate() {
                if i > 0 || self.0.leading_colon.is_some() {
                    formatter.write_str("::")?;
                }
                write!(formatter, "{}", segment.ident)?;
            }
            Ok(())
        }
    }
}
mod printing {
    use super::*;
    use proc_macro2::TokenStream;
    use quote::ToTokens;
    impl ToTokens for Attribute {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.pound_token.to_tokens(tokens);
            if let AttrStyle::Inner(b) = &self.style {
                b.to_tokens(tokens);
            }
            self.bracket_token.surround(tokens, |tokens| {
                self.meta.to_tokens(tokens);
            });
        }
    }
    impl ToTokens for MetaList {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.path.to_tokens(tokens);
            self.delimiter.surround(tokens, self.tokens.clone());
        }
    }
    impl ToTokens for MetaNameValue {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.path.to_tokens(tokens);
            self.eq_token.to_tokens(tokens);
            self.value.to_tokens(tokens);
        }
    }
}
