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
    pub fn parse_args_with<T: Parser>(&self, parser: T) -> Result<T::Output> {
        match &self.meta {
            Meta::Path(x) => Err(crate::error::new2(
                x.segments.first().unwrap().ident.span(),
                x.segments.last().unwrap().ident.span(),
                format!(
                    "expected attribute arguments in parentheses: {}[{}(...)]",
                    parsing::DisplayAttrStyle(&self.style),
                    parsing::DisplayPath(x),
                ),
            )),
            Meta::NameValue(x) => Err(Error::new(
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
        self.parse_args_with(meta::parser(x))
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
        Err(Error::new(y, "unexpected token in attribute"))
    }
    pub fn require_list(&self) -> Result<&MetaList> {
        match self {
            Meta::List(x) => Ok(x),
            Meta::Path(x) => Err(crate::error::new2(
                x.segments.first().unwrap().ident.span(),
                x.segments.last().unwrap().ident.span(),
                format!(
                    "expected attribute arguments in parentheses: `{}(...)`",
                    parsing::DisplayPath(x),
                ),
            )),
            Meta::NameValue(x) => Err(Error::new(x.eq_token.span, "expected `(`")),
        }
    }
    pub fn require_name_value(&self) -> Result<&MetaNameValue> {
        match self {
            Meta::NameValue(x) => Ok(x),
            Meta::Path(x) => Err(crate::error::new2(
                x.segments.first().unwrap().ident.span(),
                x.segments.last().unwrap().ident.span(),
                format!(
                    "expected a value for this attribute: `{} = ...`",
                    parsing::DisplayPath(x),
                ),
            )),
            Meta::List(x) => Err(Error::new(x.delimiter.span().open(), "expected `=`")),
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
        self.parse_args_with(meta::parser(x))
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
pub(crate) mod parsing {
    use super::{
        parse::{discouraged::Speculative, Parse, ParseStream, Result},
        *,
    };
    use std::fmt::{self, Display};
    pub(crate) fn parse_inner(x: ParseStream, ys: &mut Vec<Attribute>) -> Result<()> {
        while x.peek(Token![#]) && x.peek2(Token![!]) {
            ys.push(x.call(parsing::single_parse_inner)?);
        }
        Ok(())
    }
    pub(crate) fn single_parse_inner(x: ParseStream) -> Result<Attribute> {
        let content;
        Ok(Attribute {
            pound_token: x.parse()?,
            style: AttrStyle::Inner(x.parse()?),
            bracket_token: bracketed!(content in x),
            meta: content.parse()?,
        })
    }
    pub(crate) fn single_parse_outer(x: ParseStream) -> Result<Attribute> {
        let content;
        Ok(Attribute {
            pound_token: x.parse()?,
            style: AttrStyle::Outer,
            bracket_token: bracketed!(content in x),
            meta: content.parse()?,
        })
    }
    impl Parse for Meta {
        fn parse(x: ParseStream) -> Result<Self> {
            let path = x.call(Path::parse_mod_style)?;
            parse_meta_after_path(path, x)
        }
    }
    impl Parse for MetaList {
        fn parse(x: ParseStream) -> Result<Self> {
            let path = x.call(Path::parse_mod_style)?;
            parse_meta_list_after_path(path, x)
        }
    }
    impl Parse for MetaNameValue {
        fn parse(x: ParseStream) -> Result<Self> {
            let path = x.call(Path::parse_mod_style)?;
            parse_meta_name_value_after_path(path, x)
        }
    }
    pub(crate) fn parse_meta_after_path(path: Path, x: ParseStream) -> Result<Meta> {
        if x.peek(token::Paren) || x.peek(token::Bracket) || x.peek(token::Brace) {
            parse_meta_list_after_path(path, x).map(Meta::List)
        } else if x.peek(Token![=]) {
            parse_meta_name_value_after_path(path, x).map(Meta::NameValue)
        } else {
            Ok(Meta::Path(path))
        }
    }
    fn parse_meta_list_after_path(path: Path, x: ParseStream) -> Result<MetaList> {
        let (delimiter, tokens) = mac::parse_delimiter(x)?;
        Ok(MetaList {
            path,
            delimiter,
            tokens,
        })
    }
    fn parse_meta_name_value_after_path(path: Path, x: ParseStream) -> Result<MetaNameValue> {
        let eq_token: Token![=] = x.parse()?;
        let ahead = x.fork();
        let lit: Option<Lit> = ahead.parse()?;
        let value = if let (Some(lit), true) = (lit, ahead.is_empty()) {
            x.advance_to(&ahead);
            Expr::Lit(ExprLit { attrs: Vec::new(), lit })
        } else if x.peek(Token![#]) && x.peek2(token::Bracket) {
            return Err(x.error("unexpected attribute inside of attribute"));
        } else {
            x.parse()?
        };
        Ok(MetaNameValue { path, eq_token, value })
    }
    pub(super) struct DisplayAttrStyle<'a>(pub &'a AttrStyle);
    impl<'a> Display for DisplayAttrStyle<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str(match self.0 {
                AttrStyle::Outer => "#",
                AttrStyle::Inner(_) => "#!",
            })
        }
    }
    pub(super) struct DisplayPath<'a>(pub &'a Path);
    impl<'a> Display for DisplayPath<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            for (i, x) in self.0.segments.iter().enumerate() {
                if i > 0 || self.0.leading_colon.is_some() {
                    f.write_str("::")?;
                }
                write!(f, "{}", x.ident)?;
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
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.pound_token.to_tokens(xs);
            if let AttrStyle::Inner(x) = &self.style {
                x.to_tokens(xs);
            }
            self.bracket_token.surround(xs, |x| {
                self.meta.to_tokens(x);
            });
        }
    }
    impl ToTokens for MetaList {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.path.to_tokens(xs);
            self.delimiter.surround(xs, self.tokens.clone());
        }
    }
    impl ToTokens for MetaNameValue {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.path.to_tokens(xs);
            self.eq_token.to_tokens(xs);
            self.value.to_tokens(xs);
        }
    }
}
