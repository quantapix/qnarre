#![allow(non_camel_case_types)]
#![allow(
    clippy::bool_to_int_with_if,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment,
    clippy::default_trait_access,
    clippy::derivable_impls,
    clippy::doc_markdown,
    clippy::expl_impl_clone_on_copy,
    clippy::explicit_auto_deref,
    clippy::if_not_else,
    clippy::inherent_to_string,
    clippy::items_after_statements,
    clippy::large_enum_variant,
    clippy::let_underscore_untyped, // https://github.com/rust-lang/rust-clippy/issues/10410
    clippy::manual_assert,
    clippy::manual_let_else,
    clippy::match_like_matches_macro,
    clippy::match_on_vec_items,
    clippy::match_same_arms,
    clippy::match_wildcard_for_single_variants, // clippy bug: https://github.com/rust-lang/rust-clippy/issues/6984
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_doctest_main,
    clippy::needless_pass_by_value,
    clippy::never_loop,
    clippy::range_plus_one,
    clippy::redundant_else,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref,
    clippy::uninlined_format_args,
    clippy::unnecessary_box_returns,
    clippy::unnecessary_unwrap,
    clippy::used_underscore_binding,
    clippy::wildcard_imports,
)]

use crate::{
    lookahead,
    parse::{Parse, ParseStream, Parser, Result},
    token::{Brace, Bracket, Paren},
};
use proc_macro2::{extra::DelimSpan, Delimiter, Group, Ident, Span, TokenStream, TokenTree};
use quote::{spanned, ToTokens};
use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    ops,
    thread::{self, ThreadId},
};

extern crate proc_macro;
#[macro_use]
mod macros;
#[macro_use]
mod group {
    use crate::{err::Result, parse::ParseBuffer, token};
    use proc_macro2::{extra::DelimSpan, Delimiter};
    pub struct Parens<'a> {
        pub token: token::Paren,
        pub content: ParseBuffer<'a>,
    }
    pub fn parse_parens<'a>(x: &ParseBuffer<'a>) -> Result<Parens<'a>> {
        parse_delimited(x, Delimiter::Parenthesis).map(|(span, content)| Parens {
            token: token::Paren(span),
            content,
        })
    }
    #[macro_export]
    macro_rules! parenthesized {
        ($content:ident in $cur:expr) => {
            match $crate::group::parse_parens(&$cur) {
                $crate::__private::Ok(x) => {
                    $content = x.content;
                    x.token
                },
                $crate::__private::Err(x) => {
                    return $crate::__private::Err(x);
                },
            }
        };
    }
    pub struct Braces<'a> {
        pub token: token::Brace,
        pub content: ParseBuffer<'a>,
    }
    pub fn parse_braces<'a>(x: &ParseBuffer<'a>) -> Result<Braces<'a>> {
        parse_delimited(x, Delimiter::Brace).map(|(span, content)| Braces {
            token: token::Brace(span),
            content,
        })
    }
    #[macro_export]
    macro_rules! braced {
        ($content:ident in $cur:expr) => {
            match $crate::group::parse_braces(&$cur) {
                $crate::__private::Ok(x) => {
                    $content = x.content;
                    x.token
                },
                $crate::__private::Err(x) => {
                    return $crate::__private::Err(x);
                },
            }
        };
    }
    pub struct Brackets<'a> {
        pub token: token::Bracket,
        pub content: ParseBuffer<'a>,
    }
    pub fn parse_brackets<'a>(x: &ParseBuffer<'a>) -> Result<Brackets<'a>> {
        parse_delimited(x, Delimiter::Bracket).map(|(span, content)| Brackets {
            token: token::Bracket(span),
            content,
        })
    }
    #[macro_export]
    macro_rules! bracketed {
        ($content:ident in $cur:expr) => {
            match $crate::group::parse_brackets(&$cur) {
                $crate::__private::Ok(x) => {
                    $content = x.content;
                    x.token
                },
                $crate::__private::Err(x) => {
                    return $crate::__private::Err(x);
                },
            }
        };
    }
    pub struct Group<'a> {
        pub token: token::Group,
        pub content: ParseBuffer<'a>,
    }
    pub fn parse_group<'a>(x: &ParseBuffer<'a>) -> Result<Group<'a>> {
        parse_delimited(x, Delimiter::None).map(|(span, content)| Group {
            token: token::Group(span.join()),
            content,
        })
    }
    fn parse_delimited<'a>(x: &ParseBuffer<'a>, delim: Delimiter) -> Result<(DelimSpan, ParseBuffer<'a>)> {
        x.step(|cursor| {
            if let Some((content, span, rest)) = cursor.group(delim) {
                let scope = crate::buffer::close_span_of_group(*cursor);
                let nested = crate::parse::advance_step_cursor(cursor, content);
                let unexpected = crate::parse::get_unexpected(x);
                let content = crate::parse::new_parse_buffer(scope, nested, unexpected);
                Ok(((span, content), rest))
            } else {
                let y = match delim {
                    Delimiter::Parenthesis => "expected parentheses",
                    Delimiter::Brace => "expected curly braces",
                    Delimiter::Bracket => "expected square brackets",
                    Delimiter::None => "expected invisible group",
                };
                Err(cursor.error(y))
            }
        })
    }
}
#[macro_use]
pub mod token;

mod attr;
pub use crate::attr::{AttrStyle, Attribute, Meta, MetaList, MetaNameValue};
pub mod buffer;
mod expr;
pub use crate::expr::{
    Arm, Expr, ExprArray, ExprAssign, ExprAsync, ExprAwait, ExprBinary, ExprBlock, ExprBreak, ExprCall, ExprCast,
    ExprClosure, ExprConst, ExprContinue, ExprField, ExprForLoop, ExprGroup, ExprIf, ExprIndex, ExprInfer, ExprLet,
    ExprLit, ExprLoop, ExprMacro, ExprMatch, ExprMethodCall, ExprParen, ExprPath, ExprRange, ExprReference, ExprRepeat,
    ExprReturn, ExprStruct, ExprTry, ExprTryBlock, ExprTuple, ExprUnary, ExprUnsafe, ExprWhile, ExprYield, FieldValue,
    Index, Label, Member, RangeLimits,
};
pub use crate::expr::{
    ExprConst as PatConst, ExprLit as PatLit, ExprMacro as PatMacro, ExprPath as PatPath, ExprRange as PatRange,
};
mod generic;
pub use crate::generic::{
    BoundLifetimes, ConstParam, GenericParam, Generics, ImplGenerics, LifetimeParam, PredicateLifetime, PredicateType,
    TraitBound, TraitBoundModifier, Turbofish, TypeGenerics, TypeParam, TypeParamBound, WhereClause, WherePredicate,
};
mod item;
pub use crate::item::{
    FnArg, ForeignItem, ForeignItemFn, ForeignItemMacro, ForeignItemStatic, ForeignItemType, ImplItem, ImplItemConst,
    ImplItemFn, ImplItemMacro, ImplItemType, ImplRestriction, Item, ItemConst, ItemEnum, ItemExternCrate, ItemFn,
    ItemForeignMod, ItemImpl, ItemMacro, ItemMod, ItemStatic, ItemStruct, ItemTrait, ItemTraitAlias, ItemType,
    ItemUnion, ItemUse, Receiver, Signature, StaticMutability, TraitItem, TraitItemConst, TraitItemFn, TraitItemMacro,
    TraitItemType, UseGlob, UseGroup, UseName, UsePath, UseRename, UseTree, Variadic,
};
pub mod punctuated;
use punctuated::Punctuated;
mod lit;
pub use crate::lit::{Lit, LitBool, LitByte, LitByteStr, LitChar, LitFloat, LitInt, LitStr, StrStyle};
mod pat;
pub use crate::pat::{
    FieldPat, Pat, PatIdent, PatOr, PatParen, PatReference, PatRest, PatSlice, PatStruct, PatTuple, PatTupleStruct,
    PatType, PatWild,
};
mod path;
pub use crate::path::{
    AngleBracketedGenericArguments, AssocConst, AssocType, Constraint, GenericArgument, ParenthesizedGenericArguments,
    Path, PathArguments, PathSegment, QSelf,
};

ast_struct! {
    pub struct Block {
        pub brace_token: token::Brace,
        pub stmts: Vec<Stmt>,
    }
}
ast_enum! {
    pub enum Stmt {
        Local(Local),
        Item(Item),
        Expr(Expr, Option<Token![;]>),
        Macro(StmtMacro),
    }
}
ast_struct! {
    pub struct Local {
        pub attrs: Vec<Attribute>,
        pub let_token: Token![let],
        pub pat: Pat,
        pub init: Option<LocalInit>,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct LocalInit {
        pub eq_token: Token![=],
        pub expr: Box<Expr>,
        pub diverge: Option<(Token![else], Box<Expr>)>,
    }
}
ast_struct! {
    pub struct StmtMacro {
        pub attrs: Vec<Attribute>,
        pub mac: Macro,
        pub semi_token: Option<Token![;]>,
    }
}

mod parsing {
    use super::*;
    use crate::parse::{discouraged::Speculative, Parse, ParseStream, Result};
    use proc_macro2::TokenStream;
    struct AllowNoSemi(bool);
    impl Block {
        pub fn parse_within(x: ParseStream) -> Result<Vec<Stmt>> {
            let mut ys = Vec::new();
            loop {
                while let semi @ Some(_) = x.parse()? {
                    ys.push(Stmt::Expr(Expr::Verbatim(TokenStream::new()), semi));
                }
                if x.is_empty() {
                    break;
                }
                let stmt = parse_stmt(x, AllowNoSemi(true))?;
                let requires_semicolon = match &stmt {
                    Stmt::Expr(x, None) => expr::requires_terminator(x),
                    Stmt::Macro(x) => x.semi_token.is_none() && !x.mac.delimiter.is_brace(),
                    Stmt::Local(_) | Stmt::Item(_) | Stmt::Expr(_, Some(_)) => false,
                };
                ys.push(stmt);
                if x.is_empty() {
                    break;
                } else if requires_semicolon {
                    return Err(x.error("unexpected token, expected `;`"));
                }
            }
            Ok(ys)
        }
    }
    impl Parse for Block {
        fn parse(x: ParseStream) -> Result<Self> {
            let content;
            Ok(Block {
                brace_token: braced!(content in x),
                stmts: content.call(Block::parse_within)?,
            })
        }
    }
    impl Parse for Stmt {
        fn parse(x: ParseStream) -> Result<Self> {
            let allow_nosemi = AllowNoSemi(false);
            parse_stmt(x, allow_nosemi)
        }
    }
    fn parse_stmt(x: ParseStream, allow_nosemi: AllowNoSemi) -> Result<Stmt> {
        let begin = x.fork();
        let attrs = x.call(Attribute::parse_outer)?;
        let ahead = x.fork();
        let mut is_item_macro = false;
        if let Ok(path) = ahead.call(Path::parse_mod_style) {
            if ahead.peek(Token![!]) {
                if ahead.peek2(Ident) || ahead.peek2(Token![try]) {
                    is_item_macro = true;
                } else if ahead.peek2(token::Brace) && !(ahead.peek3(Token![.]) || ahead.peek3(Token![?])) {
                    x.advance_to(&ahead);
                    return stmt_mac(x, attrs, path).map(Stmt::Macro);
                }
            }
        }
        if x.peek(Token![let]) {
            stmt_local(x, attrs).map(Stmt::Local)
        } else if x.peek(Token![pub])
            || x.peek(Token![crate]) && !x.peek2(Token![::])
            || x.peek(Token![extern])
            || x.peek(Token![use])
            || x.peek(Token![static])
                && (x.peek2(Token![mut])
                    || x.peek2(Ident) && !(x.peek2(Token![async]) && (x.peek3(Token![move]) || x.peek3(Token![|]))))
            || x.peek(Token![const])
                && !(x.peek2(token::Brace)
                    || x.peek2(Token![static])
                    || x.peek2(Token![async])
                        && !(x.peek3(Token![unsafe]) || x.peek3(Token![extern]) || x.peek3(Token![fn]))
                    || x.peek2(Token![move])
                    || x.peek2(Token![|]))
            || x.peek(Token![unsafe]) && !x.peek2(token::Brace)
            || x.peek(Token![async]) && (x.peek2(Token![unsafe]) || x.peek2(Token![extern]) || x.peek2(Token![fn]))
            || x.peek(Token![fn])
            || x.peek(Token![mod])
            || x.peek(Token![type])
            || x.peek(Token![struct])
            || x.peek(Token![enum])
            || x.peek(Token![union]) && x.peek2(Ident)
            || x.peek(Token![auto]) && x.peek2(Token![trait])
            || x.peek(Token![trait])
            || x.peek(Token![default]) && (x.peek2(Token![unsafe]) || x.peek2(Token![impl]))
            || x.peek(Token![impl])
            || x.peek(Token![macro])
            || is_item_macro
        {
            let item = item::parsing::parse_rest_of_item(begin, attrs, x)?;
            Ok(Stmt::Item(item))
        } else {
            stmt_expr(x, allow_nosemi, attrs)
        }
    }
    fn stmt_mac(x: ParseStream, attrs: Vec<Attribute>, path: Path) -> Result<StmtMacro> {
        let bang_token: Token![!] = x.parse()?;
        let (delimiter, tokens) = mac_parse_delimiter(x)?;
        let semi_token: Option<Token![;]> = x.parse()?;
        Ok(StmtMacro {
            attrs,
            mac: Macro {
                path,
                bang_token,
                delimiter,
                tokens,
            },
            semi_token,
        })
    }
    fn stmt_local(x: ParseStream, attrs: Vec<Attribute>) -> Result<Local> {
        let let_token: Token![let] = x.parse()?;
        let mut pat = Pat::parse_single(x)?;
        if x.peek(Token![:]) {
            let colon_token: Token![:] = x.parse()?;
            let ty: Type = x.parse()?;
            pat = Pat::Type(PatType {
                attrs: Vec::new(),
                pat: Box::new(pat),
                colon_token,
                ty: Box::new(ty),
            });
        }
        let init = if let Some(eq_token) = x.parse()? {
            let eq_token: Token![=] = eq_token;
            let expr: Expr = x.parse()?;
            let diverge = if let Some(else_token) = x.parse()? {
                let else_token: Token![else] = else_token;
                let diverge = ExprBlock {
                    attrs: Vec::new(),
                    label: None,
                    block: x.parse()?,
                };
                Some((else_token, Box::new(Expr::Block(diverge))))
            } else {
                None
            };
            Some(LocalInit {
                eq_token,
                expr: Box::new(expr),
                diverge,
            })
        } else {
            None
        };
        let semi_token: Token![;] = x.parse()?;
        Ok(Local {
            attrs,
            let_token,
            pat,
            init,
            semi_token,
        })
    }
    fn stmt_expr(x: ParseStream, allow_nosemi: AllowNoSemi, mut attrs: Vec<Attribute>) -> Result<Stmt> {
        let mut e = expr::parsing::expr_early(x)?;
        let mut attr_target = &mut e;
        loop {
            attr_target = match attr_target {
                Expr::Assign(e) => &mut e.left,
                Expr::Binary(e) => &mut e.left,
                Expr::Cast(e) => &mut e.expr,
                Expr::Array(_)
                | Expr::Async(_)
                | Expr::Await(_)
                | Expr::Block(_)
                | Expr::Break(_)
                | Expr::Call(_)
                | Expr::Closure(_)
                | Expr::Const(_)
                | Expr::Continue(_)
                | Expr::Field(_)
                | Expr::ForLoop(_)
                | Expr::Group(_)
                | Expr::If(_)
                | Expr::Index(_)
                | Expr::Infer(_)
                | Expr::Let(_)
                | Expr::Lit(_)
                | Expr::Loop(_)
                | Expr::Macro(_)
                | Expr::Match(_)
                | Expr::MethodCall(_)
                | Expr::Paren(_)
                | Expr::Path(_)
                | Expr::Range(_)
                | Expr::Reference(_)
                | Expr::Repeat(_)
                | Expr::Return(_)
                | Expr::Struct(_)
                | Expr::Try(_)
                | Expr::TryBlock(_)
                | Expr::Tuple(_)
                | Expr::Unary(_)
                | Expr::Unsafe(_)
                | Expr::While(_)
                | Expr::Yield(_)
                | Expr::Verbatim(_) => break,
            };
        }
        attrs.extend(attr_target.replace_attrs(Vec::new()));
        attr_target.replace_attrs(attrs);
        let semi_token: Option<Token![;]> = x.parse()?;
        match e {
            Expr::Macro(ExprMacro { attrs, mac }) if semi_token.is_some() || mac.delimiter.is_brace() => {
                return Ok(Stmt::Macro(StmtMacro { attrs, mac, semi_token }));
            },
            _ => {},
        }
        if semi_token.is_some() {
            Ok(Stmt::Expr(e, semi_token))
        } else if allow_nosemi.0 || !expr::requires_terminator(&e) {
            Ok(Stmt::Expr(e, None))
        } else {
            Err(x.error("expected semicolon"))
        }
    }
}
mod printing {
    use super::*;
    use proc_macro2::TokenStream;
    use quote::{ToTokens, TokenStreamExt};
    impl ToTokens for Block {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.brace_token.surround(tokens, |tokens| {
                tokens.append_all(&self.stmts);
            });
        }
    }
    impl ToTokens for Stmt {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                Stmt::Local(local) => local.to_tokens(tokens),
                Stmt::Item(item) => item.to_tokens(tokens),
                Stmt::Expr(expr, semi) => {
                    expr.to_tokens(tokens);
                    semi.to_tokens(tokens);
                },
                Stmt::Macro(mac) => mac.to_tokens(tokens),
            }
        }
    }
    impl ToTokens for Local {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            expr::printing::outer_attrs_to_tokens(&self.attrs, tokens);
            self.let_token.to_tokens(tokens);
            self.pat.to_tokens(tokens);
            if let Some(init) = &self.init {
                init.eq_token.to_tokens(tokens);
                init.expr.to_tokens(tokens);
                if let Some((else_token, diverge)) = &init.diverge {
                    else_token.to_tokens(tokens);
                    diverge.to_tokens(tokens);
                }
            }
            self.semi_token.to_tokens(tokens);
        }
    }
    impl ToTokens for StmtMacro {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            expr::printing::outer_attrs_to_tokens(&self.attrs, tokens);
            self.mac.to_tokens(tokens);
            self.semi_token.to_tokens(tokens);
        }
    }
}

mod ty;
pub use crate::ty::{
    Abi, BareFnArg, BareVariadic, ReturnType, Type, TypeArray, TypeBareFn, TypeGroup, TypeImplTrait, TypeInfer,
    TypeMacro, TypeNever, TypeParen, TypePath, TypePtr, TypeReference, TypeSlice, TypeTraitObject, TypeTuple,
};

pub struct BigInt {
    digits: Vec<u8>,
}
impl BigInt {
    pub fn new() -> Self {
        BigInt { digits: Vec::new() }
    }
    pub fn to_string(&self) -> String {
        let mut y = String::with_capacity(self.digits.len());
        let mut has_nonzero = false;
        for x in self.digits.iter().rev() {
            has_nonzero |= *x != 0;
            if has_nonzero {
                y.push((*x + b'0') as char);
            }
        }
        if y.is_empty() {
            y.push('0');
        }
        y
    }
    fn reserve_two_digits(&mut self) {
        let len = self.digits.len();
        let desired = len + !self.digits.ends_with(&[0, 0]) as usize + !self.digits.ends_with(&[0]) as usize;
        self.digits.resize(desired, 0);
    }
}
impl ops::AddAssign<u8> for BigInt {
    fn add_assign(&mut self, mut increment: u8) {
        self.reserve_two_digits();
        let mut i = 0;
        while increment > 0 {
            let sum = self.digits[i] + increment;
            self.digits[i] = sum % 10;
            increment = sum / 10;
            i += 1;
        }
    }
}
impl ops::MulAssign<u8> for BigInt {
    fn mul_assign(&mut self, base: u8) {
        self.reserve_two_digits();
        let mut carry = 0;
        for digit in &mut self.digits {
            let prod = *digit * base + carry;
            *digit = prod % 10;
            carry = prod / 10;
        }
    }
}

ast_struct! {
    pub struct Variant {
        pub attrs: Vec<Attribute>,
        pub ident: Ident,
        pub fields: Fields,
        pub discriminant: Option<(Token![=], Expr)>,
    }
}
ast_enum_of_structs! {
    pub enum Fields {
        Named(FieldsNamed),
        Unnamed(FieldsUnnamed),
        Unit,
    }
}
ast_struct! {
    pub struct FieldsNamed {
        pub brace_token: token::Brace,
        pub named: Punctuated<Field, Token![,]>,
    }
}
ast_struct! {
    pub struct FieldsUnnamed {
        pub paren_token: token::Paren,
        pub unnamed: Punctuated<Field, Token![,]>,
    }
}
impl Fields {
    pub fn iter(&self) -> punctuated::Iter<Field> {
        match self {
            Fields::Unit => crate::punctuated::empty_punctuated_iter(),
            Fields::Named(f) => f.named.iter(),
            Fields::Unnamed(f) => f.unnamed.iter(),
        }
    }
    pub fn iter_mut(&mut self) -> punctuated::IterMut<Field> {
        match self {
            Fields::Unit => crate::punctuated::empty_punctuated_iter_mut(),
            Fields::Named(f) => f.named.iter_mut(),
            Fields::Unnamed(f) => f.unnamed.iter_mut(),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Fields::Unit => 0,
            Fields::Named(f) => f.named.len(),
            Fields::Unnamed(f) => f.unnamed.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        match self {
            Fields::Unit => true,
            Fields::Named(f) => f.named.is_empty(),
            Fields::Unnamed(f) => f.unnamed.is_empty(),
        }
    }
}
impl IntoIterator for Fields {
    type Item = Field;
    type IntoIter = punctuated::IntoIter<Field>;
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Fields::Unit => Punctuated::<Field, ()>::new().into_iter(),
            Fields::Named(f) => f.named.into_iter(),
            Fields::Unnamed(f) => f.unnamed.into_iter(),
        }
    }
}
impl<'a> IntoIterator for &'a Fields {
    type Item = &'a Field;
    type IntoIter = punctuated::Iter<'a, Field>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a> IntoIterator for &'a mut Fields {
    type Item = &'a mut Field;
    type IntoIter = punctuated::IterMut<'a, Field>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
ast_struct! {
    pub struct Field {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub mutability: FieldMutability,
        pub ident: Option<Ident>,
        pub colon_token: Option<Token![:]>,
        pub ty: Type,
    }
}

mod parsing {
    use super::*;
    use crate::ext::IdentExt;
    use crate::parse::{Parse, ParseStream, Result};
    impl Parse for Variant {
        fn parse(input: ParseStream) -> Result<Self> {
            let attrs = input.call(Attribute::parse_outer)?;
            let _visibility: Visibility = input.parse()?;
            let ident: Ident = input.parse()?;
            let fields = if input.peek(token::Brace) {
                Fields::Named(input.parse()?)
            } else if input.peek(token::Paren) {
                Fields::Unnamed(input.parse()?)
            } else {
                Fields::Unit
            };
            let discriminant = if input.peek(Token![=]) {
                let eq_token: Token![=] = input.parse()?;
                let discriminant: Expr = input.parse()?;
                Some((eq_token, discriminant))
            } else {
                None
            };
            Ok(Variant {
                attrs,
                ident,
                fields,
                discriminant,
            })
        }
    }
    impl Parse for FieldsNamed {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            Ok(FieldsNamed {
                brace_token: braced!(content in input),
                named: content.parse_terminated(Field::parse_named, Token![,])?,
            })
        }
    }
    impl Parse for FieldsUnnamed {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            Ok(FieldsUnnamed {
                paren_token: parenthesized!(content in input),
                unnamed: content.parse_terminated(Field::parse_unnamed, Token![,])?,
            })
        }
    }
    impl Field {
        pub fn parse_named(input: ParseStream) -> Result<Self> {
            Ok(Field {
                attrs: input.call(Attribute::parse_outer)?,
                vis: input.parse()?,
                mutability: FieldMutability::None,
                ident: Some(if input.peek(Token![_]) {
                    input.call(Ident::parse_any)
                } else {
                    input.parse()
                }?),
                colon_token: Some(input.parse()?),
                ty: input.parse()?,
            })
        }
        pub fn parse_unnamed(input: ParseStream) -> Result<Self> {
            Ok(Field {
                attrs: input.call(Attribute::parse_outer)?,
                vis: input.parse()?,
                mutability: FieldMutability::None,
                ident: None,
                colon_token: None,
                ty: input.parse()?,
            })
        }
    }
}
mod printing {
    use super::*;
    use crate::TokensOrDefault;
    use proc_macro2::TokenStream;
    use quote::{ToTokens, TokenStreamExt};
    impl ToTokens for Variant {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.append_all(&self.attrs);
            self.ident.to_tokens(tokens);
            self.fields.to_tokens(tokens);
            if let Some((eq_token, disc)) = &self.discriminant {
                eq_token.to_tokens(tokens);
                disc.to_tokens(tokens);
            }
        }
    }
    impl ToTokens for FieldsNamed {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.brace_token.surround(tokens, |tokens| {
                self.named.to_tokens(tokens);
            });
        }
    }
    impl ToTokens for FieldsUnnamed {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.paren_token.surround(tokens, |tokens| {
                self.unnamed.to_tokens(tokens);
            });
        }
    }
    impl ToTokens for Field {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.append_all(&self.attrs);
            self.vis.to_tokens(tokens);
            if let Some(ident) = &self.ident {
                ident.to_tokens(tokens);
                TokensOrDefault(&self.colon_token).to_tokens(tokens);
            }
            self.ty.to_tokens(tokens);
        }
    }
}

ast_struct! {
    pub struct DeriveInput {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub ident: Ident,
        pub generics: Generics,
        pub data: Data,
    }
}
ast_enum! {
    pub enum Data {
        Struct(DataStruct),
        Enum(DataEnum),
        Union(DataUnion),
    }
}
ast_struct! {
    pub struct DataStruct {
        pub struct_token: Token![struct],
        pub fields: Fields,
        pub semi_token: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct DataEnum {
        pub enum_token: Token![enum],
        pub brace_token: token::Brace,
        pub variants: Punctuated<Variant, Token![,]>,
    }
}
ast_struct! {
    pub struct DataUnion {
        pub union_token: Token![union],
        pub fields: FieldsNamed,
    }
}

mod parsing {
    use super::*;
    use crate::parse::{Parse, ParseStream, Result};
    impl Parse for DeriveInput {
        fn parse(input: ParseStream) -> Result<Self> {
            let attrs = input.call(Attribute::parse_outer)?;
            let vis = input.parse::<Visibility>()?;
            let lookahead = input.lookahead1();
            if lookahead.peek(Token![struct]) {
                let struct_token = input.parse::<Token![struct]>()?;
                let ident = input.parse::<Ident>()?;
                let generics = input.parse::<Generics>()?;
                let (where_clause, fields, semi) = data_struct(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    generics: Generics {
                        where_clause,
                        ..generics
                    },
                    data: Data::Struct(DataStruct {
                        struct_token,
                        fields,
                        semi_token: semi,
                    }),
                })
            } else if lookahead.peek(Token![enum]) {
                let enum_token = input.parse::<Token![enum]>()?;
                let ident = input.parse::<Ident>()?;
                let generics = input.parse::<Generics>()?;
                let (where_clause, brace, variants) = data_enum(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    generics: Generics {
                        where_clause,
                        ..generics
                    },
                    data: Data::Enum(DataEnum {
                        enum_token,
                        brace_token: brace,
                        variants,
                    }),
                })
            } else if lookahead.peek(Token![union]) {
                let union_token = input.parse::<Token![union]>()?;
                let ident = input.parse::<Ident>()?;
                let generics = input.parse::<Generics>()?;
                let (where_clause, fields) = data_union(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    generics: Generics {
                        where_clause,
                        ..generics
                    },
                    data: Data::Union(DataUnion { union_token, fields }),
                })
            } else {
                Err(lookahead.error())
            }
        }
    }
    pub fn data_struct(input: ParseStream) -> Result<(Option<WhereClause>, Fields, Option<Token![;]>)> {
        let mut lookahead = input.lookahead1();
        let mut where_clause = None;
        if lookahead.peek(Token![where]) {
            where_clause = Some(input.parse()?);
            lookahead = input.lookahead1();
        }
        if where_clause.is_none() && lookahead.peek(token::Paren) {
            let fields = input.parse()?;
            lookahead = input.lookahead1();
            if lookahead.peek(Token![where]) {
                where_clause = Some(input.parse()?);
                lookahead = input.lookahead1();
            }
            if lookahead.peek(Token![;]) {
                let semi = input.parse()?;
                Ok((where_clause, Fields::Unnamed(fields), Some(semi)))
            } else {
                Err(lookahead.error())
            }
        } else if lookahead.peek(token::Brace) {
            let fields = input.parse()?;
            Ok((where_clause, Fields::Named(fields), None))
        } else if lookahead.peek(Token![;]) {
            let semi = input.parse()?;
            Ok((where_clause, Fields::Unit, Some(semi)))
        } else {
            Err(lookahead.error())
        }
    }
    pub fn data_enum(
        input: ParseStream,
    ) -> Result<(Option<WhereClause>, token::Brace, Punctuated<Variant, Token![,]>)> {
        let where_clause = input.parse()?;
        let content;
        let brace = braced!(content in input);
        let variants = content.parse_terminated(Variant::parse, Token![,])?;
        Ok((where_clause, brace, variants))
    }
    pub fn data_union(input: ParseStream) -> Result<(Option<WhereClause>, FieldsNamed)> {
        let where_clause = input.parse()?;
        let fields = input.parse()?;
        Ok((where_clause, fields))
    }
}
mod printing {
    use super::*;
    use crate::{attr::FilterAttrs, TokensOrDefault};
    use proc_macro2::TokenStream;
    use quote::ToTokens;
    impl ToTokens for DeriveInput {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            for attr in self.attrs.outer() {
                attr.to_tokens(tokens);
            }
            self.vis.to_tokens(tokens);
            match &self.data {
                Data::Struct(d) => d.struct_token.to_tokens(tokens),
                Data::Enum(d) => d.enum_token.to_tokens(tokens),
                Data::Union(d) => d.union_token.to_tokens(tokens),
            }
            self.ident.to_tokens(tokens);
            self.generics.to_tokens(tokens);
            match &self.data {
                Data::Struct(data) => match &data.fields {
                    Fields::Named(fields) => {
                        self.generics.where_clause.to_tokens(tokens);
                        fields.to_tokens(tokens);
                    },
                    Fields::Unnamed(fields) => {
                        fields.to_tokens(tokens);
                        self.generics.where_clause.to_tokens(tokens);
                        TokensOrDefault(&data.semi_token).to_tokens(tokens);
                    },
                    Fields::Unit => {
                        self.generics.where_clause.to_tokens(tokens);
                        TokensOrDefault(&data.semi_token).to_tokens(tokens);
                    },
                },
                Data::Enum(data) => {
                    self.generics.where_clause.to_tokens(tokens);
                    data.brace_token.surround(tokens, |tokens| {
                        data.variants.to_tokens(tokens);
                    });
                },
                Data::Union(data) => {
                    self.generics.where_clause.to_tokens(tokens);
                    data.fields.to_tokens(tokens);
                },
            }
        }
    }
}

mod err {
    use crate::{buffer::Cursor, ThreadBound};
    use proc_macro2::{Delimiter, Group, Ident, LexError, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
    use quote::ToTokens;
    use std::fmt::{self, Debug, Display};
    use std::slice;
    use std::vec;
    pub type Result<T> = std::result::Result<T, Err>;
    pub struct Err {
        messages: Vec<ErrMsg>,
    }
    struct ErrMsg {
        span: ThreadBound<SpanRange>,
        message: String,
    }
    struct SpanRange {
        start: Span,
        end: Span,
    }
    #[cfg(test)]
    struct _Test
    where
        Err: Send + Sync;
    impl Err {
        pub fn new<T: Display>(span: Span, message: T) -> Self {
            return new(span, message.to_string());
            fn new(span: Span, message: String) -> Err {
                Err {
                    messages: vec![ErrMsg {
                        span: ThreadBound::new(SpanRange { start: span, end: span }),
                        message,
                    }],
                }
            }
        }
        pub fn new_spanned<T: ToTokens, U: Display>(tokens: T, message: U) -> Self {
            return new_spanned(tokens.into_token_stream(), message.to_string());
            fn new_spanned(tokens: TokenStream, message: String) -> Err {
                let mut iter = tokens.into_iter();
                let start = iter.next().map_or_else(Span::call_site, |t| t.span());
                let end = iter.last().map_or(start, |t| t.span());
                Err {
                    messages: vec![ErrMsg {
                        span: ThreadBound::new(SpanRange { start, end }),
                        message,
                    }],
                }
            }
        }
        pub fn span(&self) -> Span {
            let SpanRange { start, end } = match self.messages[0].span.get() {
                Some(span) => *span,
                None => return Span::call_site(),
            };
            start.join(end).unwrap_or(start)
        }
        pub fn to_compile_error(&self) -> TokenStream {
            self.messages.iter().map(ErrMsg::to_compile_error).collect()
        }
        pub fn into_compile_error(self) -> TokenStream {
            self.to_compile_error()
        }
        pub fn combine(&mut self, another: Err) {
            self.messages.extend(another.messages);
        }
    }
    impl ErrMsg {
        fn to_compile_error(&self) -> TokenStream {
            let (start, end) = match self.span.get() {
                Some(range) => (range.start, range.end),
                None => (Span::call_site(), Span::call_site()),
            };
            TokenStream::from_iter(vec![
                TokenTree::Punct({
                    let mut punct = Punct::new(':', Spacing::Joint);
                    punct.set_span(start);
                    punct
                }),
                TokenTree::Punct({
                    let mut punct = Punct::new(':', Spacing::Alone);
                    punct.set_span(start);
                    punct
                }),
                TokenTree::Ident(Ident::new("core", start)),
                TokenTree::Punct({
                    let mut punct = Punct::new(':', Spacing::Joint);
                    punct.set_span(start);
                    punct
                }),
                TokenTree::Punct({
                    let mut punct = Punct::new(':', Spacing::Alone);
                    punct.set_span(start);
                    punct
                }),
                TokenTree::Ident(Ident::new("compile_error", start)),
                TokenTree::Punct({
                    let mut punct = Punct::new('!', Spacing::Alone);
                    punct.set_span(start);
                    punct
                }),
                TokenTree::Group({
                    let mut group = Group::new(Delimiter::Brace, {
                        TokenStream::from_iter(vec![TokenTree::Literal({
                            let mut string = Literal::string(&self.message);
                            string.set_span(end);
                            string
                        })])
                    });
                    group.set_span(end);
                    group
                }),
            ])
        }
    }
    pub fn new_at<T: Display>(scope: Span, cursor: Cursor, message: T) -> Err {
        if cursor.eof() {
            Err::new(scope, format!("unexpected end of input, {}", message))
        } else {
            let span = crate::buffer::open_span_of_group(cursor);
            Err::new(span, message)
        }
    }
    pub fn new2<T: Display>(start: Span, end: Span, message: T) -> Err {
        return new2(start, end, message.to_string());
        fn new2(start: Span, end: Span, message: String) -> Err {
            Err {
                messages: vec![ErrMsg {
                    span: ThreadBound::new(SpanRange { start, end }),
                    message,
                }],
            }
        }
    }
    impl Debug for Err {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            if self.messages.len() == 1 {
                formatter.debug_tuple("Error").field(&self.messages[0]).finish()
            } else {
                formatter.debug_tuple("Error").field(&self.messages).finish()
            }
        }
    }
    impl Debug for ErrMsg {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(&self.message, formatter)
        }
    }
    impl Display for Err {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str(&self.messages[0].message)
        }
    }
    impl Clone for Err {
        fn clone(&self) -> Self {
            Err {
                messages: self.messages.clone(),
            }
        }
    }
    impl Clone for ErrMsg {
        fn clone(&self) -> Self {
            ErrMsg {
                span: self.span,
                message: self.message.clone(),
            }
        }
    }
    impl Clone for SpanRange {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl Copy for SpanRange {}
    impl std::error::Error for Err {}
    impl From<LexError> for Err {
        fn from(err: LexError) -> Self {
            Err::new(err.span(), "lex error")
        }
    }
    impl IntoIterator for Err {
        type Item = Err;
        type IntoIter = IntoIter;
        fn into_iter(self) -> Self::IntoIter {
            IntoIter {
                messages: self.messages.into_iter(),
            }
        }
    }
    pub struct IntoIter {
        messages: vec::IntoIter<ErrMsg>,
    }
    impl Iterator for IntoIter {
        type Item = Err;
        fn next(&mut self) -> Option<Self::Item> {
            Some(Err {
                messages: vec![self.messages.next()?],
            })
        }
    }
    impl<'a> IntoIterator for &'a Err {
        type Item = Err;
        type IntoIter = Iter<'a>;
        fn into_iter(self) -> Self::IntoIter {
            Iter {
                messages: self.messages.iter(),
            }
        }
    }
    pub struct Iter<'a> {
        messages: slice::Iter<'a, ErrMsg>,
    }
    impl<'a> Iterator for Iter<'a> {
        type Item = Err;
        fn next(&mut self) -> Option<Self::Item> {
            Some(Err {
                messages: vec![self.messages.next()?.clone()],
            })
        }
    }
    impl Extend<Err> for Err {
        fn extend<T: IntoIterator<Item = Err>>(&mut self, iter: T) {
            for err in iter {
                self.combine(err);
            }
        }
    }
}
pub use crate::err::{Err, Result};

pub mod ext {
    use crate::buffer::Cursor;
    use crate::parse::Peek;
    use crate::parse::{ParseStream, Result};
    use crate::sealed::lookahead;
    use crate::token::CustomToken;
    use proc_macro2::Ident;
    pub trait IdentExt: Sized + private::Sealed {
        fn parse_any(input: ParseStream) -> Result<Self>;
        #[allow(non_upper_case_globals)]
        const peek_any: private::PeekFn = private::PeekFn;
        fn unraw(&self) -> Ident;
    }
    impl IdentExt for Ident {
        fn parse_any(input: ParseStream) -> Result<Self> {
            input.step(|cursor| match cursor.ident() {
                Some((ident, rest)) => Ok((ident, rest)),
                None => Err(cursor.error("expected ident")),
            })
        }
        fn unraw(&self) -> Ident {
            let string = self.to_string();
            if let Some(string) = string.strip_prefix("r#") {
                Ident::new(string, self.span())
            } else {
                self.clone()
            }
        }
    }
    impl Peek for private::PeekFn {
        type Token = private::IdentAny;
    }
    impl CustomToken for private::IdentAny {
        fn peek(cursor: Cursor) -> bool {
            cursor.ident().is_some()
        }
        fn display() -> &'static str {
            "identifier"
        }
    }
    impl lookahead::Sealed for private::PeekFn {}
    mod private {
        use proc_macro2::Ident;
        pub trait Sealed {}
        impl Sealed for Ident {}
        pub struct PeekFn;
        pub struct IdentAny;
        impl Copy for PeekFn {}
        impl Clone for PeekFn {
            fn clone(&self) -> Self {
                *self
            }
        }
    }
}

ast_struct! {
    pub struct File {
        pub shebang: Option<String>,
        pub attrs: Vec<Attribute>,
        pub items: Vec<Item>,
    }
}
mod parsing {
    use super::*;
    use crate::parse::{Parse, ParseStream, Result};
    impl Parse for File {
        fn parse(input: ParseStream) -> Result<Self> {
            Ok(File {
                shebang: None,
                attrs: input.call(Attribute::parse_inner)?,
                items: {
                    let mut items = Vec::new();
                    while !input.is_empty() {
                        items.push(input.parse()?);
                    }
                    items
                },
            })
        }
    }
}
mod printing {
    use super::*;
    use crate::attr::FilterAttrs;
    use proc_macro2::TokenStream;
    use quote::{ToTokens, TokenStreamExt};
    impl ToTokens for File {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.append_all(self.attrs.inner());
            tokens.append_all(&self.items);
        }
    }
}

mod ident {
    use crate::lookahead;
    pub use proc_macro2::Ident;
    #[allow(non_snake_case)]
    pub fn Ident(x: lookahead::TokenMarker) -> Ident {
        match x {}
    }
    macro_rules! ident_from_token {
        ($token:ident) => {
            impl From<Token![$token]> for Ident {
                fn from(token: Token![$token]) -> Ident {
                    Ident::new(stringify!($token), token.span)
                }
            }
        };
    }
    ident_from_token!(self);
    ident_from_token!(Self);
    ident_from_token!(super);
    ident_from_token!(crate);
    ident_from_token!(extern);
    impl From<Token![_]> for Ident {
        fn from(token: Token![_]) -> Ident {
            Ident::new("_", token.span)
        }
    }
    pub fn xid_ok(symbol: &str) -> bool {
        let mut chars = symbol.chars();
        let first = chars.next().unwrap();
        if !(first == '_' || unicode_ident::is_xid_start(first)) {
            return false;
        }
        for ch in chars {
            if !unicode_ident::is_xid_continue(ch) {
                return false;
            }
        }
        true
    }
    mod parsing {
        use crate::buffer::Cursor;
        use crate::parse::{Parse, ParseStream, Result};
        use crate::token::Token;
        use proc_macro2::Ident;
        fn accept_as_ident(ident: &Ident) -> bool {
            match ident.to_string().as_str() {
                "_" | "abstract" | "as" | "async" | "await" | "become" | "box" | "break" | "const" | "continue"
                | "crate" | "do" | "dyn" | "else" | "enum" | "extern" | "false" | "final" | "fn" | "for" | "if"
                | "impl" | "in" | "let" | "loop" | "macro" | "match" | "mod" | "move" | "mut" | "override" | "priv"
                | "pub" | "ref" | "return" | "Self" | "self" | "static" | "struct" | "super" | "trait" | "true"
                | "try" | "type" | "typeof" | "unsafe" | "unsized" | "use" | "virtual" | "where" | "while"
                | "yield" => false,
                _ => true,
            }
        }

        impl Parse for Ident {
            fn parse(input: ParseStream) -> Result<Self> {
                input.step(|cursor| {
                    if let Some((ident, rest)) = cursor.ident() {
                        if accept_as_ident(&ident) {
                            Ok((ident, rest))
                        } else {
                            Err(cursor.error(format_args!("expected identifier, found keyword `{}`", ident,)))
                        }
                    } else {
                        Err(cursor.error("expected identifier"))
                    }
                })
            }
        }
        impl Token for Ident {
            fn peek(cursor: Cursor) -> bool {
                if let Some((ident, _rest)) = cursor.ident() {
                    accept_as_ident(&ident)
                } else {
                    false
                }
            }
            fn display() -> &'static str {
                "identifier"
            }
        }
    }
}
pub use crate::ident::Ident;

pub struct Lifetime {
    pub apostrophe: Span,
    pub ident: Ident,
}
impl Lifetime {
    pub fn new(symbol: &str, span: Span) -> Self {
        if !symbol.starts_with('\'') {
            panic!(
                "lifetime name must start with apostrophe as in \"'a\", got {:?}",
                symbol
            );
        }
        if symbol == "'" {
            panic!("lifetime name must not be empty");
        }
        if !crate::ident::xid_ok(&symbol[1..]) {
            panic!("{:?} is not a valid lifetime name", symbol);
        }
        Lifetime {
            apostrophe: span,
            ident: Ident::new(&symbol[1..], span),
        }
    }
    pub fn span(&self) -> Span {
        self.apostrophe.join(self.ident.span()).unwrap_or(self.apostrophe)
    }
    pub fn set_span(&mut self, span: Span) {
        self.apostrophe = span;
        self.ident.set_span(span);
    }
}
impl Display for Lifetime {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        "'".fmt(formatter)?;
        self.ident.fmt(formatter)
    }
}
impl Clone for Lifetime {
    fn clone(&self) -> Self {
        Lifetime {
            apostrophe: self.apostrophe,
            ident: self.ident.clone(),
        }
    }
}
impl PartialEq for Lifetime {
    fn eq(&self, other: &Lifetime) -> bool {
        self.ident.eq(&other.ident)
    }
}
impl Eq for Lifetime {}
impl PartialOrd for Lifetime {
    fn partial_cmp(&self, other: &Lifetime) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Lifetime {
    fn cmp(&self, other: &Lifetime) -> Ordering {
        self.ident.cmp(&other.ident)
    }
}
impl Hash for Lifetime {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.ident.hash(h);
    }
}
#[allow(non_snake_case)]
pub fn Lifetime(marker: lookahead::TokenMarker) -> Lifetime {
    match marker {}
}

mod parsing {
    use super::*;
    use crate::parse::{Parse, ParseStream, Result};
    impl Parse for Lifetime {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| cursor.lifetime().ok_or_else(|| cursor.error("expected lifetime")))
        }
    }
}
mod printing {
    use super::*;
    use proc_macro2::{Punct, Spacing, TokenStream};
    use quote::{ToTokens, TokenStreamExt};
    impl ToTokens for Lifetime {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let mut apostrophe = Punct::new('\'', Spacing::Joint);
            apostrophe.set_span(self.apostrophe);
            tokens.append(apostrophe);
            self.ident.to_tokens(tokens);
        }
    }
}

mod lookahead {
    use crate::buffer::Cursor;
    use crate::err::{self, Err};
    use crate::sealed::lookahead::Sealed;
    use crate::token::Token;
    use crate::IntoSpans;
    use proc_macro2::{Delimiter, Span};
    use std::cell::RefCell;
    pub struct Lookahead1<'a> {
        scope: Span,
        cursor: Cursor<'a>,
        comparisons: RefCell<Vec<&'static str>>,
    }
    impl<'a> Lookahead1<'a> {
        pub fn peek<T: Peek>(&self, token: T) -> bool {
            let _ = token;
            peek_impl(self, T::Token::peek, T::Token::display)
        }
        pub fn error(self) -> Err {
            let comparisons = self.comparisons.borrow();
            match comparisons.len() {
                0 => {
                    if self.cursor.eof() {
                        Err::new(self.scope, "unexpected end of input")
                    } else {
                        Err::new(self.cursor.span(), "unexpected token")
                    }
                },
                1 => {
                    let message = format!("expected {}", comparisons[0]);
                    err::new_at(self.scope, self.cursor, message)
                },
                2 => {
                    let message = format!("expected {} or {}", comparisons[0], comparisons[1]);
                    err::new_at(self.scope, self.cursor, message)
                },
                _ => {
                    let join = comparisons.join(", ");
                    let message = format!("expected one of: {}", join);
                    err::new_at(self.scope, self.cursor, message)
                },
            }
        }
    }

    pub fn new(scope: Span, cursor: Cursor) -> Lookahead1 {
        Lookahead1 {
            scope,
            cursor,
            comparisons: RefCell::new(Vec::new()),
        }
    }
    fn peek_impl(lookahead: &Lookahead1, peek: fn(Cursor) -> bool, display: fn() -> &'static str) -> bool {
        if peek(lookahead.cursor) {
            return true;
        }
        lookahead.comparisons.borrow_mut().push(display());
        false
    }

    pub trait Peek: Sealed {
        type Token: Token;
    }
    impl<F: Copy + FnOnce(TokenMarker) -> T, T: Token> Peek for F {
        type Token = T;
    }

    pub enum TokenMarker {}
    impl<S> IntoSpans<S> for TokenMarker {
        fn into_spans(self) -> S {
            match self {}
        }
    }

    pub fn is_delimiter(cursor: Cursor, delimiter: Delimiter) -> bool {
        cursor.group(delimiter).is_some()
    }

    impl<F: Copy + FnOnce(TokenMarker) -> T, T: Token> Sealed for F {}
}

ast_struct! {
    pub struct Macro {
        pub path: Path,
        pub bang_token: Token![!],
        pub delimiter: MacroDelimiter,
        pub tokens: TokenStream,
    }
}
ast_enum! {
    pub enum MacroDelimiter {
        Paren(Paren),
        Brace(Brace),
        Bracket(Bracket),
    }
}
impl MacroDelimiter {
    pub fn span(&self) -> &DelimSpan {
        match self {
            MacroDelimiter::Paren(token) => &token.span,
            MacroDelimiter::Brace(token) => &token.span,
            MacroDelimiter::Bracket(token) => &token.span,
        }
    }
}
impl Macro {
    pub fn parse_body<T: Parse>(&self) -> Result<T> {
        self.parse_body_with(T::parse)
    }

    pub fn parse_body_with<F: Parser>(&self, parser: F) -> Result<F::Output> {
        let scope = self.delimiter.span().close();
        crate::parse::parse_scoped(parser, scope, self.tokens.clone())
    }
}
fn mac_parse_delimiter(input: ParseStream) -> Result<(MacroDelimiter, TokenStream)> {
    input.step(|cursor| {
        if let Some((TokenTree::Group(g), rest)) = cursor.token_tree() {
            let span = g.delim_span();
            let delimiter = match g.delimiter() {
                Delimiter::Parenthesis => MacroDelimiter::Paren(Paren(span)),
                Delimiter::Brace => MacroDelimiter::Brace(Brace(span)),
                Delimiter::Bracket => MacroDelimiter::Bracket(Bracket(span)),
                Delimiter::None => {
                    return Err(cursor.error("expected delimiter"));
                },
            };
            Ok(((delimiter, g.stream()), rest))
        } else {
            Err(cursor.error("expected delimiter"))
        }
    })
}
mod parsing {
    use super::*;
    use crate::parse::{Parse, ParseStream, Result};

    impl Parse for Macro {
        fn parse(input: ParseStream) -> Result<Self> {
            let tokens;
            Ok(Macro {
                path: input.call(Path::parse_mod_style)?,
                bang_token: input.parse()?,
                delimiter: {
                    let (delimiter, content) = mac_parse_delimiter(input)?;
                    tokens = content;
                    delimiter
                },
                tokens,
            })
        }
    }
}
mod printing {
    use super::*;
    use proc_macro2::TokenStream;
    use quote::ToTokens;
    impl MacroDelimiter {
        pub fn surround(&self, tokens: &mut TokenStream, inner: TokenStream) {
            let (delim, span) = match self {
                MacroDelimiter::Paren(paren) => (Delimiter::Parenthesis, paren.span),
                MacroDelimiter::Brace(brace) => (Delimiter::Brace, brace.span),
                MacroDelimiter::Bracket(bracket) => (Delimiter::Bracket, bracket.span),
            };
            token::printing::delim(delim, span.join(), tokens, inner);
        }
    }
    impl ToTokens for Macro {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.path.to_tokens(tokens);
            self.bang_token.to_tokens(tokens);
            self.delimiter.surround(tokens, self.tokens.clone());
        }
    }
}

ast_enum! {
    #[non_exhaustive]
    pub enum BinOp {
        Add(Token![+]),
        Sub(Token![-]),
        Mul(Token![*]),
        Div(Token![/]),
        Rem(Token![%]),
        And(Token![&&]),
        Or(Token![||]),
        BitXor(Token![^]),
        BitAnd(Token![&]),
        BitOr(Token![|]),
        Shl(Token![<<]),
        Shr(Token![>>]),
        Eq(Token![==]),
        Lt(Token![<]),
        Le(Token![<=]),
        Ne(Token![!=]),
        Ge(Token![>=]),
        Gt(Token![>]),
        AddAssign(Token![+=]),
        SubAssign(Token![-=]),
        MulAssign(Token![*=]),
        DivAssign(Token![/=]),
        RemAssign(Token![%=]),
        BitXorAssign(Token![^=]),
        BitAndAssign(Token![&=]),
        BitOrAssign(Token![|=]),
        ShlAssign(Token![<<=]),
        ShrAssign(Token![>>=]),
    }
}
ast_enum! {
    #[non_exhaustive]
    pub enum UnOp {
        Deref(Token![*]),
        Not(Token![!]),
        Neg(Token![-]),
    }
}
mod parsing {
    use super::*;
    use crate::parse::{Parse, ParseStream, Result};
    fn parse_binop(input: ParseStream) -> Result<BinOp> {
        if input.peek(Token![&&]) {
            input.parse().map(BinOp::And)
        } else if input.peek(Token![||]) {
            input.parse().map(BinOp::Or)
        } else if input.peek(Token![<<]) {
            input.parse().map(BinOp::Shl)
        } else if input.peek(Token![>>]) {
            input.parse().map(BinOp::Shr)
        } else if input.peek(Token![==]) {
            input.parse().map(BinOp::Eq)
        } else if input.peek(Token![<=]) {
            input.parse().map(BinOp::Le)
        } else if input.peek(Token![!=]) {
            input.parse().map(BinOp::Ne)
        } else if input.peek(Token![>=]) {
            input.parse().map(BinOp::Ge)
        } else if input.peek(Token![+]) {
            input.parse().map(BinOp::Add)
        } else if input.peek(Token![-]) {
            input.parse().map(BinOp::Sub)
        } else if input.peek(Token![*]) {
            input.parse().map(BinOp::Mul)
        } else if input.peek(Token![/]) {
            input.parse().map(BinOp::Div)
        } else if input.peek(Token![%]) {
            input.parse().map(BinOp::Rem)
        } else if input.peek(Token![^]) {
            input.parse().map(BinOp::BitXor)
        } else if input.peek(Token![&]) {
            input.parse().map(BinOp::BitAnd)
        } else if input.peek(Token![|]) {
            input.parse().map(BinOp::BitOr)
        } else if input.peek(Token![<]) {
            input.parse().map(BinOp::Lt)
        } else if input.peek(Token![>]) {
            input.parse().map(BinOp::Gt)
        } else {
            Err(input.error("expected binary operator"))
        }
    }

    impl Parse for BinOp {
        #[cfg(not(feature = "full"))]
        fn parse(input: ParseStream) -> Result<Self> {
            parse_binop(input)
        }
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(Token![+=]) {
                input.parse().map(BinOp::AddAssign)
            } else if input.peek(Token![-=]) {
                input.parse().map(BinOp::SubAssign)
            } else if input.peek(Token![*=]) {
                input.parse().map(BinOp::MulAssign)
            } else if input.peek(Token![/=]) {
                input.parse().map(BinOp::DivAssign)
            } else if input.peek(Token![%=]) {
                input.parse().map(BinOp::RemAssign)
            } else if input.peek(Token![^=]) {
                input.parse().map(BinOp::BitXorAssign)
            } else if input.peek(Token![&=]) {
                input.parse().map(BinOp::BitAndAssign)
            } else if input.peek(Token![|=]) {
                input.parse().map(BinOp::BitOrAssign)
            } else if input.peek(Token![<<=]) {
                input.parse().map(BinOp::ShlAssign)
            } else if input.peek(Token![>>=]) {
                input.parse().map(BinOp::ShrAssign)
            } else {
                parse_binop(input)
            }
        }
    }

    impl Parse for UnOp {
        fn parse(input: ParseStream) -> Result<Self> {
            let lookahead = input.lookahead1();
            if lookahead.peek(Token![*]) {
                input.parse().map(UnOp::Deref)
            } else if lookahead.peek(Token![!]) {
                input.parse().map(UnOp::Not)
            } else if lookahead.peek(Token![-]) {
                input.parse().map(UnOp::Neg)
            } else {
                Err(lookahead.error())
            }
        }
    }
}
mod printing {
    use super::*;
    use proc_macro2::TokenStream;
    use quote::ToTokens;
    impl ToTokens for BinOp {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                BinOp::Add(t) => t.to_tokens(tokens),
                BinOp::Sub(t) => t.to_tokens(tokens),
                BinOp::Mul(t) => t.to_tokens(tokens),
                BinOp::Div(t) => t.to_tokens(tokens),
                BinOp::Rem(t) => t.to_tokens(tokens),
                BinOp::And(t) => t.to_tokens(tokens),
                BinOp::Or(t) => t.to_tokens(tokens),
                BinOp::BitXor(t) => t.to_tokens(tokens),
                BinOp::BitAnd(t) => t.to_tokens(tokens),
                BinOp::BitOr(t) => t.to_tokens(tokens),
                BinOp::Shl(t) => t.to_tokens(tokens),
                BinOp::Shr(t) => t.to_tokens(tokens),
                BinOp::Eq(t) => t.to_tokens(tokens),
                BinOp::Lt(t) => t.to_tokens(tokens),
                BinOp::Le(t) => t.to_tokens(tokens),
                BinOp::Ne(t) => t.to_tokens(tokens),
                BinOp::Ge(t) => t.to_tokens(tokens),
                BinOp::Gt(t) => t.to_tokens(tokens),
                BinOp::AddAssign(t) => t.to_tokens(tokens),
                BinOp::SubAssign(t) => t.to_tokens(tokens),
                BinOp::MulAssign(t) => t.to_tokens(tokens),
                BinOp::DivAssign(t) => t.to_tokens(tokens),
                BinOp::RemAssign(t) => t.to_tokens(tokens),
                BinOp::BitXorAssign(t) => t.to_tokens(tokens),
                BinOp::BitAndAssign(t) => t.to_tokens(tokens),
                BinOp::BitOrAssign(t) => t.to_tokens(tokens),
                BinOp::ShlAssign(t) => t.to_tokens(tokens),
                BinOp::ShrAssign(t) => t.to_tokens(tokens),
            }
        }
    }
    impl ToTokens for UnOp {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                UnOp::Deref(t) => t.to_tokens(tokens),
                UnOp::Not(t) => t.to_tokens(tokens),
                UnOp::Neg(t) => t.to_tokens(tokens),
            }
        }
    }
}

pub mod parse;
mod parse_macro_input {
    #[macro_export]
    macro_rules! parse_macro_input {
        ($tokenstream:ident as $ty:ty) => {
            match $crate::parse::<$ty>($tokenstream) {
                $crate::__private::Ok(data) => data,
                $crate::__private::Err(err) => {
                    return $crate::__private::TokenStream::from(err.to_compile_error());
                },
            }
        };
        ($tokenstream:ident with $parser:path) => {
            match $crate::parse::Parser::parse($parser, $tokenstream) {
                $crate::__private::Ok(data) => data,
                $crate::__private::Err(err) => {
                    return $crate::__private::TokenStream::from(err.to_compile_error());
                },
            }
        };
        ($tokenstream:ident) => {
            $crate::parse_macro_input!($tokenstream as _)
        };
    }
}
mod parse_quote {
    #[macro_export]
    macro_rules! parse_quote {
    ($($tt:tt)*) => {
        $crate::__private::parse_quote($crate::__private::quote::quote!($($tt)*))
    };
}
    #[macro_export]
    macro_rules! parse_quote_spanned {
    ($span:expr=> $($tt:tt)*) => {
        $crate::__private::parse_quote($crate::__private::quote::quote_spanned!($span=> $($tt)*))
    };
}
    use crate::parse::{Parse, ParseStream, Parser, Result};
    use proc_macro2::TokenStream;
    pub fn parse<T: ParseQuote>(token_stream: TokenStream) -> T {
        let parser = T::parse;
        match parser.parse2(token_stream) {
            Ok(t) => t,
            Err(err) => panic!("{}", err),
        }
    }
    pub trait ParseQuote: Sized {
        fn parse(input: ParseStream) -> Result<Self>;
    }
    impl<T: Parse> ParseQuote for T {
        fn parse(input: ParseStream) -> Result<Self> {
            <T as Parse>::parse(input)
        }
    }
    use crate::punctuated::Punctuated;

    use crate::{attr, Attribute};
    use crate::{Block, Pat, Stmt};

    impl ParseQuote for Attribute {
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(Token![#]) && input.peek2(Token![!]) {
                attr::parsing::single_parse_inner(input)
            } else {
                attr::parsing::single_parse_outer(input)
            }
        }
    }
    impl ParseQuote for Pat {
        fn parse(input: ParseStream) -> Result<Self> {
            Pat::parse_multi_with_leading_vert(input)
        }
    }
    impl ParseQuote for Box<Pat> {
        fn parse(input: ParseStream) -> Result<Self> {
            <Pat as ParseQuote>::parse(input).map(Box::new)
        }
    }
    impl<T: Parse, P: Parse> ParseQuote for Punctuated<T, P> {
        fn parse(input: ParseStream) -> Result<Self> {
            Self::parse_terminated(input)
        }
    }
    impl ParseQuote for Vec<Stmt> {
        fn parse(input: ParseStream) -> Result<Self> {
            Block::parse_within(input)
        }
    }
}

struct TokensOrDefault<'a, T: 'a>(pub &'a Option<T>);
impl<'a, T> ToTokens for TokensOrDefault<'a, T>
where
    T: ToTokens + Default,
{
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self.0 {
            Some(t) => t.to_tokens(tokens),
            None => T::default().to_tokens(tokens),
        }
    }
}

mod restriction {
    use super::*;
    ast_enum! {
        pub enum Visibility {
            Public(Token![pub]),
            Restricted(VisRestricted),
            Inherited,
        }
    }
    ast_struct! {
        pub struct VisRestricted {
            pub pub_token: Token![pub],
            pub paren_token: token::Paren,
            pub in_token: Option<Token![in]>,
            pub path: Box<Path>,
        }
    }
    ast_enum! {
        #[non_exhaustive]
        pub enum FieldMutability {
            None,
        }
    }
    mod parsing {
        use super::*;
        use crate::ext::IdentExt;
        use crate::parse::discouraged::Speculative;
        use crate::parse::{Parse, ParseStream, Result};

        impl Parse for Visibility {
            fn parse(input: ParseStream) -> Result<Self> {
                if input.peek(token::Group) {
                    let ahead = input.fork();
                    let group = crate::group::parse_group(&ahead)?;
                    if group.content.is_empty() {
                        input.advance_to(&ahead);
                        return Ok(Visibility::Inherited);
                    }
                }
                if input.peek(Token![pub]) {
                    Self::parse_pub(input)
                } else {
                    Ok(Visibility::Inherited)
                }
            }
        }
        impl Visibility {
            fn parse_pub(input: ParseStream) -> Result<Self> {
                let pub_token = input.parse::<Token![pub]>()?;
                if input.peek(token::Paren) {
                    let ahead = input.fork();
                    let content;
                    let paren_token = parenthesized!(content in ahead);
                    if content.peek(Token![crate]) || content.peek(Token![self]) || content.peek(Token![super]) {
                        let path = content.call(Ident::parse_any)?;
                        if content.is_empty() {
                            input.advance_to(&ahead);
                            return Ok(Visibility::Restricted(VisRestricted {
                                pub_token,
                                paren_token,
                                in_token: None,
                                path: Box::new(Path::from(path)),
                            }));
                        }
                    } else if content.peek(Token![in]) {
                        let in_token: Token![in] = content.parse()?;
                        let path = content.call(Path::parse_mod_style)?;
                        input.advance_to(&ahead);
                        return Ok(Visibility::Restricted(VisRestricted {
                            pub_token,
                            paren_token,
                            in_token: Some(in_token),
                            path: Box::new(path),
                        }));
                    }
                }
                Ok(Visibility::Public(pub_token))
            }
            pub fn is_some(&self) -> bool {
                match self {
                    Visibility::Inherited => false,
                    _ => true,
                }
            }
        }
    }
    mod printing {
        use super::*;
        use proc_macro2::TokenStream;
        use quote::ToTokens;
        impl ToTokens for Visibility {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                match self {
                    Visibility::Public(pub_token) => pub_token.to_tokens(tokens),
                    Visibility::Restricted(vis_restricted) => vis_restricted.to_tokens(tokens),
                    Visibility::Inherited => {},
                }
            }
        }
        impl ToTokens for VisRestricted {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                self.pub_token.to_tokens(tokens);
                self.paren_token.surround(tokens, |tokens| {
                    self.in_token.to_tokens(tokens);
                    self.path.to_tokens(tokens);
                });
            }
        }
    }
}
pub use crate::restriction::{FieldMutability, VisRestricted, Visibility};
mod sealed {
    pub mod lookahead {
        pub trait Sealed: Copy {}
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
        let mut group = Group::new(Delimiter::None, TokenStream::new());
        group.set_span(self);
        group.delim_span()
    }
}
impl IntoSpans<DelimSpan> for DelimSpan {
    fn into_spans(self) -> DelimSpan {
        self
    }
}

pub trait Spanned: private::Sealed {
    fn span(&self) -> Span;
}
impl<T: ?Sized + spanned::Spanned> Spanned for T {
    fn span(&self) -> Span {
        self.__span()
    }
}
mod private {
    use super::*;
    pub trait Sealed {}
    impl<T: ?Sized + spanned::Spanned> Sealed for T {}
}

struct ThreadBound<T> {
    value: T,
    thread_id: ThreadId,
}
unsafe impl<T> Sync for ThreadBound<T> {}
unsafe impl<T: Copy> Send for ThreadBound<T> {}
impl<T> ThreadBound<T> {
    pub fn new(value: T) -> Self {
        ThreadBound {
            value,
            thread_id: thread::current().id(),
        }
    }
    pub fn get(&self) -> Option<&T> {
        if thread::current().id() == self.thread_id {
            Some(&self.value)
        } else {
            None
        }
    }
}
impl<T: Debug> Debug for ThreadBound<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self.get() {
            Some(value) => Debug::fmt(value, formatter),
            None => formatter.write_str("unknown"),
        }
    }
}
impl<T: Copy> Copy for ThreadBound<T> {}
impl<T: Copy> Clone for ThreadBound<T> {
    fn clone(&self) -> Self {
        *self
    }
}

struct TokenTreeHelper<'a>(pub &'a TokenTree);
impl<'a> PartialEq for TokenTreeHelper<'a> {
    fn eq(&self, other: &Self) -> bool {
        use proc_macro2::Spacing;
        match (self.0, other.0) {
            (TokenTree::Group(g1), TokenTree::Group(g2)) => {
                match (g1.delimiter(), g2.delimiter()) {
                    (Delimiter::Parenthesis, Delimiter::Parenthesis)
                    | (Delimiter::Brace, Delimiter::Brace)
                    | (Delimiter::Bracket, Delimiter::Bracket)
                    | (Delimiter::None, Delimiter::None) => {},
                    _ => return false,
                }
                let s1 = g1.stream().into_iter();
                let mut s2 = g2.stream().into_iter();
                for item1 in s1 {
                    let item2 = match s2.next() {
                        Some(item) => item,
                        None => return false,
                    };
                    if TokenTreeHelper(&item1) != TokenTreeHelper(&item2) {
                        return false;
                    }
                }
                s2.next().is_none()
            },
            (TokenTree::Punct(o1), TokenTree::Punct(o2)) => {
                o1.as_char() == o2.as_char()
                    && match (o1.spacing(), o2.spacing()) {
                        (Spacing::Alone, Spacing::Alone) | (Spacing::Joint, Spacing::Joint) => true,
                        _ => false,
                    }
            },
            (TokenTree::Literal(l1), TokenTree::Literal(l2)) => l1.to_string() == l2.to_string(),
            (TokenTree::Ident(s1), TokenTree::Ident(s2)) => s1 == s2,
            _ => false,
        }
    }
}
impl<'a> Hash for TokenTreeHelper<'a> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        use proc_macro2::Spacing;
        match self.0 {
            TokenTree::Group(g) => {
                0u8.hash(h);
                match g.delimiter() {
                    Delimiter::Parenthesis => 0u8.hash(h),
                    Delimiter::Brace => 1u8.hash(h),
                    Delimiter::Bracket => 2u8.hash(h),
                    Delimiter::None => 3u8.hash(h),
                }
                for item in g.stream() {
                    TokenTreeHelper(&item).hash(h);
                }
                0xffu8.hash(h); // terminator w/ a variant we don't normally hash
            },
            TokenTree::Punct(op) => {
                1u8.hash(h);
                op.as_char().hash(h);
                match op.spacing() {
                    Spacing::Alone => 0u8.hash(h),
                    Spacing::Joint => 1u8.hash(h),
                }
            },
            TokenTree::Literal(lit) => (2u8, lit.to_string()).hash(h),
            TokenTree::Ident(word) => (3u8, word).hash(h),
        }
    }
}

struct TokenStreamHelper<'a>(pub &'a TokenStream);
impl<'a> PartialEq for TokenStreamHelper<'a> {
    fn eq(&self, other: &Self) -> bool {
        let left = self.0.clone().into_iter().collect::<Vec<_>>();
        let right = other.0.clone().into_iter().collect::<Vec<_>>();
        if left.len() != right.len() {
            return false;
        }
        for (a, b) in left.into_iter().zip(right) {
            if TokenTreeHelper(&a) != TokenTreeHelper(&b) {
                return false;
            }
        }
        true
    }
}
impl<'a> Hash for TokenStreamHelper<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let tts = self.0.clone().into_iter().collect::<Vec<_>>();
        tts.len().hash(state);
        for tt in tts {
            TokenTreeHelper(&tt).hash(state);
        }
    }
}

pub fn verbatim_between<'a>(begin: ParseStream<'a>, end: ParseStream<'a>) -> TokenStream {
    let end = end.cursor();
    let mut cursor = begin.cursor();
    assert!(crate::buffer::same_buffer(end, cursor));
    let mut tokens = TokenStream::new();
    while cursor != end {
        let (tt, next) = cursor.token_tree().unwrap();
        if crate::buffer::cmp_assuming_same_buffer(end, next) == Ordering::Less {
            if let Some((inside, _span, after)) = cursor.group(Delimiter::None) {
                assert!(next == after);
                cursor = inside;
                continue;
            } else {
                panic!("verbatim end must not be inside a delimited group");
            }
        }
        tokens.extend(iter::once(tt));
        cursor = next;
    }
    tokens
}

pub fn ws_skip(mut s: &str) -> &str {
    'skip: while !s.is_empty() {
        let byte = s.as_bytes()[0];
        if byte == b'/' {
            if s.starts_with("//") && (!s.starts_with("///") || s.starts_with("////")) && !s.starts_with("//!") {
                if let Some(i) = s.find('\n') {
                    s = &s[i + 1..];
                    continue;
                } else {
                    return "";
                }
            } else if s.starts_with("/**/") {
                s = &s[4..];
                continue;
            } else if s.starts_with("/*") && (!s.starts_with("/**") || s.starts_with("/***")) && !s.starts_with("/*!") {
                let mut depth = 0;
                let bytes = s.as_bytes();
                let mut i = 0;
                let upper = bytes.len() - 1;
                while i < upper {
                    if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                        depth += 1;
                        i += 1; // eat '*'
                    } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        depth -= 1;
                        if depth == 0 {
                            s = &s[i + 2..];
                            continue 'skip;
                        }
                        i += 1; // eat '/'
                    }
                    i += 1;
                }
                return s;
            }
        }
        match byte {
            b' ' | 0x09..=0x0d => {
                s = &s[1..];
                continue;
            },
            b if b <= 0x7f => {},
            _ => {
                let ch = s.chars().next().unwrap();
                if is_whitespace(ch) {
                    s = &s[ch.len_utf8()..];
                    continue;
                }
            },
        }
        return s;
    }
    s
}
fn is_whitespace(x: char) -> bool {
    x.is_whitespace() || x == '\u{200e}' || x == '\u{200f}'
}

mod gen {
    #[rustfmt::skip]
    pub mod fold;
    #[rustfmt::skip]
    pub mod visit;
    #[rustfmt::skip]
    pub mod visit_mut;
        #[rustfmt::skip]
    mod clone;
        #[rustfmt::skip]
    mod debug;
        #[rustfmt::skip]
    mod eq;
        #[rustfmt::skip]
    mod hash;
    mod helper {
        pub mod fold {
            use crate::punctuated::{Pair, Punctuated};
            pub trait FoldHelper {
                type Item;
                fn lift<F>(self, f: F) -> Self
                where
                    F: FnMut(Self::Item) -> Self::Item;
            }
            impl<T> FoldHelper for Vec<T> {
                type Item = T;
                fn lift<F>(self, f: F) -> Self
                where
                    F: FnMut(Self::Item) -> Self::Item,
                {
                    self.into_iter().map(f).collect()
                }
            }
            impl<T, U> FoldHelper for Punctuated<T, U> {
                type Item = T;
                fn lift<F>(self, mut f: F) -> Self
                where
                    F: FnMut(Self::Item) -> Self::Item,
                {
                    self.into_pairs()
                        .map(Pair::into_tuple)
                        .map(|(t, u)| Pair::new(f(t), u))
                        .collect()
                }
            }
        }
    }
}
pub use crate::gen::*;
pub mod __private {
    pub use crate::parse_quote::parse as parse_quote;
    pub use crate::token::parsing::{peek_punct, punct as parse_punct};
    pub use crate::token::printing::punct as print_punct;
    pub use proc_macro::TokenStream;
    pub use proc_macro2::{Span, TokenStream as TokenStream2};
    pub use quote;
    pub use quote::{ToTokens, TokenStreamExt};
    pub use std::clone::Clone;
    pub use std::cmp::{Eq, PartialEq};
    pub use std::concat;
    pub use std::default::Default;
    pub use std::fmt::{self, Debug, Formatter};
    pub use std::hash::{Hash, Hasher};
    pub use std::marker::Copy;
    pub use std::option::Option::{None, Some};
    pub use std::result::Result::{Err, Ok};
    pub use std::stringify;
    pub type bool = help::Bool;
    pub type str = help::Str;
    mod help {
        pub type Bool = bool;
        pub type Str = str;
    }
    pub struct private(pub(crate) ());
}
pub fn parse<T: parse::Parse>(tokens: proc_macro::TokenStream) -> Result<T> {
    parse::Parser::parse(T::parse, tokens)
}
pub fn parse2<T: parse::Parse>(tokens: proc_macro2::TokenStream) -> Result<T> {
    parse::Parser::parse2(T::parse, tokens)
}
pub fn parse_str<T: parse::Parse>(s: &str) -> Result<T> {
    parse::Parser::parse_str(T::parse, s)
}
pub fn parse_file(mut content: &str) -> Result<File> {
    const BOM: &str = "\u{feff}";
    if content.starts_with(BOM) {
        content = &content[BOM.len()..];
    }
    let mut shebang = None;
    if content.starts_with("#!") {
        let rest = whitespace::ws_skip(&content[2..]);
        if !rest.starts_with('[') {
            if let Some(idx) = content.find('\n') {
                shebang = Some(content[..idx].to_string());
                content = &content[idx..];
            } else {
                shebang = Some(content.to_string());
                content = "";
            }
        }
    }
    let mut file: File = parse_str(content)?;
    file.shebang = shebang;
    Ok(file)
}
