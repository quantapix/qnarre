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
extern crate proc_macro;
#[macro_use]
mod macros;
#[macro_use]
mod group;
#[macro_use]
pub mod token;

mod attr;

pub use crate::attr::{AttrStyle, Attribute, Meta, MetaList, MetaNameValue};
mod bigint {
    use std::ops::{AddAssign, MulAssign};
    pub(crate) struct BigInt {
        digits: Vec<u8>,
    }
    impl BigInt {
        pub(crate) fn new() -> Self {
            BigInt { digits: Vec::new() }
        }
        pub(crate) fn to_string(&self) -> String {
            let mut repr = String::with_capacity(self.digits.len());
            let mut has_nonzero = false;
            for digit in self.digits.iter().rev() {
                has_nonzero |= *digit != 0;
                if has_nonzero {
                    repr.push((*digit + b'0') as char);
                }
            }
            if repr.is_empty() {
                repr.push('0');
            }
            repr
        }
        fn reserve_two_digits(&mut self) {
            let len = self.digits.len();
            let desired = len + !self.digits.ends_with(&[0, 0]) as usize + !self.digits.ends_with(&[0]) as usize;
            self.digits.resize(desired, 0);
        }
    }
    impl AddAssign<u8> for BigInt {
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
    impl MulAssign<u8> for BigInt {
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
}

pub mod buffer;
mod custom_keyword;
mod custom_punctuation;

mod data;

pub use crate::data::{Field, Fields, FieldsNamed, FieldsUnnamed, Variant};

mod derive;
pub use crate::derive::{Data, DataEnum, DataStruct, DataUnion, DeriveInput};
mod drops {
    use std::iter;
    use std::mem::ManuallyDrop;
    use std::ops::{Deref, DerefMut};
    use std::option;
    use std::slice;
    #[repr(transparent)]
    pub(crate) struct NoDrop<T: ?Sized>(ManuallyDrop<T>);
    impl<T> NoDrop<T> {
        pub(crate) fn new(value: T) -> Self
        where
            T: TrivialDrop,
        {
            NoDrop(ManuallyDrop::new(value))
        }
    }
    impl<T: ?Sized> Deref for NoDrop<T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T: ?Sized> DerefMut for NoDrop<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }
    pub(crate) trait TrivialDrop {}
    impl<T> TrivialDrop for iter::Empty<T> {}
    impl<'a, T> TrivialDrop for slice::Iter<'a, T> {}
    impl<'a, T> TrivialDrop for slice::IterMut<'a, T> {}
    impl<'a, T> TrivialDrop for option::IntoIter<&'a T> {}
    impl<'a, T> TrivialDrop for option::IntoIter<&'a mut T> {}
    #[test]
    fn test_needs_drop() {
        use std::mem::needs_drop;
        struct NeedsDrop;
        impl Drop for NeedsDrop {
            fn drop(&mut self) {}
        }
        assert!(needs_drop::<NeedsDrop>());
        assert!(!needs_drop::<iter::Empty<NeedsDrop>>());
        assert!(!needs_drop::<slice::Iter<NeedsDrop>>());
        assert!(!needs_drop::<slice::IterMut<NeedsDrop>>());
        assert!(!needs_drop::<option::IntoIter<&NeedsDrop>>());
        assert!(!needs_drop::<option::IntoIter<&mut NeedsDrop>>());
    }
}
mod error;
pub use crate::error::{Error, Result};

mod expr;
pub use crate::expr::{Arm, FieldValue, Label, RangeLimits};

pub use crate::expr::{
    Expr, ExprArray, ExprAssign, ExprAsync, ExprAwait, ExprBinary, ExprBlock, ExprBreak, ExprCall, ExprCast,
    ExprClosure, ExprConst, ExprContinue, ExprField, ExprForLoop, ExprGroup, ExprIf, ExprIndex, ExprInfer, ExprLet,
    ExprLit, ExprLoop, ExprMacro, ExprMatch, ExprMethodCall, ExprParen, ExprPath, ExprRange, ExprReference, ExprRepeat,
    ExprReturn, ExprStruct, ExprTry, ExprTryBlock, ExprTuple, ExprUnary, ExprUnsafe, ExprWhile, ExprYield, Index,
    Member,
};

pub mod ext;
mod file;
pub use crate::file::File;

mod generics;

pub use crate::generics::{
    BoundLifetimes, ConstParam, GenericParam, Generics, LifetimeParam, PredicateLifetime, PredicateType, TraitBound,
    TraitBoundModifier, TypeParam, TypeParamBound, WhereClause, WherePredicate,
};
pub use crate::generics::{ImplGenerics, Turbofish, TypeGenerics};
mod ident;
pub use crate::ident::Ident;
mod item;
pub use crate::item::{
    FnArg, ForeignItem, ForeignItemFn, ForeignItemMacro, ForeignItemStatic, ForeignItemType, ImplItem, ImplItemConst,
    ImplItemFn, ImplItemMacro, ImplItemType, ImplRestriction, Item, ItemConst, ItemEnum, ItemExternCrate, ItemFn,
    ItemForeignMod, ItemImpl, ItemMacro, ItemMod, ItemStatic, ItemStruct, ItemTrait, ItemTraitAlias, ItemType,
    ItemUnion, ItemUse, Receiver, Signature, StaticMutability, TraitItem, TraitItemConst, TraitItemFn, TraitItemMacro,
    TraitItemType, UseGlob, UseGroup, UseName, UsePath, UseRename, UseTree, Variadic,
};
mod lifetime;
pub use crate::lifetime::Lifetime;
mod lit;
pub use crate::lit::{Lit, LitBool, LitByte, LitByteStr, LitChar, LitFloat, LitInt, LitStr, StrStyle};
mod lookahead;

mod mac;

pub use crate::mac::{Macro, MacroDelimiter};
pub mod meta;

mod op;

pub use crate::op::{BinOp, UnOp};

pub mod parse;
mod parse_macro_input;
mod parse_quote;
mod pat;
pub use crate::expr::{
    ExprConst as PatConst, ExprLit as PatLit, ExprMacro as PatMacro, ExprPath as PatPath, ExprRange as PatRange,
};
pub use crate::pat::{
    FieldPat, Pat, PatIdent, PatOr, PatParen, PatReference, PatRest, PatSlice, PatStruct, PatTuple, PatTupleStruct,
    PatType, PatWild,
};

mod path;

pub use crate::path::{
    AngleBracketedGenericArguments, AssocConst, AssocType, Constraint, GenericArgument, ParenthesizedGenericArguments,
    Path, PathArguments, PathSegment, QSelf,
};
mod print {
    use proc_macro2::TokenStream;
    use quote::ToTokens;
    pub(crate) struct TokensOrDefault<'a, T: 'a>(pub &'a Option<T>);
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
}
pub mod punctuated;

mod restriction;

pub use crate::restriction::{FieldMutability, VisRestricted, Visibility};
mod sealed {
    pub(crate) mod lookahead {
        pub trait Sealed: Copy {}
    }
}
mod span {
    use proc_macro2::extra::DelimSpan;
    use proc_macro2::{Delimiter, Group, Span, TokenStream};
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
}
pub mod spanned;
mod stmt;
pub use crate::stmt::{Block, Local, LocalInit, Stmt, StmtMacro};
mod thread {
    use std::fmt::{self, Debug};
    use std::thread::{self, ThreadId};
    pub(crate) struct ThreadBound<T> {
        value: T,
        thread_id: ThreadId,
    }
    unsafe impl<T> Sync for ThreadBound<T> {}
    unsafe impl<T: Copy> Send for ThreadBound<T> {}
    impl<T> ThreadBound<T> {
        pub(crate) fn new(value: T) -> Self {
            ThreadBound {
                value,
                thread_id: thread::current().id(),
            }
        }
        pub(crate) fn get(&self) -> Option<&T> {
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
}
mod tt {
    use proc_macro2::{Delimiter, TokenStream, TokenTree};
    use std::hash::{Hash, Hasher};
    pub(crate) struct TokenTreeHelper<'a>(pub &'a TokenTree);
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
    pub(crate) struct TokenStreamHelper<'a>(pub &'a TokenStream);
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
}

mod ty;

pub use crate::ty::{
    Abi, BareFnArg, BareVariadic, ReturnType, Type, TypeArray, TypeBareFn, TypeGroup, TypeImplTrait, TypeInfer,
    TypeMacro, TypeNever, TypeParen, TypePath, TypePtr, TypeReference, TypeSlice, TypeTraitObject, TypeTuple,
};
mod verbatim {
    use crate::parse::ParseStream;
    use proc_macro2::{Delimiter, TokenStream};
    use std::cmp::Ordering;
    use std::iter;
    pub(crate) fn between<'a>(begin: ParseStream<'a>, end: ParseStream<'a>) -> TokenStream {
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
}
mod whitespace {
    pub(crate) fn skip(mut s: &str) -> &str {
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
                } else if s.starts_with("/*")
                    && (!s.starts_with("/**") || s.starts_with("/***"))
                    && !s.starts_with("/*!")
                {
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
    fn is_whitespace(ch: char) -> bool {
        ch.is_whitespace() || ch == '\u{200e}' || ch == '\u{200f}'
    }
}
mod gen {
    #[cfg(feature = "fold")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "fold")))]
    #[rustfmt::skip]
    pub mod fold;
    #[cfg(feature = "visit")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "visit")))]
    #[rustfmt::skip]
    pub mod visit;
    #[cfg(feature = "visit-mut")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "visit-mut")))]
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
        #[cfg(feature = "fold")]
        pub(crate) mod fold {
            use crate::punctuated::{Pair, Punctuated};
            pub(crate) trait FoldHelper {
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
    pub use crate::group::{parse_braces, parse_brackets, parse_parens};
    pub use crate::parse_quote::parse as parse_quote;
    pub use crate::span::IntoSpans;
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
    #[allow(non_camel_case_types)]
    pub type bool = help::Bool;
    #[allow(non_camel_case_types)]
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
        let rest = whitespace::skip(&content[2..]);
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
