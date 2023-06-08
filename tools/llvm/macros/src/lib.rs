use convert_case::{Case, Casing};
use once_cell::sync::Lazy;
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;
use regex::{Captures, Regex};
use std::error::Error;
use syn::{
    bracketed,
    fold::Fold,
    parse::{self, Parse, ParseStream},
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    spanned::Spanned,
    Attribute, Field, LitFloat, Token, Variant,
};

struct IdentList {
    idents: Vec<Ident>,
}
impl IdentList {
    pub fn identifiers(&self) -> &[Ident] {
        &self.idents
    }
}
impl Parse for IdentList {
    fn parse(xs: ParseStream) -> parse::Result<Self> {
        Ok(Self {
            idents: Punctuated::<Ident, Token![,]>::parse_terminated(xs)?
                .into_iter()
                .collect(),
        })
    }
}

struct DialectOpSet {
    dialect: Ident,
    idents: IdentList,
}
impl DialectOpSet {
    pub fn dialect(&self) -> &Ident {
        &self.dialect
    }
    pub fn identifiers(&self) -> &[Ident] {
        self.idents.identifiers()
    }
}
impl Parse for DialectOpSet {
    fn parse(xs: ParseStream) -> parse::Result<Self> {
        let dialect = Ident::parse(xs)?;
        <Token![,]>::parse(xs)?;
        Ok(Self {
            dialect,
            idents: {
                let x;
                bracketed!(x in xs);
                x.parse::<IdentList>()?
            },
        })
    }
}

#[proc_macro]
pub fn unary_operations(x: TokenStream) -> TokenStream {
    let y = parse_macro_input!(x as DialectOpSet);
    convert_result(gen_unary(y.dialect(), y.identifiers()))
}
fn convert_result(x: Result<TokenStream, Box<dyn Error>>) -> TokenStream {
    x.unwrap_or_else(|y| quote! { compile_error!(#y.to_string()) }.into())
}
fn gen_unary(dialect: &Ident, xs: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut ys = TokenStream::new();
    for x in xs {
        let doc = create_doc(dialect, x);
        let op = create_op_name(dialect, x);
        ys.extend(TokenStream::from(quote! {
            #[doc = #doc]
            pub fn #x<'c>(
                val: crate::ir::Value,
                loc: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                unary_operator(#op, val, loc)
            }
        }));
    }
    ys.extend(TokenStream::from(quote! {
        fn unary_operator<'c>(
            name: &str,
            val: crate::ir::Value,
            loc: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::OperationBuilder::new(name, loc)
                .add_operands(&[val])
                .enable_result_type_inference()
                .build()
        }
    }));
    Ok(ys)
}
fn create_doc(dialect: &Ident, x: &Ident) -> String {
    format!(" Creates an `{}` operation.", create_op_name(dialect, x))
}
fn create_op_name(dialect: &Ident, x: &Ident) -> String {
    format!("{}.{}", dialect, x)
}

#[proc_macro]
pub fn typed_unary_operations(x: TokenStream) -> TokenStream {
    let y = parse_macro_input!(x as DialectOpSet);
    convert_result(gen_typed_unary(y.dialect(), y.identifiers()))
}
fn gen_typed_unary(dialect: &Ident, xs: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut ys = TokenStream::new();
    for x in xs {
        let doc = create_doc(dialect, x);
        let op = create_op_name(dialect, x);
        ys.extend(TokenStream::from(quote! {
            #[doc = #doc]
            pub fn #x<'c>(
                val: crate::ir::Value,
                ty: crate::ir::Type<'c>,
                loc: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                typed_unary_operator(#op, val, ty, loc)
            }
        }));
    }
    ys.extend(TokenStream::from(quote! {
        fn typed_unary_operator<'c>(
            name: &str,
            val: crate::ir::Value,
            ty: crate::ir::Type<'c>,
            loc: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::OperationBuilder::new(name, loc)
                .add_operands(&[val])
                .add_results(&[ty])
                .build()
        }
    }));
    Ok(ys)
}

#[proc_macro]
pub fn binary_operations(x: TokenStream) -> TokenStream {
    let y = parse_macro_input!(x as DialectOpSet);
    convert_result(gen_binary(y.dialect(), y.identifiers()))
}
fn gen_binary(dialect: &Ident, xs: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut ys = TokenStream::new();
    for x in xs {
        let doc = create_doc(dialect, x);
        let op = create_op_name(dialect, x);
        ys.extend(TokenStream::from(quote! {
            #[doc = #doc]
            pub fn #x<'c>(
                lhs: crate::ir::Value,
                rhs: crate::ir::Value,
                loc: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                binary_operator(#op, lhs, rhs, loc)
            }
        }));
    }
    ys.extend(TokenStream::from(quote! {
        fn binary_operator<'c>(
            name: &str,
            lhs: crate::ir::Value,
            rhs: crate::ir::Value,
            loc: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::OperationBuilder::new(name, loc)
                .add_operands(&[lhs, rhs])
                .enable_result_type_inference()
                .build()
        }
    }));
    Ok(ys)
}

#[proc_macro]
pub fn type_check_functions(x: TokenStream) -> TokenStream {
    let y = parse_macro_input!(x as IdentList);
    convert_result(gen_type(y.identifiers()))
}
fn gen_type(xs: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut ys = TokenStream::new();
    for x in xs {
        let name = map_name(&x.to_string().strip_prefix("mlirTypeIsA").unwrap().to_case(Case::Snake));
        let func = Ident::new(&format!("is_{}", &name), x.span());
        let doc = format!(" Returns `true` if a type is {}.", name.replace('_', " "));
        ys.extend(TokenStream::from(quote! {
            #[doc = #doc]
            fn #func(&self) -> bool {
                unsafe { mlir_lib::#x(self.to_raw()) }
            }
        }));
    }
    Ok(ys)
}
static PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(bf_16|f_16|f_32|f_64|i_8|i_16|i_32|i_64|float_8_e_[0-9]_m_[0-9](_fn)?)"#).unwrap());
fn map_name(name: &str) -> String {
    PATTERN
        .replace_all(name, |x: &Captures| x.get(0).unwrap().as_str().replace('_', ""))
        .to_string()
}

#[proc_macro]
pub fn attribute_check_functions(x: TokenStream) -> TokenStream {
    let y = parse_macro_input!(x as IdentList);
    convert_result(gen_attr(y.identifiers()))
}
fn gen_attr(xs: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut ys = TokenStream::new();
    for x in xs {
        let name = map_name(
            &x.to_string()
                .strip_prefix("mlirAttributeIsA")
                .unwrap()
                .to_case(Case::Snake),
        );
        let func = Ident::new(&format!("is_{}", &name), x.span());
        let doc = format!(" Returns `true` if an attribute is {}.", name.replace('_', " "));
        ys.extend(TokenStream::from(quote! {
            #[doc = #doc]
            fn #func(&self) -> bool {
                unsafe { mlir_lib::#x(self.to_raw()) }
            }
        }));
    }
    Ok(ys)
}

#[proc_macro]
pub fn async_passes(x: TokenStream) -> TokenStream {
    let y = parse_macro_input!(x as IdentList);
    convert_result(gen_pass(y.identifiers(), |x| x.strip_prefix("Async").unwrap().into()))
}
const CREATE_PRE: &str = "mlirCreate";
fn gen_pass(xs: &[Ident], extract: impl Fn(&str) -> String) -> Result<TokenStream, Box<dyn Error>> {
    let mut ys = TokenStream::new();
    for x in xs {
        let n = x.to_string();
        let n = n.strip_prefix(CREATE_PRE).unwrap();
        let pass = extract(n);
        let func = create_fn_name("create", &pass, x.span());
        let doc = format!(" Creates a `{}` pass.", pass);
        ys.extend(TokenStream::from(quote! {
            #[doc = #doc]
            pub fn #func() -> crate::pass::Pass {
                unsafe { crate::pass::Pass::__private_from_raw_fn(mlir_lib::#x) }
            }
        }));
        let foreign_function_name = Ident::new(&("mlirRegister".to_owned() + n), x.span());
        let func = create_fn_name("register", &pass, x.span());
        let doc = format!(" Registers a `{}` pass.", pass);
        ys.extend(TokenStream::from(quote! {
            #[doc = #doc]
            pub fn #func() {
                unsafe { mlir_lib::#foreign_function_name() }
            }
        }));
    }
    Ok(ys)
}
fn create_fn_name(pre: &str, pass: &str, span: Span) -> Ident {
    Ident::new(&format!("{}_{}", pre, &pass.to_case(Case::Snake)), span)
}

#[proc_macro]
pub fn conversion_passes(x: TokenStream) -> TokenStream {
    let y = parse_macro_input!(x as IdentList);
    convert_result(gen_pass(y.identifiers(), |mut x| {
        x = x.strip_prefix("Conversion").unwrap();
        x = x.strip_prefix("Convert").unwrap_or(x);
        x.strip_suffix("ConversionPass").unwrap_or(x).into()
    }))
}
#[proc_macro]
pub fn gpu_passes(x: TokenStream) -> TokenStream {
    let y = parse_macro_input!(x as IdentList);
    convert_result(gen_pass(y.identifiers(), |x| x.strip_prefix("GPU").unwrap().into()))
}
#[proc_macro]
pub fn transform_passes(x: TokenStream) -> TokenStream {
    let y = parse_macro_input!(x as IdentList);
    convert_result(gen_pass(y.identifiers(), |x| {
        x.strip_prefix("Transforms").unwrap().into()
    }))
}
#[proc_macro]
pub fn linalg_passes(x: TokenStream) -> TokenStream {
    let y = parse_macro_input!(x as IdentList);
    convert_result(gen_pass(y.identifiers(), |x| x.strip_prefix("Linalg").unwrap().into()))
}
#[proc_macro]
pub fn sparse_tensor_passes(x: TokenStream) -> TokenStream {
    let y = parse_macro_input!(x as IdentList);
    convert_result(gen_pass(y.identifiers(), |x| {
        x.strip_prefix("SparseTensor").unwrap().into()
    }))
}

#[derive(Debug)]
enum VersionType {
    Specific(f64, Span),
    Inclusive((f64, Span), (f64, Span)),
    InclusiveToLatest(f64, Span),
    Exclusive((f64, Span), (f64, Span)),
    ExclusiveToLatest(f64, Span),
}
impl Parse for VersionType {
    fn parse(xs: ParseStream) -> parse::Result<Self> {
        let y = xs.lookahead1();
        if y.peek(LitFloat) {
            let from = xs.parse::<LitFloat>().unwrap();
            let from_val = from.base10_parse().unwrap();
            if xs.is_empty() {
                return Ok(VersionType::Specific(from_val, from.span()));
            }
            let y = xs.lookahead1();
            if y.peek(Token![..=]) {
                let _: Token![..=] = xs.parse().unwrap();
                let y = xs.lookahead1();
                if y.peek(syn::Ident) {
                    let to = xs.parse::<Ident>().unwrap();
                    if to == "latest" {
                        Ok(VersionType::InclusiveToLatest(from_val, from.span()))
                    } else {
                        Err(parse::Error::new(to.span(), "expected `latest` or `X.Y`"))
                    }
                } else if y.peek(LitFloat) {
                    let to = xs.parse::<LitFloat>().unwrap();
                    let to_val = to.base10_parse().unwrap();
                    Ok(VersionType::Inclusive((from_val, from.span()), (to_val, to.span())))
                } else {
                    Err(y.error())
                }
            } else if y.peek(Token![..]) {
                let _: Token![..] = xs.parse().unwrap();
                let y = xs.lookahead1();
                if y.peek(syn::Ident) {
                    let to = xs.parse::<Ident>().unwrap();
                    if to == "latest" {
                        Ok(VersionType::ExclusiveToLatest(from_val, from.span()))
                    } else {
                        Err(parse::Error::new(to.span(), "expected `latest` or `X.Y`"))
                    }
                } else if y.peek(LitFloat) {
                    let to = xs.parse::<LitFloat>().unwrap();
                    let to_val = to.base10_parse().unwrap();
                    Ok(VersionType::Exclusive((from_val, from.span()), (to_val, to.span())))
                } else {
                    Err(y.error())
                }
            } else {
                Err(y.error())
            }
        } else {
            Err(y.error())
        }
    }
}

#[derive(Debug)]
struct ParenthesizedFeatureSet(FeatureSet);
impl Parse for ParenthesizedFeatureSet {
    fn parse(xs: ParseStream) -> parse::Result<Self> {
        xs.parse::<FeatureSet>().map(Self)
    }
}

#[derive(Clone, Debug)]
struct FeatureSet(std::vec::IntoIter<&'static str>, Option<parse::Error>);
impl Default for FeatureSet {
    fn default() -> Self {
        #[allow(clippy::unnecessary_to_owned)] // Falsely fires since array::IntoIter != vec::IntoIter
        Self(FEATURE_VERSIONS.to_vec().into_iter(), None)
    }
}
impl Parse for FeatureSet {
    fn parse(xs: ParseStream) -> parse::Result<Self> {
        let y = xs.parse::<VersionType>()?;
        let ys = get_features(y)?;
        Ok(Self(ys.into_iter(), None))
    }
}
impl Iterator for FeatureSet {
    type Item = &'static str;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
impl FeatureSet {
    #[inline]
    fn has_error(&self) -> bool {
        self.1.is_some()
    }
    #[inline]
    fn set_error(&mut self, x: parse::Error) {
        self.1 = Some(x);
    }
    fn into_error(self) -> parse::Error {
        self.1.unwrap()
    }
    fn into_compile_error(self) -> TokenStream {
        TokenStream::from(self.1.unwrap().to_compile_error())
    }
    fn expand_llvm_versions_attr(&mut self, x: &Attribute) -> Attribute {
        if self.has_error() {
            return x.clone();
        }
        if !x.path().is_ident("llvm_versions") {
            return x.clone();
        }
        match x.parse_args() {
            Ok(ParenthesizedFeatureSet(y)) => {
                parse_quote! {
                    #[cfg(any(#(feature = #y),*))]
                }
            },
            Err(y) => {
                self.set_error(y);
                x.clone()
            },
        }
    }
}
impl Fold for FeatureSet {
    fn fold_variant(&mut self, mut x: Variant) -> Variant {
        if self.has_error() {
            return x;
        }
        let ys = x
            .attrs
            .iter()
            .map(|x| self.expand_llvm_versions_attr(x))
            .collect::<Vec<_>>();
        x.attrs = ys;
        x
    }
    fn fold_field(&mut self, mut x: Field) -> Field {
        if self.has_error() {
            return x;
        }
        let ys = x
            .attrs
            .iter()
            .map(|x| self.expand_llvm_versions_attr(x))
            .collect::<Vec<_>>();
        x.attrs = ys;
        x
    }
}

const FEATURE_VERSIONS: [&str; 1] = ["llvm16-0"];
fn get_features(x: VersionType) -> syn::Result<Vec<&'static str>> {
    let ys = FEATURE_VERSIONS;
    let idx = get_latest(&ys);
    match x {
        VersionType::Specific(v, span) => {
            let feat = f64_to_feat(v);
            let i = get_idx(&ys, feat, span)?;
            Ok(ys[i..=i].to_vec())
        },
        VersionType::InclusiveToLatest(v, span) => {
            let feat = f64_to_feat(v);
            let i = get_idx(&ys, feat, span)?;
            Ok(ys[i..=idx].to_vec())
        },
        VersionType::Inclusive((start, start_span), (end, end_span)) => {
            let start_feat = f64_to_feat(start);
            let end_feat = f64_to_feat(end);
            let start_i = get_idx(&ys, start_feat, start_span)?;
            let end_i = get_idx(&ys, end_feat, end_span)?;
            if end_i < start_i {
                let m = format!("Invalid version range: {} to {}", start, end);
                Err(parse::Error::new(end_span, m))
            } else {
                Ok(ys[start_i..=end_i].to_vec())
            }
        },
        VersionType::ExclusiveToLatest(v, span) => {
            let feat = f64_to_feat(v);
            let i = get_idx(&ys, feat, span)?;
            if idx == i {
                let m = format!("Invalid version range: {}..latest", v);
                Err(parse::Error::new(span, m))
            } else {
                Ok(ys[i..idx].to_vec())
            }
        },
        VersionType::Exclusive((start, start_span), (end, end_span)) => {
            let start_feat = f64_to_feat(start);
            let end_feat = f64_to_feat(end);
            let start_i = get_idx(&ys, start_feat, start_span)?;
            let end_i = get_idx(&ys, end_feat, end_span)?;
            match end_i.cmp(&start_i) {
                std::cmp::Ordering::Equal => {
                    let m = format!("Invalid version range: {}..{}", start, end);
                    Err(parse::Error::new(start_span, m))
                },
                std::cmp::Ordering::Less => {
                    let m = format!("Invalid version range: {} to {}", start, end);
                    Err(parse::Error::new(end_span, m))
                },
                std::cmp::Ordering::Greater => Ok(ys[start_i..end_i].to_vec()),
            }
        },
    }
}
fn f64_to_feat(x: f64) -> String {
    let int = x as u64;
    format!("llvm{}-{}", int, (x * 10.) % 10.)
}
fn get_latest(xs: &[&str]) -> usize {
    xs.len() - 1
}
fn get_idx(xs: &[&str], feat: String, span: Span) -> syn::Result<usize> {
    let y = feat.as_str();
    match xs.iter().position(|&x| x == y) {
        None => Err(parse::Error::new(span, format!("Invalid feature version: {}", feat))),
        Some(x) => Ok(x),
    }
}

struct EnumVariant {
    llvm_variant: Ident,
    rust_variant: Ident,
    attrs: Vec<Attribute>,
}
impl EnumVariant {
    fn new(x: &Variant) -> Self {
        let rust_variant = x.ident.clone();
        let llvm_variant = Ident::new(&format!("LLVM{}", rust_variant), x.span());
        let mut attrs = x.attrs.clone();
        attrs.retain(|x| !x.path().is_ident("llvm_variant"));
        Self {
            llvm_variant,
            rust_variant,
            attrs,
        }
    }
    fn with_name(x: &Variant, mut llvm_variant: Ident) -> Self {
        let rust_variant = x.ident.clone();
        llvm_variant.set_span(rust_variant.span());
        let mut attrs = x.attrs.clone();
        attrs.retain(|x| !x.path().is_ident("llvm_variant"));
        Self {
            llvm_variant,
            rust_variant,
            attrs,
        }
    }
}

#[derive(Default)]
struct EnumVariants {
    variants: Vec<EnumVariant>,
    error: Option<parse::Error>,
}
impl EnumVariants {
    #[inline]
    fn len(&self) -> usize {
        self.variants.len()
    }
    #[inline]
    fn iter(&self) -> core::slice::Iter<'_, EnumVariant> {
        self.variants.iter()
    }
    #[inline]
    fn has_error(&self) -> bool {
        self.error.is_some()
    }
    #[inline]
    fn set_error(&mut self, err: &str, span: Span) {
        self.error = Some(parse::Error::new(span, err));
    }
    fn into_error(self) -> parse::Error {
        self.error.unwrap()
    }
}
impl Fold for EnumVariants {
    fn fold_variant(&mut self, mut y: Variant) -> Variant {
        use syn::Meta;
        if self.has_error() {
            return y;
        }
        if let Some(x) = y.attrs.iter().find(|x| x.path().is_ident("llvm_variant")) {
            if let Meta::List(meta) = &x.meta {
                if let Ok(Meta::Path(name)) = meta.parse_args() {
                    self.variants
                        .push(EnumVariant::with_name(&y, name.get_ident().unwrap().clone()));
                    y.attrs.retain(|x| !x.path().is_ident("llvm_variant"));
                    return y;
                }
            }
            self.set_error("expected #[llvm_variant(VARIANT_NAME)]", x.span());
            return y;
        }
        self.variants.push(EnumVariant::new(&y));
        y
    }
}

struct LLVMEnumType {
    name: Ident,
    decl: syn::ItemEnum,
    variants: EnumVariants,
}
impl Parse for LLVMEnumType {
    fn parse(xs: ParseStream) -> parse::Result<Self> {
        let decl = xs.parse::<syn::ItemEnum>()?;
        let name = decl.ident.clone();
        let mut ys = FeatureSet::default();
        let decl = ys.fold_item_enum(decl);
        if ys.has_error() {
            return Err(ys.into_error());
        }
        let mut ys = EnumVariants::default();
        let decl = ys.fold_item_enum(decl);
        if ys.has_error() {
            return Err(ys.into_error());
        }
        Ok(Self {
            name,
            decl,
            variants: ys,
        })
    }
}

#[proc_macro_attribute]
pub fn llvm_enum(args: TokenStream, attributee: TokenStream) -> TokenStream {
    use syn::{Arm, PatPath, Path};
    let llvm_ty = parse_macro_input!(args as Path);
    let llvm_enum_type = parse_macro_input!(attributee as LLVMEnumType);
    let mut from_arms = Vec::with_capacity(llvm_enum_type.variants.len());
    for variant in llvm_enum_type.variants.iter() {
        let src_variant = variant.llvm_variant.clone();
        let src_attrs: Vec<_> = variant
            .attrs
            .iter()
            .filter(|&attr| !attr.meta.path().is_ident("doc"))
            .collect();
        let src_ty = llvm_ty.clone();
        let dst_variant = variant.rust_variant.clone();
        let dst_ty = llvm_enum_type.name.clone();
        let pat = PatPath {
            attrs: Vec::new(),
            qself: None,
            path: parse_quote!(#src_ty::#src_variant),
        };
        let arm: Arm = parse_quote! {
            #(#src_attrs)*
            #pat => { #dst_ty::#dst_variant }
        };
        from_arms.push(arm);
    }
    let mut to_arms = Vec::with_capacity(llvm_enum_type.variants.len());
    for variant in llvm_enum_type.variants.iter() {
        let src_variant = variant.rust_variant.clone();
        let src_attrs: Vec<_> = variant
            .attrs
            .iter()
            .filter(|&attr| !attr.meta.path().is_ident("doc"))
            .collect();
        let src_ty = llvm_enum_type.name.clone();
        let dst_variant = variant.llvm_variant.clone();
        let dst_ty = llvm_ty.clone();
        let pat = PatPath {
            attrs: Vec::new(),
            qself: None,
            path: parse_quote!(#src_ty::#src_variant),
        };
        let arm: Arm = parse_quote! {
            #(#src_attrs)*
            #pat => { #dst_ty::#dst_variant }
        };
        to_arms.push(arm);
    }
    let enum_ty = llvm_enum_type.name.clone();
    let enum_decl = llvm_enum_type.decl;
    let q = quote! {
        #enum_decl
        impl #enum_ty {
            fn new(src: #llvm_ty) -> Self {
                match src {
                    #(#from_arms)*
                }
            }
        }
        impl From<#llvm_ty> for #enum_ty {
            fn from(src: #llvm_ty) -> Self {
                Self::new(src)
            }
        }
        impl Into<#llvm_ty> for #enum_ty {
            fn into(self) -> #llvm_ty {
                match self {
                    #(#to_arms),*
                }
            }
        }
    };
    q.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn map_normal_type_name() {
        assert_eq!(map_name("index"), "index");
        assert_eq!(map_name("integer"), "integer");
    }
    #[test]
    fn map_integer_type_name() {
        assert_eq!(map_name("i_64"), "i64");
    }
    #[test]
    fn map_float_type_name() {
        assert_eq!(map_name("f_64"), "f64");
        assert_eq!(map_name("float_8_e_5_m_2"), "float8e5m2");
        assert_eq!(map_name("float_8_e_4_m_3_fn"), "float8e4m3fn");
    }
}
