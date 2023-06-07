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
    parse::{Error, Parse, ParseStream, Result},
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    spanned::Spanned,
    Attribute, Field, Ident, Item, LitFloat, Result, Token, Variant,
};

#[proc_macro]
pub fn binary_operations(stream: TokenStream) -> TokenStream {
    let set = parse_macro_input!(stream as DialectOperationSet);
    convert_result(generate_binary(set.dialect(), set.identifiers()))
}
#[proc_macro]
pub fn unary_operations(stream: TokenStream) -> TokenStream {
    let set = parse_macro_input!(stream as DialectOperationSet);
    convert_result(generate_unary(set.dialect(), set.identifiers()))
}
#[proc_macro]
pub fn typed_unary_operations(stream: TokenStream) -> TokenStream {
    let set = parse_macro_input!(stream as DialectOperationSet);
    convert_result(generate_typed_unary(set.dialect(), set.identifiers()))
}
#[proc_macro]
pub fn type_check_functions(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);
    convert_result(r#type::generate(identifiers.identifiers()))
}
#[proc_macro]
pub fn attribute_check_functions(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);
    convert_result(attribute::generate(identifiers.identifiers()))
}
#[proc_macro]
pub fn async_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);
    convert_result(pass::generate(identifiers.identifiers(), |name| {
        name.strip_prefix("Async").unwrap().into()
    }))
}
#[proc_macro]
pub fn conversion_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);
    convert_result(pass::generate(identifiers.identifiers(), |mut name| {
        name = name.strip_prefix("Conversion").unwrap();
        name = name.strip_prefix("Convert").unwrap_or(name);
        name.strip_suffix("ConversionPass").unwrap_or(name).into()
    }))
}
#[proc_macro]
pub fn gpu_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);
    convert_result(pass::generate(identifiers.identifiers(), |name| {
        name.strip_prefix("GPU").unwrap().into()
    }))
}
#[proc_macro]
pub fn transform_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);
    convert_result(pass::generate(identifiers.identifiers(), |name| {
        name.strip_prefix("Transforms").unwrap().into()
    }))
}
#[proc_macro]
pub fn linalg_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);
    convert_result(pass::generate(identifiers.identifiers(), |name| {
        name.strip_prefix("Linalg").unwrap().into()
    }))
}
#[proc_macro]
pub fn sparse_tensor_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);
    convert_result(pass::generate(identifiers.identifiers(), |name| {
        name.strip_prefix("SparseTensor").unwrap().into()
    }))
}
fn convert_result(result: Result<TokenStream, Box<dyn Error>>) -> TokenStream {
    result.unwrap_or_else(|error| {
        let message = error.to_string();
        quote! { compile_error!(#message) }.into()
    })
}
pub fn generate(identifiers: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();
    for identifier in identifiers {
        let name = map_name(
            &identifier
                .to_string()
                .strip_prefix("mlirAttributeIsA")
                .unwrap()
                .to_case(Case::Snake),
        );
        let function_name = Ident::new(&format!("is_{}", &name), identifier.span());
        let document = format!(" Returns `true` if an attribute is {}.", name.replace('_', " "));
        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            fn #function_name(&self) -> bool {
                unsafe { mlir_lib::#identifier(self.to_raw()) }
            }
        }));
    }
    Ok(stream)
}
pub fn generate_binary(dialect: &Ident, names: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();
    for name in names {
        let document = create_document(dialect, name);
        let operation_name = create_operation_name(dialect, name);
        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #name<'c>(
                lhs: crate::ir::Value,
                rhs: crate::ir::Value,
                location: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                binary_operator(#operation_name, lhs, rhs, location)
            }
        }));
    }
    stream.extend(TokenStream::from(quote! {
        fn binary_operator<'c>(
            name: &str,
            lhs: crate::ir::Value,
            rhs: crate::ir::Value,
            location: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::OperationBuilder::new(name, location)
                .add_operands(&[lhs, rhs])
                .enable_result_type_inference()
                .build()
        }
    }));
    Ok(stream)
}
pub fn generate_unary(dialect: &Ident, names: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();
    for name in names {
        let document = create_document(dialect, name);
        let operation_name = create_operation_name(dialect, name);
        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #name<'c>(
                value: crate::ir::Value,
                location: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                unary_operator(#operation_name, value, location)
            }
        }));
    }
    stream.extend(TokenStream::from(quote! {
        fn unary_operator<'c>(
            name: &str,
            value: crate::ir::Value,
            location: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::OperationBuilder::new(name, location)
                .add_operands(&[value])
                .enable_result_type_inference()
                .build()
        }
    }));
    Ok(stream)
}
pub fn generate_typed_unary(dialect: &Ident, names: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();
    for name in names {
        let document = create_document(dialect, name);
        let operation_name = create_operation_name(dialect, name);
        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #name<'c>(
                value: crate::ir::Value,
                r#type: crate::ir::Type<'c>,
                location: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                typed_unary_operator(#operation_name, value, r#type, location)
            }
        }));
    }
    stream.extend(TokenStream::from(quote! {
        fn typed_unary_operator<'c>(
            name: &str,
            value: crate::ir::Value,
            r#type: crate::ir::Type<'c>,
            location: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::OperationBuilder::new(name, location)
                .add_operands(&[value])
                .add_results(&[r#type])
                .build()
        }
    }));
    Ok(stream)
}
fn create_document(dialect: &Ident, name: &Ident) -> String {
    format!(" Creates an `{}` operation.", create_operation_name(dialect, name))
}
fn create_operation_name(dialect: &Ident, name: &Ident) -> String {
    format!("{}.{}", dialect, name)
}

pub struct IdentifierList {
    identifiers: Vec<Ident>,
}
impl IdentifierList {
    pub fn identifiers(&self) -> &[Ident] {
        &self.identifiers
    }
}
impl Parse for IdentifierList {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            identifiers: Punctuated::<Ident, Token![,]>::parse_terminated(input)?
                .into_iter()
                .collect(),
        })
    }
}

pub struct DialectOperationSet {
    dialect: Ident,
    identifiers: IdentifierList,
}
impl DialectOperationSet {
    pub fn dialect(&self) -> &Ident {
        &self.dialect
    }
    pub fn identifiers(&self) -> &[Ident] {
        self.identifiers.identifiers()
    }
}
impl Parse for DialectOperationSet {
    fn parse(input: ParseStream) -> Result<Self> {
        let dialect = Ident::parse(input)?;
        <Token![,]>::parse(input)?;
        Ok(Self {
            dialect,
            identifiers: {
                let content;
                bracketed!(content in input);
                content.parse::<IdentifierList>()?
            },
        })
    }
}

const CREATE_FUNCTION_PREFIX: &str = "mlirCreate";

pub fn generate(names: &[Ident], extract_pass_name: impl Fn(&str) -> String) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();
    for name in names {
        let foreign_name = name.to_string();
        let foreign_name = foreign_name.strip_prefix(CREATE_FUNCTION_PREFIX).unwrap();
        let pass_name = extract_pass_name(foreign_name);
        let function_name = create_function_name("create", &pass_name, name.span());
        let document = format!(" Creates a `{}` pass.", pass_name);
        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #function_name() -> crate::pass::Pass {
                unsafe { crate::pass::Pass::__private_from_raw_fn(mlir_lib::#name) }
            }
        }));
        let foreign_function_name = Ident::new(&("mlirRegister".to_owned() + foreign_name), name.span());
        let function_name = create_function_name("register", &pass_name, name.span());
        let document = format!(" Registers a `{}` pass.", pass_name);
        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #function_name() {
                unsafe { mlir_lib::#foreign_function_name() }
            }
        }));
    }
    Ok(stream)
}
fn create_function_name(prefix: &str, pass_name: &str, span: Span) -> Ident {
    Ident::new(&format!("{}_{}", prefix, &pass_name.to_case(Case::Snake)), span)
}
pub fn generate(identifiers: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();
    for identifier in identifiers {
        let name = map_name(
            &identifier
                .to_string()
                .strip_prefix("mlirTypeIsA")
                .unwrap()
                .to_case(Case::Snake),
        );
        let function_name = Ident::new(&format!("is_{}", &name), identifier.span());
        let document = format!(" Returns `true` if a type is {}.", name.replace('_', " "));
        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            fn #function_name(&self) -> bool {
                unsafe { mlir_lib::#identifier(self.to_raw()) }
            }
        }));
    }
    Ok(stream)
}
static PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(bf_16|f_16|f_32|f_64|i_8|i_16|i_32|i_64|float_8_e_[0-9]_m_[0-9](_fn)?)"#).unwrap());
pub fn map_name(name: &str) -> String {
    PATTERN
        .replace_all(name, |captures: &Captures| {
            captures.get(0).unwrap().as_str().replace('_', "")
        })
        .to_string()
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

const FEATURE_VERSIONS: [&str] = ["llvm16-0"];

fn get_latest_feature_index(features: &[&str]) -> usize {
    features.len() - 1
}

fn get_feature_index(features: &[&str], feature: String, span: Span) -> Result<usize> {
    let feat = feature.as_str();
    match features.iter().position(|&s| s == feat) {
        None => Err(Error::new(
            span,
            format!("Invalid feature version: {}, not defined", feature),
        )),
        Some(index) => Ok(index),
    }
}

fn get_features(vt: VersionType) -> Result<Vec<&'static str>> {
    let features = FEATURE_VERSIONS;
    let latest = get_latest_feature_index(&features);
    match vt {
        VersionType::Specific(version, span) => {
            let feature = f64_to_feature_string(version);
            let index = get_feature_index(&features, feature, span)?;
            Ok(features[index..=index].to_vec())
        },
        VersionType::InclusiveRangeToLatest(version, span) => {
            let feature = f64_to_feature_string(version);
            let index = get_feature_index(&features, feature, span)?;
            Ok(features[index..=latest].to_vec())
        },
        VersionType::InclusiveRange((start, start_span), (end, end_span)) => {
            let start_feature = f64_to_feature_string(start);
            let end_feature = f64_to_feature_string(end);
            let start_index = get_feature_index(&features, start_feature, start_span)?;
            let end_index = get_feature_index(&features, end_feature, end_span)?;
            if end_index < start_index {
                let message = format!(
                    "Invalid version range: {} must be greater than or equal to {}",
                    start, end
                );
                Err(Error::new(end_span, message))
            } else {
                Ok(features[start_index..=end_index].to_vec())
            }
        },
        VersionType::ExclusiveRangeToLatest(version, span) => {
            let feature = f64_to_feature_string(version);
            let index = get_feature_index(&features, feature, span)?;
            if latest == index {
                let message = format!(
                    "Invalid version range: {}..latest produces an empty feature set",
                    version
                );
                Err(Error::new(span, message))
            } else {
                Ok(features[index..latest].to_vec())
            }
        },
        VersionType::ExclusiveRange((start, start_span), (end, end_span)) => {
            let start_feature = f64_to_feature_string(start);
            let end_feature = f64_to_feature_string(end);
            let start_index = get_feature_index(&features, start_feature, start_span)?;
            let end_index = get_feature_index(&features, end_feature, end_span)?;

            match end_index.cmp(&start_index) {
                std::cmp::Ordering::Equal => {
                    let message = format!(
                        "Invalid version range: {}..{} produces an empty feature set",
                        start, end
                    );
                    Err(Error::new(start_span, message))
                },
                std::cmp::Ordering::Less => {
                    let message = format!("Invalid version range: {} must be greater than {}", start, end);
                    Err(Error::new(end_span, message))
                },

                std::cmp::Ordering::Greater => Ok(features[start_index..end_index].to_vec()),
            }
        },
    }
}

fn f64_to_feature_string(float: f64) -> String {
    let int = float as u64;

    format!("llvm{}-{}", int, (float * 10.) % 10.)
}

#[derive(Debug)]
enum VersionType {
    Specific(f64, Span),
    InclusiveRange((f64, Span), (f64, Span)),
    InclusiveRangeToLatest(f64, Span),
    ExclusiveRange((f64, Span), (f64, Span)),
    ExclusiveRangeToLatest(f64, Span),
}
impl Parse for VersionType {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(LitFloat) {
            let from = input.parse::<LitFloat>().unwrap();
            let from_val = from.base10_parse().unwrap();
            if input.is_empty() {
                return Ok(VersionType::Specific(from_val, from.span()));
            }
            let lookahead = input.lookahead1();
            if lookahead.peek(Token![..=]) {
                let _: Token![..=] = input.parse().unwrap();
                let lookahead = input.lookahead1();
                if lookahead.peek(Ident) {
                    let to = input.parse::<Ident>().unwrap();
                    if to == "latest" {
                        Ok(VersionType::InclusiveRangeToLatest(from_val, from.span()))
                    } else {
                        Err(Error::new(to.span(), "expected `latest` or `X.Y`"))
                    }
                } else if lookahead.peek(LitFloat) {
                    let to = input.parse::<LitFloat>().unwrap();
                    let to_val = to.base10_parse().unwrap();
                    Ok(VersionType::InclusiveRange(
                        (from_val, from.span()),
                        (to_val, to.span()),
                    ))
                } else {
                    Err(lookahead.error())
                }
            } else if lookahead.peek(Token![..]) {
                let _: Token![..] = input.parse().unwrap();
                let lookahead = input.lookahead1();
                if lookahead.peek(Ident) {
                    let to = input.parse::<Ident>().unwrap();
                    if to == "latest" {
                        Ok(VersionType::ExclusiveRangeToLatest(from_val, from.span()))
                    } else {
                        Err(Error::new(to.span(), "expected `latest` or `X.Y`"))
                    }
                } else if lookahead.peek(LitFloat) {
                    let to = input.parse::<LitFloat>().unwrap();
                    let to_val = to.base10_parse().unwrap();
                    Ok(VersionType::ExclusiveRange(
                        (from_val, from.span()),
                        (to_val, to.span()),
                    ))
                } else {
                    Err(lookahead.error())
                }
            } else {
                Err(lookahead.error())
            }
        } else {
            Err(lookahead.error())
        }
    }
}

#[derive(Debug)]
struct ParenthesizedFeatureSet(FeatureSet);
impl Parse for ParenthesizedFeatureSet {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<FeatureSet>().map(Self)
    }
}

#[derive(Clone, Debug)]
struct FeatureSet(std::vec::IntoIter<&'static str>, Option<Error>);
impl Default for FeatureSet {
    fn default() -> Self {
        #[allow(clippy::unnecessary_to_owned)] // Falsely fires since array::IntoIter != vec::IntoIter
        Self(FEATURE_VERSIONS.to_vec().into_iter(), None)
    }
}
impl Parse for FeatureSet {
    fn parse(input: ParseStream) -> Result<Self> {
        let version_type = input.parse::<VersionType>()?;
        let features = get_features(version_type)?;
        Ok(Self(features.into_iter(), None))
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
    fn set_error(&mut self, err: Error) {
        self.1 = Some(err);
    }

    fn into_error(self) -> Error {
        self.1.unwrap()
    }

    fn into_compile_error(self) -> TokenStream {
        TokenStream::from(self.1.unwrap().to_compile_error())
    }

    fn expand_llvm_versions_attr(&mut self, attr: &Attribute) -> Attribute {
        if self.has_error() {
            return attr.clone();
        }

        if !attr.path().is_ident("llvm_versions") {
            return attr.clone();
        }

        match attr.parse_args() {
            Ok(ParenthesizedFeatureSet(features)) => {
                parse_quote! {
                    #[cfg(any(#(feature = #features),*))]
                }
            },
            Err(err) => {
                self.set_error(err);
                attr.clone()
            },
        }
    }
}
impl Fold for FeatureSet {
    fn fold_variant(&mut self, mut variant: Variant) -> Variant {
        if self.has_error() {
            return variant;
        }

        let attrs = variant
            .attrs
            .iter()
            .map(|attr| self.expand_llvm_versions_attr(attr))
            .collect::<Vec<_>>();
        variant.attrs = attrs;
        variant
    }

    fn fold_field(&mut self, mut field: Field) -> Field {
        if self.has_error() {
            return field;
        }

        let attrs = field
            .attrs
            .iter()
            .map(|attr| self.expand_llvm_versions_attr(attr))
            .collect::<Vec<_>>();
        field.attrs = attrs;
        field
    }
}

#[proc_macro_attribute]
pub fn llvm_versions(attribute_args: TokenStream, attributee: TokenStream) -> TokenStream {
    let mut features = parse_macro_input!(attribute_args as FeatureSet);

    let attributee = parse_macro_input!(attributee as Item);
    let folded = features.fold_item(attributee);

    if features.has_error() {
        return features.into_compile_error();
    }

    let doc = if cfg!(feature = "nightly") {
        let features2 = features.clone();
        quote! {
            #[doc(cfg(any(#(feature = #features2),*)))]
        }
    } else {
        quote! {}
    };

    let q = quote! {
        #[cfg(any(#(feature = #features),*))]
        #doc
        #folded
    };

    q.into()
}

#[proc_macro_attribute]
pub fn llvm_versioned_item(_attribute_args: TokenStream, attributee: TokenStream) -> TokenStream {
    let attributee = parse_macro_input!(attributee as Item);

    let mut features = FeatureSet::default();
    let folded = features.fold_item(attributee);

    if features.has_error() {
        return features.into_compile_error();
    }

    quote!(#folded).into()
}

struct EnumVariant {
    llvm_variant: Ident,
    rust_variant: Ident,
    attrs: Vec<Attribute>,
}
impl EnumVariant {
    fn new(variant: &Variant) -> Self {
        let rust_variant = variant.ident.clone();
        let llvm_variant = Ident::new(&format!("LLVM{}", rust_variant), variant.span());
        let mut attrs = variant.attrs.clone();
        attrs.retain(|attr| !attr.path().is_ident("llvm_variant"));
        Self {
            llvm_variant,
            rust_variant,
            attrs,
        }
    }

    fn with_name(variant: &Variant, mut llvm_variant: Ident) -> Self {
        let rust_variant = variant.ident.clone();
        llvm_variant.set_span(rust_variant.span());
        let mut attrs = variant.attrs.clone();
        attrs.retain(|attr| !attr.path().is_ident("llvm_variant"));
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
    error: Option<Error>,
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
        self.error = Some(Error::new(span, err));
    }

    fn into_error(self) -> Error {
        self.error.unwrap()
    }
}
impl Fold for EnumVariants {
    fn fold_variant(&mut self, mut variant: Variant) -> Variant {
        use syn::Meta;

        if self.has_error() {
            return variant;
        }

        if let Some(attr) = variant.attrs.iter().find(|attr| attr.path().is_ident("llvm_variant")) {
            if let Meta::List(meta) = &attr.meta {
                if let Ok(Meta::Path(name)) = meta.parse_args() {
                    self.variants
                        .push(EnumVariant::with_name(&variant, name.get_ident().unwrap().clone()));
                    variant.attrs.retain(|attr| !attr.path().is_ident("llvm_variant"));
                    return variant;
                }
            }

            self.set_error("expected #[llvm_variant(VARIANT_NAME)]", attr.span());
            return variant;
        }

        self.variants.push(EnumVariant::new(&variant));
        variant
    }
}

struct LLVMEnumType {
    name: Ident,
    decl: syn::ItemEnum,
    variants: EnumVariants,
}
impl Parse for LLVMEnumType {
    fn parse(input: ParseStream) -> Result<Self> {
        let decl = input.parse::<syn::ItemEnum>()?;
        let name = decl.ident.clone();
        let mut features = FeatureSet::default();
        let decl = features.fold_item_enum(decl);
        if features.has_error() {
            return Err(features.into_error());
        }

        let mut variants = EnumVariants::default();
        let decl = variants.fold_item_enum(decl);
        if variants.has_error() {
            return Err(variants.into_error());
        }

        Ok(Self { name, decl, variants })
    }
}

#[proc_macro_attribute]
pub fn llvm_enum(attribute_args: TokenStream, attributee: TokenStream) -> TokenStream {
    use syn::{Arm, PatPath, Path};

    let llvm_ty = parse_macro_input!(attribute_args as Path);
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
