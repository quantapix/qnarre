use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::fold::Fold;
use syn::parse::{Error, Parse, ParseStream, Result};
use syn::spanned::Spanned;
use syn::{parse_macro_input, parse_quote};
use syn::{Attribute, Field, Ident, Item, LitFloat, Token, Variant};

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

///
///
///
///
///
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

///
///
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

///
///
///
///
///
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
