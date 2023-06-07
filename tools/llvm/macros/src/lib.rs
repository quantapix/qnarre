use convert_case::{Case, Casing};
use once_cell::sync::Lazy;
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;
use regex::{Captures, Regex};
use std::error::Error;
use syn::{
    bracketed,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Result, Token,
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
