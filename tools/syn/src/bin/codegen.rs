extern crate proc_macro as pm;
use anyhow::Result;
use indexmap::IndexMap;
use inflections::Inflect;
use semver::Version;
use serde::de::{Deserialize, Deserializer};
use serde_derive::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use syn::*;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct Features {
    pub any: BTreeSet<String>,
}
fn config(xs: &Features) -> Stream {
    let xs = &xs.any;
    match xs.len() {
        0 => quote!(),
        1 => quote!(#[cfg(feature = #(#xs)*)]),
        _ => quote!(#[cfg(any(#(feature = #xs),*))]),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Type {
    Syn(String),
    Std(String),
    #[serde(rename = "proc_macro2")]
    Ext(String),
    Token(String),
    Group(String),
    Punct(Punct),
    Option(Box<Type>),
    Box(Box<Type>),
    Vec(Box<Type>),
    Tuple(Vec<Type>),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Punct {
    pub typ: Box<Type>,
    pub punct: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum Data {
    Private,
    #[serde(rename = "fields")]
    Struct(Fields),
    #[serde(rename = "variants")]
    Enum(Variants),
}
type Fields = IndexMap<String, Type>;
type Variants = IndexMap<String, Vec<Type>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Node {
    pub ident: String,
    pub feats: Features,
    #[serde(flatten, skip_serializing_if = "is_private", deserialize_with = "private_if_absent")]
    pub data: Data,
    #[serde(skip_serializing_if = "is_true", default = "bool_true")]
    pub exhaustive: bool,
}
fn is_private(x: &Data) -> bool {
    use Data::*;
    match x {
        Private => true,
        Struct(_) | Data::Enum(_) => false,
    }
}
fn private_if_absent<'a, D>(x: D) -> Result<Data, D::Error>
where
    D: Deserializer<'a>,
{
    let y = Option::deserialize(x)?;
    Ok(y.unwrap_or(Data::Private))
}
fn is_true(x: &bool) -> bool {
    *x
}
fn bool_true() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Defs {
    pub ver: Version,
    pub nodes: Vec<Node>,
    pub toks: BTreeMap<String, String>,
}

enum Operand {
    Borrow(Stream),
    Own(Stream),
}
use Operand::*;
impl Operand {
    pub fn tokens(&self) -> &Stream {
        match self {
            Borrow(x) | Own(x) => x,
        }
    }
    pub fn ref_tokens(&self) -> Stream {
        match self {
            Borrow(x) => x.clone(),
            Own(x) => quote!(&#n),
        }
    }
    pub fn ref_mut_tokens(&self) -> Stream {
        match self {
            Borrow(x) => x.clone(),
            Own(x) => quote!(&mut #n),
        }
    }
    pub fn owned_tokens(&self) -> Stream {
        match self {
            Borrow(x) => quote!(*#n),
            Own(x) => x.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Package {
    ver: Version,
}
#[derive(Debug, Deserialize)]
struct Manifest {
    pack: Package,
}
fn get_version() -> Result<Version> {
    let x = get_rel_path("Cargo.toml");
    let x = fs::read_to_string(x)?;
    let y: Manifest = toml::from_str(&x)?;
    Ok(y.pack.ver)
}
fn get_rel_path(x: impl AsRef<Path>) -> PathBuf {
    let mut y = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    assert!(y.pop());
    y.push(x);
    y
}

mod parse {
    use super::*;
    use anyhow::{bail, Result};
    use indexmap::IndexMap;
    use syn::parse::Error;
    use syn::{
        parse_quote, Data, DataEnum, DataStruct, Fields, GenericArgument, Ident, Input, Item, PathArguments, TypeMacro,
        TypePath, TypeTuple, UseTree, Visibility,
    };
    use syn_codegen as types;
    use thiserror::Error;
    const SYN_CRATE_ROOT: &str = "src/lib.rs";
    const TOKEN_SRC: &str = "src/token.rs";
    const IGNORED_MODS: &[&str] = &["fold", "visit", "visit_mut"];
    const EXTRA_TYPES: &[&str] = &["Lifetime"];
    struct Lookup {
        items: BTreeMap<Ident, AstItem>,
        tokens: BTreeMap<String, String>,
        aliases: BTreeMap<Ident, Ident>,
    }
    pub fn parse() -> Result<types::Definitions> {
        let tokens = load_token_file(TOKEN_SRC)?;
        let mut lookup = Lookup {
            items: BTreeMap::new(),
            tokens,
            aliases: BTreeMap::new(),
        };
        load_file(SYN_CRATE_ROOT, &[], &mut lookup)?;
        let version = get_version()?;
        let types = lookup
            .items
            .values()
            .map(|item| introspect_item(item, &lookup))
            .collect();
        let tokens = lookup.tokens.into_iter().map(|(name, ty)| (ty, name)).collect();
        Ok(types::Definitions { version, types, tokens })
    }
    pub struct AstItem {
        ast: Input,
        features: Vec<attr::Attr>,
    }
    fn introspect_item(item: &AstItem, lookup: &Lookup) -> types::Node {
        let features = introspect_features(&item.features);
        match &item.ast.data {
            Data::Enum(data) => types::Node {
                ident: item.ast.ident.to_string(),
                features,
                data: types::Data::Enum(introspect_enum(data, lookup)),
                exhaustive: !(is_non_exhaustive(&item.ast.attrs)
                    || data.variants.iter().any(|v| is_doc_hidden(&v.attrs))),
            },
            Data::Struct(data) => types::Node {
                ident: item.ast.ident.to_string(),
                features,
                data: {
                    if data.fields.iter().all(|f| is_pub(&f.vis)) {
                        types::Data::Struct(introspect_struct(data, lookup))
                    } else {
                        types::Data::Private
                    }
                },
                exhaustive: true,
            },
            Data::Union(..) => panic!("Union not supported"),
        }
    }
    fn introspect_enum(item: &DataEnum, lookup: &Lookup) -> types::Variants {
        item.variants
            .iter()
            .filter_map(|variant| {
                if is_doc_hidden(&variant.attrs) {
                    return None;
                }
                let fields = match &variant.fields {
                    Fields::Unnamed(fields) => fields
                        .unnamed
                        .iter()
                        .map(|field| introspect_type(&field.ty, lookup))
                        .collect(),
                    Fields::Unit => vec![],
                    Fields::Named(_) => panic!("Enum representation not supported"),
                };
                Some((variant.ident.to_string(), fields))
            })
            .collect()
    }
    fn introspect_struct(item: &DataStruct, lookup: &Lookup) -> types::Fields {
        match &item.fields {
            Fields::Named(fields) => fields
                .named
                .iter()
                .map(|field| {
                    (
                        field.ident.as_ref().unwrap().to_string(),
                        introspect_type(&field.ty, lookup),
                    )
                })
                .collect(),
            Fields::Unit => IndexMap::new(),
            Fields::Unnamed(_) => panic!("Struct representation not supported"),
        }
    }
    fn introspect_type(item: &syn::Type, lookup: &Lookup) -> types::Type {
        match item {
            syn::Type::Path(TypePath { qself: None, path }) => {
                let last = path.segs.last().unwrap();
                let string = last.ident.to_string();
                match string.as_str() {
                    "Option" => {
                        let nested = introspect_type(first_arg(&last.args), lookup);
                        types::Type::Option(Box::new(nested))
                    },
                    "Punctuated" => {
                        let nested = introspect_type(first_arg(&last.args), lookup);
                        let punct = match introspect_type(last_arg(&last.args), lookup) {
                            types::Type::Token(s) => s,
                            _ => panic!(),
                        };
                        types::Type::Punctuated(types::Punctuated {
                            element: Box::new(nested),
                            punct,
                        })
                    },
                    "Vec" => {
                        let nested = introspect_type(first_arg(&last.args), lookup);
                        types::Type::Vec(Box::new(nested))
                    },
                    "Box" => {
                        let nested = introspect_type(first_arg(&last.args), lookup);
                        types::Type::Box(Box::new(nested))
                    },
                    "Brace" | "Bracket" | "Parenth" | "Group" => types::Type::Group(string),
                    "Stream" | "Literal" | "Ident" | "Span" => types::Type::Ext(string),
                    "String" | "u32" | "usize" | "bool" => types::Type::Std(string),
                    _ => {
                        let mut resolved = &last.ident;
                        while let Some(alias) = lookup.aliases.get(resolved) {
                            resolved = alias;
                        }
                        if lookup.items.get(resolved).is_some() {
                            types::Type::Syn(resolved.to_string())
                        } else {
                            unimplemented!("{}", resolved);
                        }
                    },
                }
            },
            syn::Type::Tuple(TypeTuple { elems, .. }) => {
                let tys = elems.iter().map(|ty| introspect_type(ty, lookup)).collect();
                types::Type::Tuple(tys)
            },
            syn::Type::Macro(TypeMacro { mac }) if mac.path.segs.last().unwrap().ident == "Token" => {
                let content = mac.tokens.to_string();
                let ty = lookup.tokens.get(&content).unwrap().to_string();
                types::Type::Token(ty)
            },
            _ => panic!("{}", quote!(#item).to_string()),
        }
    }
    fn introspect_features(attrs: &[attr::Attr]) -> types::Features {
        let mut ret = types::Features::default();
        for attr in attrs {
            if !attr.path().is_ident("cfg") {
                continue;
            }
            let features = attr.parse_args_with(parsing::parse_features).unwrap();
            if ret.any.is_empty() {
                ret = features;
            } else if ret.any.len() < features.any.len() {
                assert!(ret.any.iter().all(|f| features.any.contains(f)));
            } else {
                assert!(features.any.iter().all(|f| ret.any.contains(f)));
                ret = features;
            }
        }
        ret
    }
    fn is_pub(vis: &Visibility) -> bool {
        match vis {
            Visibility::Public(_) => true,
            _ => false,
        }
    }
    fn is_non_exhaustive(attrs: &[attr::Attr]) -> bool {
        for attr in attrs {
            if attr.path().is_ident("non_exhaustive") {
                return true;
            }
        }
        false
    }
    fn is_doc_hidden(attrs: &[attr::Attr]) -> bool {
        for attr in attrs {
            if attr.path().is_ident("doc") && attr.parse_args::<parsing::kw::hidden>().is_ok() {
                return true;
            }
        }
        false
    }
    fn first_arg(params: &PathArguments) -> &syn::Type {
        let data = match params {
            PathArguments::AngleBracketed(data) => data,
            _ => panic!("Expected at least 1 type argument here"),
        };
        match data.args.first().expect("Expected at least 1 type argument here") {
            GenericArgument::Type(ty) => ty,
            _ => panic!("Expected at least 1 type argument here"),
        }
    }
    fn last_arg(params: &PathArguments) -> &syn::Type {
        let data = match params {
            PathArguments::AngleBracketed(data) => data,
            _ => panic!("Expected at least 1 type argument here"),
        };
        match data.args.last().expect("Expected at least 1 type argument here") {
            GenericArgument::Type(ty) => ty,
            _ => panic!("Expected at least 1 type argument here"),
        }
    }
    mod parsing {
        use super::AstItem;
        use std::collections::{BTreeMap, BTreeSet};
        use syn::*;
        use syn_codegen as types;
        fn peek_tag(s: parse::Stream, tag: &str) -> bool {
            let ahead = s.fork();
            ahead.parse::<Token![#]>().is_ok() && ahead.parse::<Ident>().map(|ident| ident == tag).unwrap_or(false)
        }
        fn full(s: parse::Stream) -> Vec<attr::Attr> {
            if peek_tag(s, "full") {
                s.parse::<Token![#]>().unwrap();
                s.parse::<Ident>().unwrap();
                vec![parse_quote!(#[cfg(feature = "full")])]
            } else {
                vec![]
            }
        }
        fn ast_struct_inner(s: parse::Stream) -> Result<AstItem> {
            let ident: Ident = s.parse()?;
            let features = full(s);
            let rest: Stream = s.parse()?;
            Ok(AstItem {
                ast: syn::parse2(quote! {
                    pub struct #ident #rest
                })?,
                features,
            })
        }
        pub fn ast_struct(s: parse::Stream) -> Result<AstItem> {
            s.call(attr::Attr::parse_outer)?;
            s.parse::<Token![pub]>()?;
            s.parse::<Token![struct]>()?;
            let res = s.call(ast_struct_inner)?;
            Ok(res)
        }
        fn no_visit(s: parse::Stream) -> bool {
            if peek_tag(s, "no_visit") {
                s.parse::<Token![#]>().unwrap();
                s.parse::<Ident>().unwrap();
                true
            } else {
                false
            }
        }
        pub fn ast_enum(s: parse::Stream) -> Result<Option<AstItem>> {
            let attrs = s.call(attr::Attr::parse_outer)?;
            s.parse::<Token![pub]>()?;
            s.parse::<Token![enum]>()?;
            let ident: Ident = s.parse()?;
            let no_visit = no_visit(s);
            let rest: Stream = s.parse()?;
            Ok(if no_visit {
                None
            } else {
                Some(AstItem {
                    ast: syn::parse2(quote! {
                        #(#attrs)*
                        pub enum #ident #rest
                    })?,
                    features: vec![],
                })
            })
        }
        struct EosVariant {
            attrs: Vec<attr::Attr>,
            name: Ident,
            member: Option<Path>,
        }
        fn eos_variant(s: parse::Stream) -> Result<EosVariant> {
            let attrs = s.call(attr::Attr::parse_outer)?;
            let variant: Ident = s.parse()?;
            let member = if s.peek(tok::Parenth) {
                let content;
                parenthed!(content in s);
                let path: Path = content.parse()?;
                Some(path)
            } else {
                None
            };
            s.parse::<Token![,]>()?;
            Ok(EosVariant {
                attrs,
                name: variant,
                member,
            })
        }
        pub fn enum_of_structs(s: parse::Stream) -> Result<AstItem> {
            let attrs = s.call(attr::Attr::parse_outer)?;
            s.parse::<Token![pub]>()?;
            s.parse::<Token![enum]>()?;
            let ident: Ident = s.parse()?;
            let content;
            braced!(content in s);
            let mut variants = Vec::new();
            while !content.is_empty() {
                variants.push(content.call(eos_variant)?);
            }
            let enum_item = {
                let variants = variants.iter().map(|v| {
                    let attrs = &v.attrs;
                    let name = &v.name;
                    if let Some(member) = &v.member {
                        quote!(#(#attrs)* #name(#member))
                    } else {
                        quote!(#(#attrs)* #name)
                    }
                });
                parse_quote! {
                    #(#attrs)*
                    pub enum #ident {
                        #(#variants),*
                    }
                }
            };
            Ok(AstItem {
                ast: enum_item,
                features: vec![],
            })
        }
        pub mod kw {
            use syn::*;
            custom_kw!(hidden);
            custom_kw!(macro_rules);
            custom_kw!(Token);
        }
        pub fn parse_token_macro(s: parse::Stream) -> Result<BTreeMap<String, String>> {
            let mut tokens = BTreeMap::new();
            while !s.is_empty() {
                let pattern;
                bracketed!(pattern in s);
                let token = pattern.parse::<Stream>()?.to_string();
                s.parse::<Token![=>]>()?;
                let expansion;
                braced!(expansion in s);
                s.parse::<Token![;]>()?;
                expansion.parse::<Token![$]>()?;
                let path: Path = expansion.parse()?;
                let ty = path.segs.last().unwrap().ident.to_string();
                tokens.insert(token, ty.to_string());
            }
            Ok(tokens)
        }
        fn parse_feature(s: parse::Stream) -> Result<String> {
            let i: Ident = s.parse()?;
            assert_eq!(i, "feature");
            s.parse::<Token![=]>()?;
            let s = s.parse::<LitStr>()?;
            Ok(s.value())
        }
        pub fn parse_features(s: parse::Stream) -> Result<types::Features> {
            let mut features = BTreeSet::new();
            let i: Ident = s.fork().parse()?;
            if i == "any" {
                s.parse::<Ident>()?;
                let nested;
                parenthed!(nested in s);
                while !nested.is_empty() {
                    features.insert(parse_feature(&nested)?);
                    if !nested.is_empty() {
                        nested.parse::<Token![,]>()?;
                    }
                }
            } else if i == "feature" {
                features.insert(parse_feature(s)?);
                assert!(s.is_empty());
            } else {
                panic!("{:?}", i);
            }
            Ok(types::Features { any: features })
        }
        pub fn path_attr(attrs: &[attr::Attr]) -> Result<Option<&LitStr>> {
            for attr in attrs {
                if attr.path().is_ident("path") {
                    if let Expr::Lit(expr) = &attr.meta.require_name_value()?.value {
                        if let Lit::Str(lit) = &expr.lit {
                            return Ok(Some(lit));
                        }
                    }
                }
            }
            Ok(None)
        }
    }
    fn clone_features(features: &[attr::Attr]) -> Vec<attr::Attr> {
        features.iter().map(|attr| parse_quote!(#attr)).collect()
    }
    fn get_features(attrs: &[attr::Attr], base: &[attr::Attr]) -> Vec<attr::Attr> {
        let mut ret = clone_features(base);
        for attr in attrs {
            if attr.path().is_ident("cfg") {
                ret.push(parse_quote!(#attr));
            }
        }
        ret
    }
    #[derive(Error, Debug)]
    #[error("{path}:{line}:{column}: {error}")]
    struct LoadFileError {
        path: PathBuf,
        line: usize,
        column: usize,
        error: Error,
    }
    fn load_file(
        relative_to_workspace_root: impl AsRef<Path>,
        features: &[attr::Attr],
        lookup: &mut Lookup,
    ) -> Result<()> {
        let error = match do_load_file(&relative_to_workspace_root, features, lookup).err() {
            None => return Ok(()),
            Some(error) => error,
        };
        let error = error.downcast::<Error>()?;
        let span = error.span().start();
        bail!(LoadFileError {
            path: relative_to_workspace_root.as_ref().to_owned(),
            line: span.line,
            column: span.column + 1,
            error,
        })
    }
    fn do_load_file(
        relative_to_workspace_root: impl AsRef<Path>,
        features: &[attr::Attr],
        lookup: &mut Lookup,
    ) -> Result<()> {
        let relative_to_workspace_root = relative_to_workspace_root.as_ref();
        let parent = relative_to_workspace_root.parent().expect("no parent path");
        let src = fs::read_to_string(get_rel_path(relative_to_workspace_root))?;
        let file = syn::parse_file(&src)?;
        'items: for item in file.items {
            match item {
                Item::Mod(item) => {
                    if item.content.is_some() {
                        continue;
                    }
                    for name in IGNORED_MODS {
                        if item.ident == name {
                            continue 'items;
                        }
                    }
                    let features = if item.ident == "derive" {
                        vec![parse_quote!(#[cfg(feature = "derive")])]
                    } else {
                        get_features(&item.attrs, features)
                    };
                    let filename = if let Some(filename) = parsing::path_attr(&item.attrs)? {
                        filename.value()
                    } else {
                        format!("{}.rs", item.ident)
                    };
                    let path = parent.join(filename);
                    load_file(path, &features, lookup)?;
                },
                Item::Macro(item) => {
                    let features = get_features(&item.attrs, features);
                    let tts = item.mac.tokens.clone();
                    let found = if item.mac.path.is_ident("ast_struct") {
                        Some(parsing::ast_struct.parse2(tts)?)
                    } else if item.mac.path.is_ident("ast_enum") {
                        parsing::ast_enum.parse2(tts)?
                    } else if item.mac.path.is_ident("enum_of_structs") {
                        Some(parsing::enum_of_structs.parse2(tts)?)
                    } else {
                        continue;
                    };
                    if let Some(mut item) = found {
                        item.features.extend(clone_features(&features));
                        lookup.items.insert(item.ast.ident.clone(), item);
                    }
                },
                Item::Struct(item) => {
                    let ident = item.ident;
                    if EXTRA_TYPES.contains(&&ident.to_string()[..]) {
                        lookup.items.insert(
                            ident.clone(),
                            AstItem {
                                ast: Input {
                                    ident,
                                    vis: item.vis,
                                    attrs: item.attrs,
                                    generics: item.generics,
                                    data: Data::Struct(DataStruct {
                                        fields: item.fields,
                                        struct_token: item.struct_token,
                                        semi_token: item.semi_token,
                                    }),
                                },
                                features: clone_features(features),
                            },
                        );
                    }
                },
                Item::Use(item)
                    if relative_to_workspace_root == Path::new(SYN_CRATE_ROOT)
                        && matches!(item.vis, Visibility::Public(_)) =>
                {
                    load_aliases(item.tree, lookup);
                },
                _ => {},
            }
        }
        Ok(())
    }
    fn load_aliases(use_tree: UseTree, lookup: &mut Lookup) {
        match use_tree {
            UseTree::Path(use_tree) => load_aliases(*use_tree.tree, lookup),
            UseTree::Rename(use_tree) => {
                lookup.aliases.insert(use_tree.rename, use_tree.ident);
            },
            UseTree::Group(use_tree) => {
                for use_tree in use_tree.items {
                    load_aliases(use_tree, lookup);
                }
            },
            UseTree::Name(_) | UseTree::Glob(_) => {},
        }
    }
    fn load_token_file(relative_to_workspace_root: impl AsRef<Path>) -> Result<BTreeMap<String, String>> {
        let path = get_rel_path(relative_to_workspace_root);
        let src = fs::read_to_string(path)?;
        let file = syn::parse_file(&src)?;
        for item in file.items {
            if let Item::Macro(item) = item {
                match item.ident {
                    Some(i) if i == "Token" => {},
                    _ => continue,
                }
                let tokens = item.mac.parse_body_with(parsing::parse_token_macro)?;
                return Ok(tokens);
            }
        }
        panic!("failed to parse Token macro")
    }
}

fn write_to_file(path: impl AsRef<Path>, x: Stream) -> Result<()> {
    let mut y = Vec::new();
    writeln!(y, "// File generated by syn-codegen.")?;
    writeln!(y, "// Not intended for manual editing.")?;
    writeln!(y)?;
    let tree: syn::File = syn::parse2(x).unwrap();
    let pretty = syn::pretty::unparse(&tree);
    write!(y, "{}", pretty)?;
    let path = get_rel_path(path);
    if path.is_file() && fs::read(&path)? == y {
        return Ok(());
    }
    fs::write(path, y)?;
    Ok(())
}
fn lookup_node<'a>(xs: &'a Defs, name: &str) -> &'a Node {
    for x in &xs.nodes {
        if x.ident == name {
            return x;
        }
    }
    panic!("not found: {}", name)
}

mod clone {
    use super::*;
    const SOURCE: &str = "src/codegen/clone.rs";
    pub fn generate(xs: &Defs) -> Result<()> {
        let mut ys = Stream::new();
        for x in &xs.nodes {
            ys.extend(expand(xs, x));
        }
        write_to_file(
            SOURCE,
            quote! {
                #![allow(clippy::clone_on_copy, clippy::expl_impl_clone_on_copy)]
                use crate::*;
                #ys
            },
        )?;
        Ok(())
    }
    fn expand(xs: &Defs, n: &Node) -> Stream {
        let manual = n.data == Data::Private || n.ident == "Lifetime";
        if manual {
            return Stream::new();
        }
        let typ = Ident::new(&n.ident, Span::call_site());
        let cfg = config(&n.feats);
        let copy = n.ident == "AttrStyle"
            || n.ident == "BinOp"
            || n.ident == "RangeLimits"
            || n.ident == "TraitBoundModifier"
            || n.ident == "UnOp";
        if copy {
            return quote! {
                #cfg
                impl Copy for #typ {}
                #cfg
                impl Clone for #typ {
                    fn clone(&self) -> Self {
                        *self
                    }
                }
            };
        }
        let body = expand_body(xs, n);
        quote! {
            #cfg
            impl Clone for #typ {
                fn clone(&self) -> Self {
                    #body
                }
            }
        }
    }
    fn expand_body(ds: &Defs, n: &Node) -> Stream {
        let typ = &n.ident;
        let typ = Ident::new(typ, Span::call_site());
        match &n.data {
            Data::Enum(x) if x.is_empty() => quote!(match *self {}),
            Data::Enum(x) => {
                let arms = x.iter().map(|(name, fields)| {
                    let var = Ident::new(name, Span::call_site());
                    if fields.is_empty() {
                        quote! {
                            #typ::#var => #typ::#var,
                        }
                    } else {
                        let mut pats = Vec::new();
                        let mut clones = Vec::new();
                        for i in 0..fields.len() {
                            let x = format_ident!("v{}", i);
                            clones.push(quote!(#x.clone()));
                            pats.push(x);
                        }
                        let mut cfg = None;
                        if n.ident == "Expr" {
                            if let Type::Syn(x) = &fields[0] {
                                if !lookup_node(ds, x).feats.any.contains("derive") {
                                    cfg = Some(quote!(#[cfg(feature = "full")]));
                                }
                            }
                        }
                        quote! {
                            #cfg
                            #typ::#var(#(#pats),*) => #typ::#var(#(#clones),*),
                        }
                    }
                });
                let nonexh = if n.ident == "Expr" {
                    Some(quote! {
                        #[cfg(not(feature = "full"))]
                        _ => unreachable!(),
                    })
                } else {
                    None
                };
                quote! {
                    match self {
                        #(#arms)*
                        #nonexh
                    }
                }
            },
            Data::Struct(x) => {
                let fields = x.keys().map(|f| {
                    let field = Ident::new(f, Span::call_site());
                    quote! {
                        #field: self.#field.clone(),
                    }
                });
                quote!(#typ { #(#fields)* })
            },
            Data::Private => unreachable!(),
        }
    }
}
mod debug {
    use super::*;
    const SOURCE: &str = "src/codegen/debug.rs";
    pub fn generate(xs: &Defs) -> Result<()> {
        let mut vars = BTreeSet::new();
        for x in &xs.nodes {
            if let Data::Enum(ys) = &x.data {
                let e = &x.ident;
                for (v, fs) in ys {
                    if let Some(x) = tree_enum(e, v, fs) {
                        vars.insert(x);
                    }
                }
            }
        }
        let mut ys = Stream::new();
        for x in &xs.nodes {
            ys.extend(expand(xs, x, &vars));
        }
        write_to_file(
            SOURCE,
            quote! {
                use crate::*;
                use std::fmt::{self, Debug};
                #ys
            },
        )?;
        Ok(())
    }
    fn tree_enum<'a>(e: &str, v: &str, fs: &'a [Type]) -> Option<&'a str> {
        if fs.len() != 1 {
            return None;
        }
        const WHITELIST: &[(&str, &str)] = &[
            ("Meta", "Path"),
            ("Pat", "Const"),
            ("Pat", "Lit"),
            ("Pat", "Macro"),
            ("Pat", "Path"),
            ("Pat", "Range"),
            ("PathArguments", "AngleBracketed"),
            ("PathArguments", "Parenthesized"),
            ("Stmt", "Local"),
            ("TypeParamBound", "Lifetime"),
            ("Visibility", "Public"),
            ("Visibility", "Restricted"),
        ];
        match &fs[0] {
            Type::Syn(x) if WHITELIST.contains(&(e, v)) || e.to_owned() + v == *x => Some(x),
            _ => None,
        }
    }
    fn expand(xs: &Defs, n: &Node, vars: &BTreeSet<&str>) -> Stream {
        let manual = n.data == Data::Private || n.ident == "LitBool";
        if manual {
            return Stream::new();
        }
        let typ = Ident::new(&n.ident, Span::call_site());
        let cfg = config(&n.feats);
        let f = match &n.data {
            Data::Enum(x) if x.is_empty() => quote!(_),
            _ => quote!(f),
        };
        let body = expand_body(xs, n, vars);
        quote! {
            #cfg
            impl Debug for #typ {
                fn fmt(&self, #f: &mut fmt::Formatter) -> fmt::Result {
                    #body
                }
            }
        }
    }
    fn expand_body(defs: &Defs, n: &Node, vars: &BTreeSet<&str>) -> Stream {
        let type_name = &n.ident;
        let ident = Ident::new(type_name, Span::call_site());
        let is_syntax_tree_variant = vars.contains(type_name.as_str());
        let body = match &n.data {
            Data::Enum(x) if x.is_empty() => quote!(match *self {}),
            Data::Enum(x) => {
                assert!(!is_syntax_tree_variant);
                let arms = x.iter().map(|(name, fields)| {
                    let variant = Ident::new(name, Span::call_site());
                    if fields.is_empty() {
                        quote! {
                            #ident::#variant => formatter.write_str(#variant_name),
                        }
                    } else {
                        let mut cfg = None;
                        if n.ident == "Expr" {
                            if let Type::Syn(ty) = &fields[0] {
                                if !lookup_node(defs, ty).feats.any.contains("derive") {
                                    cfg = Some(quote!(#[cfg(feature = "full")]));
                                }
                            }
                        }
                        if tree_enum(type_name, name, fields).is_some() {
                            quote! {
                                #cfg
                                #ident::#variant(v0) => v0.debug(formatter, #variant_name),
                            }
                        } else {
                            let pats = (0..fields.len()).map(|i| format_ident!("v{}", i)).collect::<Vec<_>>();
                            quote! {
                                #cfg
                                #ident::#variant(#(#pats),*) => {
                                    let mut formatter = formatter.debug_tuple(#variant_name);
                                    #(formatter.field(#pats);)*
                                    formatter.finish()
                                }
                            }
                        }
                    }
                });
                let nonexhaustive = if n.ident == "Expr" {
                    Some(quote! {
                        #[cfg(not(feature = "full"))]
                        _ => unreachable!(),
                    })
                } else {
                    None
                };
                let prefix = format!("{}::", type_name);
                quote! {
                    formatter.write_str(#prefix)?;
                    match self {
                        #(#arms)*
                        #nonexhaustive
                    }
                }
            },
            Data::Struct(fields) => {
                let type_name = if is_syntax_tree_variant {
                    quote!(name)
                } else {
                    quote!(#type_name)
                };
                let fields = fields.keys().map(|f| {
                    let ident = Ident::new(f, Span::call_site());
                    quote! {
                        formatter.field(#f, &self.#ident);
                    }
                });
                quote! {
                    let mut formatter = formatter.debug_struct(#type_name);
                    #(#fields)*
                    formatter.finish()
                }
            },
            Data::Private => unreachable!(),
        };
        if is_syntax_tree_variant {
            quote! {
                impl #ident {
                    fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                        #body
                    }
                }
                self.debug(formatter, #type_name)
            }
        } else {
            body
        }
    }
}
mod eq {
    use super::*;
    const SOURCE: &str = "src/codegen/eq.rs";
    pub fn generate(xs: &Defs) -> Result<()> {
        let mut ys = Stream::new();
        for x in &xs.nodes {
            ys.extend(expand(xs, x));
        }
        write_to_file(
            SOURCE,
            quote! {
                use crate::*;
                #ys
            },
        )?;
        Ok(())
    }
    fn expand(defs: &Defs, node: &Node) -> Stream {
        if node.ident == "Member" || node.ident == "Index" || node.ident == "Lifetime" {
            return Stream::new();
        }
        let ident = Ident::new(&node.ident, Span::call_site());
        let cfg_features = config(&node.feats);
        let eq = quote! {
            #cfg_features
            #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
            impl Eq for #ident {}
        };
        let manual_partial_eq = node.data == Data::Private;
        if manual_partial_eq {
            return eq;
        }
        let body = expand_body(defs, node);
        let other = match &node.data {
            Data::Enum(variants) if variants.is_empty() => quote!(_other),
            Data::Struct(fields) if fields.values().all(always_eq) => quote!(_other),
            _ => quote!(other),
        };
        quote! {
            #eq
            #cfg_features
            #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
            impl PartialEq for #ident {
                fn eq(&self, #other: &Self) -> bool {
                    #body
                }
            }
        }
    }
    fn expand_body(defs: &Defs, node: &Node) -> Stream {
        let type_name = &node.ident;
        let ident = Ident::new(type_name, Span::call_site());
        match &node.data {
            Data::Enum(variants) if variants.is_empty() => quote!(match *self {}),
            Data::Enum(variants) => {
                let arms = variants.iter().map(|(variant_name, fields)| {
                    let variant = Ident::new(variant_name, Span::call_site());
                    if fields.is_empty() {
                        quote! {
                            (#ident::#variant, #ident::#variant) => true,
                        }
                    } else {
                        let mut this_pats = Vec::new();
                        let mut other_pats = Vec::new();
                        let mut comparisons = Vec::new();
                        for (i, field) in fields.iter().enumerate() {
                            if always_eq(field) {
                                this_pats.push(format_ident!("_"));
                                other_pats.push(format_ident!("_"));
                                continue;
                            }
                            let this = format_ident!("self{}", i);
                            let other = format_ident!("other{}", i);
                            comparisons.push(match field {
                                Type::Ext(ty) if ty == "Stream" => {
                                    quote!(StreamHelper(#this) == StreamHelper(#other))
                                },
                                Type::Ext(ty) if ty == "Literal" => {
                                    quote!(#this.to_string() == #other.to_string())
                                },
                                _ => quote!(#this == #other),
                            });
                            this_pats.push(this);
                            other_pats.push(other);
                        }
                        if comparisons.is_empty() {
                            comparisons.push(quote!(true));
                        }
                        let mut cfg = None;
                        if node.ident == "Expr" {
                            if let Type::Syn(ty) = &fields[0] {
                                if !lookup_node(defs, ty).feats.any.contains("derive") {
                                    cfg = Some(quote!(#[cfg(feature = "full")]));
                                }
                            }
                        }
                        quote! {
                            #cfg
                            (#ident::#variant(#(#this_pats),*), #ident::#variant(#(#other_pats),*)) => {
                                #(#comparisons)&&*
                            }
                        }
                    }
                });
                let fallthrough = if variants.len() == 1 {
                    None
                } else {
                    Some(quote!(_ => false,))
                };
                quote! {
                    match (self, other) {
                        #(#arms)*
                        #fallthrough
                    }
                }
            },
            Data::Struct(fields) => {
                let mut comparisons = Vec::new();
                for (f, ty) in fields {
                    if always_eq(ty) {
                        continue;
                    }
                    let ident = Ident::new(f, Span::call_site());
                    comparisons.push(match ty {
                        Type::Ext(ty) if ty == "Stream" => {
                            quote!(StreamHelper(&self.#ident) == StreamHelper(&other.#ident))
                        },
                        _ => quote!(self.#ident == other.#ident),
                    });
                }
                if comparisons.is_empty() {
                    quote!(true)
                } else {
                    quote!(#(#comparisons)&&*)
                }
            },
            Data::Private => unreachable!(),
        }
    }
    fn always_eq(field_type: &Type) -> bool {
        match field_type {
            Type::Ext(ty) => ty == "Span",
            Type::Token(_) | Type::Group(_) => true,
            Type::Box(inner) => always_eq(inner),
            Type::Tuple(inner) => inner.iter().all(always_eq),
            _ => false,
        }
    }
}
mod hash {
    use super::*;
    const SOURCE: &str = "src/codegen/hash.rs";
    pub fn generate(xs: &Defs) -> Result<()> {
        let mut ys = Stream::new();
        for x in &xs.nodes {
            ys.extend(expand(xs, x));
        }
        write_to_file(
            SOURCE,
            quote! {
                use crate::*;
                use std::hash::{Hash, Hasher};
                #ys
            },
        )?;
        Ok(())
    }
    fn expand(defs: &Defs, node: &Node) -> Stream {
        let manual_hash =
            node.data == Data::Private || node.ident == "Member" || node.ident == "Index" || node.ident == "Lifetime";
        if manual_hash {
            return Stream::new();
        }
        let ident = Ident::new(&node.ident, Span::call_site());
        let cfg_features = config(&node.feats);
        let body = expand_body(defs, node);
        let hasher = match &node.data {
            Data::Struct(_) if body.is_empty() => quote!(_state),
            Data::Enum(variants) if variants.is_empty() => quote!(_state),
            _ => quote!(state),
        };
        quote! {
            #cfg_features
            #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
            impl Hash for #ident {
                fn hash<H>(&self, #hasher: &mut H)
                where
                    H: Hasher,
                {
                    #body
                }
            }
        }
    }
    fn expand_body(defs: &Defs, node: &Node) -> Stream {
        let type_name = &node.ident;
        let ident = Ident::new(type_name, Span::call_site());
        match &node.data {
            Data::Enum(variants) if variants.is_empty() => quote!(match *self {}),
            Data::Enum(variants) => {
                let arms = variants.iter().enumerate().map(|(i, (variant_name, fields))| {
                    let i = u8::try_from(i).unwrap();
                    let variant = Ident::new(variant_name, Span::call_site());
                    if fields.is_empty() {
                        quote! {
                            #ident::#variant => {
                                state.write_u8(#i);
                            }
                        }
                    } else {
                        let mut pats = Vec::new();
                        let mut hashes = Vec::new();
                        for (i, field) in fields.iter().enumerate() {
                            if skip(field) {
                                pats.push(format_ident!("_"));
                                continue;
                            }
                            let var = format_ident!("v{}", i);
                            let mut hashed_val = quote!(#var);
                            match field {
                                Type::Ext(ty) if ty == "Stream" => {
                                    hashed_val = quote!(StreamHelper(#hashed_val));
                                },
                                Type::Ext(ty) if ty == "Literal" => {
                                    hashed_val = quote!(#hashed_val.to_string());
                                },
                                _ => {},
                            }
                            hashes.push(quote! {
                                #hashed_val.hash(state);
                            });
                            pats.push(var);
                        }
                        let mut cfg = None;
                        if node.ident == "Expr" {
                            if let Type::Syn(ty) = &fields[0] {
                                if !lookup_node(defs, ty).feats.any.contains("derive") {
                                    cfg = Some(quote!(#[cfg(feature = "full")]));
                                }
                            }
                        }
                        quote! {
                            #cfg
                            #ident::#variant(#(#pats),*) => {
                                state.write_u8(#i);
                                #(#hashes)*
                            }
                        }
                    }
                });
                let nonexhaustive = if node.ident == "Expr" {
                    Some(quote! {
                        #[cfg(not(feature = "full"))]
                        _ => unreachable!(),
                    })
                } else {
                    None
                };
                quote! {
                    match self {
                        #(#arms)*
                        #nonexhaustive
                    }
                }
            },
            Data::Struct(fields) => fields
                .iter()
                .filter_map(|(f, ty)| {
                    if skip(ty) {
                        return None;
                    }
                    let ident = Ident::new(f, Span::call_site());
                    let mut val = quote!(self.#ident);
                    if let Type::Ext(ty) = ty {
                        if ty == "Stream" {
                            val = quote!(StreamHelper(&#val));
                        }
                    }
                    Some(quote! {
                        #val.hash(state);
                    })
                })
                .collect(),
            Data::Private => unreachable!(),
        }
    }
    fn skip(field_type: &Type) -> bool {
        match field_type {
            Type::Ext(ty) => ty == "Span",
            Type::Token(_) | Type::Group(_) => true,
            Type::Box(inner) => skip(inner),
            Type::Tuple(inner) => inner.iter().all(skip),
            _ => false,
        }
    }
}
mod snapshot {
    use super::*;
    const SOURCE: &str = "tests/debug/gen.rs";
    pub fn generate(xs: &Defs) -> Result<()> {
        let mut ys = Stream::new();
        for x in &xs.nodes {
            ys.extend(expand(xs, x));
        }
        write_to_file(
            SOURCE,
            quote! {
                #![allow(clippy::match_wildcard_for_single_variants)]
                use super::{Lite, Present};
                use ref_cast::RefCast;
                use std::fmt::{self, Debug, Display};
                #ys
            },
        )?;
        Ok(())
    }
    fn expand(xs: &Defs, node: &Node) -> Stream {
        let ident = Ident::new(&node.ident, Span::call_site());
        let body = expand_body(xs, node, &node.ident, &Own(quote!(self.value)));
        let formatter = match &node.data {
            Data::Enum(variants) if variants.is_empty() => quote!(_formatter),
            _ => quote!(formatter),
        };
        quote! {
            impl Debug for Lite<syn::#ident> {
                fn fmt(&self, #formatter: &mut fmt::Formatter) -> fmt::Result {
                    #body
                }
            }
        }
    }
    fn expand_body(defs: &Defs, node: &Node, name: &str, val: &Operand) -> Stream {
        let ident = Ident::new(&node.ident, Span::call_site());
        match &node.data {
            Data::Enum(variants) if variants.is_empty() => quote!(unreachable!()),
            Data::Enum(variants) => {
                let arms = variants.iter().map(|(v, fields)| {
                    let path = format!("{}::{}", name, v);
                    let variant = Ident::new(v, Span::call_site());
                    if fields.is_empty() {
                        quote! {
                            syn::#ident::#variant => formatter.write_str(#path),
                        }
                    } else if let Some(inner) = tree_enum(name, v, fields) {
                        let format = expand_body(defs, lookup_node(defs, inner), &path, &Borrow(quote!(_val)));
                        quote! {
                            syn::#ident::#variant(_val) => {
                                #format
                            }
                        }
                    } else if fields.len() == 1 {
                        let val = quote!(_val);
                        let format = if variant == "Verbatim" {
                            Some(quote! {
                                formatter.write_str("(`")?;
                                Display::fmt(#val, formatter)?;
                                formatter.write_str("`)")?;
                            })
                        } else {
                            let ty = &fields[0];
                            format_field(&Borrow(val), ty).map(|format| {
                                quote! {
                                    formatter.write_str("(")?;
                                    Debug::fmt(#format, formatter)?;
                                    formatter.write_str(")")?;
                                }
                            })
                        };
                        quote! {
                            syn::#ident::#variant(_val) => {
                                formatter.write_str(#path)?;
                                #format
                                Ok(())
                            }
                        }
                    } else {
                        let pats = (0..fields.len()).map(|i| format_ident!("_v{}", i));
                        let fields = fields.iter().enumerate().filter_map(|(i, ty)| {
                            let index = format_ident!("_v{}", i);
                            let val = quote!(#index);
                            let format = format_field(&Borrow(val), ty)?;
                            Some(quote! {
                                formatter.field(#format);
                            })
                        });
                        quote! {
                            syn::#ident::#variant(#(#pats),*) => {
                                let mut formatter = formatter.debug_tuple(#path);
                                #(#fields)*
                                formatter.finish()
                            }
                        }
                    }
                });
                let nonexhaustive = if node.exhaustive {
                    None
                } else {
                    Some(quote!(_ => unreachable!()))
                };
                let val = val.ref_tokens();
                quote! {
                    match #val {
                        #(#arms)*
                        #nonexhaustive
                    }
                }
            },
            Data::Struct(fields) => {
                let fields = fields.iter().filter_map(|(f, ty)| {
                    let ident = Ident::new(f, Span::call_site());
                    if let Type::Option(ty) = ty {
                        Some(if let Some(format) = format_field(&Own(quote!(self.0)), ty) {
                            let val = val.tokens();
                            let ty = rust_type(ty);
                            quote! {
                                if let Some(val) = &#val.#ident {
                                    #[derive(RefCast)]
                                    #[repr(transparent)]
                                    struct Print(#ty);
                                    impl Debug for Print {
                                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                                            formatter.write_str("Some(")?;
                                            Debug::fmt(#format, formatter)?;
                                            formatter.write_str(")")?;
                                            Ok(())
                                        }
                                    }
                                    formatter.field(#f, Print::ref_cast(val));
                                }
                            }
                        } else {
                            let val = val.tokens();
                            quote! {
                                if #val.#ident.is_some() {
                                    formatter.field(#f, &Present);
                                }
                            }
                        })
                    } else {
                        let val = val.tokens();
                        let inner = Own(quote!(#val.#ident));
                        let format = format_field(&inner, ty)?;
                        let mut call = quote! {
                            formatter.field(#f, #format);
                        };
                        if let Type::Vec(_) | Type::Punct(_) = ty {
                            call = quote! {
                                if !#val.#ident.is_empty() {
                                    #call
                                }
                            };
                        } else if let Type::Syn(inner) = ty {
                            for node in &defs.nodes {
                                if node.ident == *inner {
                                    if let Data::Enum(variants) = &node.data {
                                        if variants.get("None").map_or(false, Vec::is_empty) {
                                            let ty = rust_type(ty);
                                            call = quote! {
                                                match #val.#ident {
                                                    #ty::None => {}
                                                    _ => { #call }
                                                }
                                            };
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        Some(call)
                    }
                });
                quote! {
                    let mut formatter = formatter.debug_struct(#name);
                    #(#fields)*
                    formatter.finish()
                }
            },
            Data::Private => {
                if node.ident == "LitInt" || node.ident == "LitFloat" {
                    let val = val.ref_tokens();
                    quote! {
                        write!(formatter, "{}", #val)
                    }
                } else {
                    let val = val.tokens();
                    quote! {
                        write!(formatter, "{:?}", #val.value())
                    }
                }
            },
        }
    }
    fn tree_enum<'a>(outer: &str, inner: &str, fields: &'a [Type]) -> Option<&'a str> {
        if fields.len() != 1 {
            return None;
        }
        const WHITELIST: &[(&str, &str)] = &[
            ("Meta", "Path"),
            ("PathArguments", "AngleBracketed"),
            ("PathArguments", "Parenthesized"),
            ("Stmt", "Local"),
            ("TypeParamBound", "Lifetime"),
            ("Visibility", "Public"),
            ("Visibility", "Restricted"),
        ];
        match &fields[0] {
            Type::Syn(ty) if WHITELIST.contains(&(outer, inner)) || outer.to_owned() + inner == *ty => Some(ty),
            _ => None,
        }
    }
    fn rust_type(ty: &Type) -> Stream {
        match ty {
            Type::Syn(ty) => {
                let ident = Ident::new(ty, Span::call_site());
                quote!(syn::#ident)
            },
            Type::Std(ty) => {
                let ident = Ident::new(ty, Span::call_site());
                quote!(#ident)
            },
            Type::Ext(ty) => {
                let ident = Ident::new(ty, Span::call_site());
                quote!(syn::pm2::#ident)
            },
            Type::Token(ty) | Type::Group(ty) => {
                let ident = Ident::new(ty, Span::call_site());
                quote!(syn::token::#ident)
            },
            Type::Punct(ty) => {
                let element = rust_type(&ty.typ);
                let punct = Ident::new(&ty.punct, Span::call_site());
                quote!(syn::punctuated::Punctuated<#element, #punct>)
            },
            Type::Option(ty) => {
                let inner = rust_type(ty);
                quote!(Option<#inner>)
            },
            Type::Box(ty) => {
                let inner = rust_type(ty);
                quote!(Box<#inner>)
            },
            Type::Vec(ty) => {
                let inner = rust_type(ty);
                quote!(Vec<#inner>)
            },
            Type::Tuple(ty) => {
                let inner = ty.iter().map(rust_type);
                quote!((#(#inner,)*))
            },
        }
    }
    fn format_field(val: &Operand, ty: &Type) -> Option<Stream> {
        if !is_printable(ty) {
            return None;
        }
        let format = match ty {
            Type::Option(ty) => {
                if let Some(format) = format_field(&Borrow(quote!(_val)), ty) {
                    let ty = rust_type(ty);
                    let val = val.ref_tokens();
                    quote!({
                        #[derive(RefCast)]
                        #[repr(transparent)]
                        struct Print(Option<#ty>);
                        impl Debug for Print {
                            fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                                match &self.0 {
                                    Some(_val) => {
                                        formatter.write_str("Some(")?;
                                        Debug::fmt(#format, formatter)?;
                                        formatter.write_str(")")?;
                                        Ok(())
                                    }
                                    None => formatter.write_str("None"),
                                }
                            }
                        }
                        Print::ref_cast(#val)
                    })
                } else {
                    let val = val.tokens();
                    quote! {
                        &super::Option { present: #val.is_some() }
                    }
                }
            },
            Type::Tuple(ty) => {
                let printable: Vec<Stream> = ty
                    .iter()
                    .enumerate()
                    .filter_map(|(i, ty)| {
                        let index = Index::from(i);
                        let val = val.tokens();
                        let inner = Own(quote!(#val.#index));
                        format_field(&inner, ty)
                    })
                    .collect();
                if printable.len() == 1 {
                    printable.into_iter().next().unwrap()
                } else {
                    quote! {
                        &(#(#printable),*)
                    }
                }
            },
            _ => {
                let val = val.ref_tokens();
                quote! { Lite(#val) }
            },
        };
        Some(format)
    }
    fn is_printable(ty: &Type) -> bool {
        match ty {
            Type::Ext(name) => name != "Span",
            Type::Box(ty) => is_printable(ty),
            Type::Tuple(ty) => ty.iter().any(is_printable),
            Type::Token(_) | Type::Group(_) => false,
            Type::Syn(_) | Type::Std(_) | Type::Punct(_) | Type::Option(_) | Type::Vec(_) => true,
        }
    }
}

fn under_name(x: &str) -> Ident {
    Ident::new(&x.to_snake_case(), Span::call_site())
}

const TERM_TYPES: &[&str] = &["Span", "Ident"];
fn traverse(ds: &Defs, f: fn(&mut Stream, &mut Stream, &Node, &Defs)) -> (Stream, Stream) {
    let mut traits = Stream::new();
    let mut impls = Stream::new();
    let mut ys = ds.nodes.clone();
    for &y in TERM_TYPES {
        ys.push(Node {
            ident: y.to_owned(),
            feats: Features::default(),
            data: Data::Private,
            exhaustive: true,
        });
    }
    ys.sort_by(|a, b| a.ident.cmp(&b.ident));
    for y in ys {
        let x = config(&y.feats);
        traits.extend(x.clone());
        impls.extend(x);
        f(&mut traits, &mut impls, &y, ds);
    }
    (traits, impls)
}

mod fold {
    use super::*;
    const SOURCE: &str = "src/codegen/fold.rs";
    pub fn generate(xs: &Defs) -> Result<()> {
        let (traits, impls) = traverse(xs, node);
        write_to_file(
            SOURCE,
            quote! {
                #![allow(unreachable_code, unused_variables)]
                #![allow(clippy::match_wildcard_for_single_variants, clippy::needless_match)]
                use crate::*;
                use syn::pm2::Span;
                pub trait Fold {
                    #traits
                }
                #impls
            },
        )?;
        Ok(())
    }
    fn node(traits: &mut Stream, impls: &mut Stream, s: &Node, defs: &Defs) {
        let under_name = under_name(&s.ident);
        let ty = Ident::new(&s.ident, Span::call_site());
        let fold_fn = format_ident!("fold_{}", under_name);
        let mut fold_impl = Stream::new();
        match &s.data {
            Data::Enum(variants) => {
                let mut fold_variants = Stream::new();
                for (variant, fields) in variants {
                    let variant_ident = Ident::new(variant, Span::call_site());
                    if fields.is_empty() {
                        fold_variants.extend(quote! {
                            #ty::#variant_ident => {
                                #ty::#variant_ident
                            }
                        });
                    } else {
                        let mut bind_fold_fields = Stream::new();
                        let mut fold_fields = Stream::new();
                        for (idx, ty) in fields.iter().enumerate() {
                            let binding = format_ident!("_binding_{}", idx);
                            bind_fold_fields.extend(quote! {
                                #binding,
                            });
                            let owned_binding = quote!(#binding);
                            fold_fields.extend(visit(ty, &s.feats, defs, &owned_binding).unwrap_or(owned_binding));
                            fold_fields.extend(quote!(,));
                        }
                        fold_variants.extend(quote! {
                            #ty::#variant_ident(#bind_fold_fields) => {
                                #ty::#variant_ident(
                                    #fold_fields
                                )
                            }
                        });
                    }
                }
                fold_impl.extend(quote! {
                    match node {
                        #fold_variants
                    }
                });
            },
            Data::Struct(fields) => {
                let mut fold_fields = Stream::new();
                for (field, ty) in fields {
                    let id = Ident::new(field, Span::call_site());
                    let ref_toks = quote!(node.#id);
                    let fold = visit(ty, &s.feats, defs, &ref_toks).unwrap_or(ref_toks);
                    fold_fields.extend(quote! {
                        #id: #fold,
                    });
                }
                if fields.is_empty() {
                    if ty == "Ident" {
                        fold_impl.extend(quote! {
                            let mut node = node;
                            let span = f.fold_span(node.span());
                            node.set_span(span);
                        });
                    }
                    fold_impl.extend(quote! {
                        node
                    });
                } else {
                    fold_impl.extend(quote! {
                        #ty {
                            #fold_fields
                        }
                    });
                }
            },
            Data::Private => {
                if ty == "Ident" {
                    fold_impl.extend(quote! {
                        let mut node = node;
                        let span = f.fold_span(node.span());
                        node.set_span(span);
                    });
                }
                fold_impl.extend(quote! {
                    node
                });
            },
        }
        let fold_span_only = s.data == Data::Private && !TERM_TYPES.contains(&s.ident.as_str());
        if fold_span_only {
            fold_impl = quote! {
                let span = f.fold_span(node.span());
                let mut node = node;
                node.set_span(span);
                node
            };
        }
        traits.extend(quote! {
            fn #fold_fn(&mut self, i: #ty) -> #ty {
                #fold_fn(self, i)
            }
        });
        impls.extend(quote! {
            pub fn #fold_fn<F>(f: &mut F, node: #ty) -> #ty
            where
                F: Fold + ?Sized,
            {
                #fold_impl
            }
        });
    }
    fn visit(ty: &Type, features: &Features, defs: &Defs, name: &Stream) -> Option<Stream> {
        match ty {
            Type::Box(t) => {
                let res = visit(t, features, defs, &quote!(*#name))?;
                Some(quote! {
                    Box::new(#res)
                })
            },
            Type::Vec(t) => {
                let operand = quote!(it);
                let val = visit(t, features, defs, &operand)?;
                Some(quote! {
                    FoldHelper::lift(#name, |it| #val)
                })
            },
            Type::Punct(p) => {
                let operand = quote!(it);
                let val = visit(&p.typ, features, defs, &operand)?;
                Some(quote! {
                    FoldHelper::lift(#name, |it| #val)
                })
            },
            Type::Option(t) => {
                let it = quote!(it);
                let val = visit(t, features, defs, &it)?;
                Some(quote! {
                    (#name).map(|it| #val)
                })
            },
            Type::Tuple(t) => {
                let mut code = Stream::new();
                for (i, elem) in t.iter().enumerate() {
                    let i = Index::from(i);
                    let it = quote!((#name).#i);
                    let val = visit(elem, features, defs, &it).unwrap_or(it);
                    code.extend(val);
                    code.extend(quote!(,));
                }
                Some(quote! {
                    (#code)
                })
            },
            Type::Syn(t) => {
                fn requires_full(features: &Features) -> bool {
                    features.any.contains("full") && features.any.len() == 1
                }
                let mut res = simple_visit(t, name);
                let target = defs.nodes.iter().find(|ty| ty.ident == *t).unwrap();
                if requires_full(&target.feats) && !requires_full(features) {
                    res = quote!(full!(#res));
                }
                Some(res)
            },
            Type::Ext(t) if TERM_TYPES.contains(&&t[..]) => Some(simple_visit(t, name)),
            Type::Ext(_) | Type::Std(_) | Type::Token(_) | Type::Group(_) => None,
        }
    }
    fn simple_visit(item: &str, name: &Stream) -> Stream {
        let ident = under_name(item);
        let method = format_ident!("fold_{}", ident);
        quote! {
            f.#method(#name)
        }
    }
}
mod visit {
    use super::*;
    const SOURCE: &str = "src/codegen/visit.rs";
    pub fn generate(xs: &Defs) -> Result<()> {
        let (traits, impls) = traverse(xs, node);
        write_to_file(
            SOURCE,
            quote! {
                #![allow(unused_variables)]
                use crate::*;
                macro_rules! skip {
                    ($($tt:tt)*) => {};
                }
                pub trait Visit<'ast> {
                    #traits
                }
                #impls
            },
        )?;
        Ok(())
    }
    fn node(traits: &mut Stream, impls: &mut Stream, s: &Node, defs: &Defs) {
        let under_name = under_name(&s.ident);
        let ty = Ident::new(&s.ident, Span::call_site());
        let visit_fn = format_ident!("visit_{}", under_name);
        let mut visit_impl = Stream::new();
        match &s.data {
            Data::Enum(variants) if variants.is_empty() => {
                visit_impl.extend(quote! {
                    match *node {}
                });
            },
            Data::Enum(variants) => {
                let mut visit_variants = Stream::new();
                for (variant, fields) in variants {
                    let variant_ident = Ident::new(variant, Span::call_site());
                    if fields.is_empty() {
                        visit_variants.extend(quote! {
                            #ty::#variant_ident => {}
                        });
                    } else {
                        let mut bind_visit_fields = Stream::new();
                        let mut visit_fields = Stream::new();
                        for (idx, ty) in fields.iter().enumerate() {
                            let binding = format_ident!("_binding_{}", idx);
                            bind_visit_fields.extend(quote! {
                                #binding,
                            });
                            let borrowed_binding = Borrow(quote!(#binding));
                            visit_fields.extend(
                                visit(ty, &s.feats, defs, &borrowed_binding)
                                    .unwrap_or_else(|| noop_visit(&borrowed_binding)),
                            );
                            visit_fields.extend(quote!(;));
                        }
                        visit_variants.extend(quote! {
                            #ty::#variant_ident(#bind_visit_fields) => {
                                #visit_fields
                            }
                        });
                    }
                }
                visit_impl.extend(quote! {
                    match node {
                        #visit_variants
                    }
                });
            },
            Data::Struct(fields) => {
                for (field, ty) in fields {
                    let id = Ident::new(field, Span::call_site());
                    let ref_toks = Own(quote!(node.#id));
                    let visit_field = visit(ty, &s.feats, defs, &ref_toks).unwrap_or_else(|| noop_visit(&ref_toks));
                    visit_impl.extend(quote! {
                        #visit_field;
                    });
                }
            },
            Data::Private => {
                if ty == "Ident" {
                    visit_impl.extend(quote! {
                        v.visit_span(&node.span());
                    });
                }
            },
        }
        let ast_lifetime = if s.ident == "Span" { None } else { Some(quote!('ast)) };
        traits.extend(quote! {
            fn #visit_fn(&mut self, i: &#ast_lifetime #ty) {
                #visit_fn(self, i);
            }
        });
        impls.extend(quote! {
            pub fn #visit_fn<'ast, V>(v: &mut V, node: &#ast_lifetime #ty)
            where
                V: Visit<'ast> + ?Sized,
            {
                #visit_impl
            }
        });
    }
    fn visit(ty: &Type, features: &Features, defs: &Defs, name: &Operand) -> Option<Stream> {
        match ty {
            Type::Box(t) => {
                let name = name.owned_tokens();
                visit(t, features, defs, &Own(quote!(*#name)))
            },
            Type::Vec(t) => {
                let operand = Borrow(quote!(it));
                let val = visit(t, features, defs, &operand)?;
                let name = name.ref_tokens();
                Some(quote! {
                    for it in #name {
                        #val;
                    }
                })
            },
            Type::Punct(p) => {
                let operand = Borrow(quote!(it));
                let val = visit(&p.typ, features, defs, &operand)?;
                let name = name.ref_tokens();
                Some(quote! {
                    for el in Punctuated::pairs(#name) {
                        let it = el.value();
                        #val;
                    }
                })
            },
            Type::Option(t) => {
                let it = Borrow(quote!(it));
                let val = visit(t, features, defs, &it)?;
                let name = name.ref_tokens();
                Some(quote! {
                    if let Some(it) = #name {
                        #val;
                    }
                })
            },
            Type::Tuple(t) => {
                let mut code = Stream::new();
                for (i, elem) in t.iter().enumerate() {
                    let name = name.tokens();
                    let i = Index::from(i);
                    let it = Own(quote!((#name).#i));
                    let val = visit(elem, features, defs, &it).unwrap_or_else(|| noop_visit(&it));
                    code.extend(val);
                    code.extend(quote!(;));
                }
                Some(code)
            },
            Type::Syn(t) => {
                fn requires_full(features: &Features) -> bool {
                    features.any.contains("full") && features.any.len() == 1
                }
                let mut res = simple_visit(t, name);
                let target = defs.nodes.iter().find(|ty| ty.ident == *t).unwrap();
                if requires_full(&target.feats) && !requires_full(features) {
                    res = quote!(full!(#res));
                }
                Some(res)
            },
            Type::Ext(t) if TERM_TYPES.contains(&&t[..]) => Some(simple_visit(t, name)),
            Type::Ext(_) | Type::Std(_) | Type::Token(_) | Type::Group(_) => None,
        }
    }
    fn simple_visit(item: &str, name: &Operand) -> Stream {
        let ident = under_name(item);
        let method = format_ident!("visit_{}", ident);
        let name = name.ref_tokens();
        quote! {
            v.#method(#name)
        }
    }
    fn noop_visit(name: &Operand) -> Stream {
        let name = name.tokens();
        quote! {
            skip!(#name)
        }
    }
}
mod visit_mut {
    use super::*;
    const SOURCE: &str = "src/codegen/visit_mut.rs";
    pub fn generate(defs: &Defs) -> Result<()> {
        let (traits, impls) = traverse(defs, node);
        write_to_file(
            SOURCE,
            quote! {
                #![allow(unused_variables)]
                #[cfg(any(feature = "full", feature = "derive"))]
                use crate::punctuated::Punctuated;
                use crate::*;
                use syn::pm2::Span;
                macro_rules! skip {
                    ($($tt:tt)*) => {};
                }
                pub trait VisitMut {
                    #traits
                }
                #impls
            },
        )?;
        Ok(())
    }
    fn node(traits: &mut Stream, impls: &mut Stream, s: &Node, defs: &Defs) {
        let under_name = under_name(&s.ident);
        let ty = Ident::new(&s.ident, Span::call_site());
        let visit_mut_fn = format_ident!("visit_{}_mut", under_name);
        let mut visit_mut_impl = Stream::new();
        match &s.data {
            Data::Enum(variants) if variants.is_empty() => {
                visit_mut_impl.extend(quote! {
                    match *node {}
                });
            },
            Data::Enum(variants) => {
                let mut visit_mut_variants = Stream::new();
                for (variant, fields) in variants {
                    let variant_ident = Ident::new(variant, Span::call_site());
                    if fields.is_empty() {
                        visit_mut_variants.extend(quote! {
                            #ty::#variant_ident => {}
                        });
                    } else {
                        let mut bind_visit_mut_fields = Stream::new();
                        let mut visit_mut_fields = Stream::new();
                        for (idx, ty) in fields.iter().enumerate() {
                            let binding = format_ident!("_binding_{}", idx);
                            bind_visit_mut_fields.extend(quote! {
                                #binding,
                            });
                            let borrowed_binding = Borrow(quote!(#binding));
                            visit_mut_fields.extend(
                                visit(ty, &s.feats, defs, &borrowed_binding)
                                    .unwrap_or_else(|| noop_visit(&borrowed_binding)),
                            );
                            visit_mut_fields.extend(quote!(;));
                        }
                        visit_mut_variants.extend(quote! {
                            #ty::#variant_ident(#bind_visit_mut_fields) => {
                                #visit_mut_fields
                            }
                        });
                    }
                }
                visit_mut_impl.extend(quote! {
                    match node {
                        #visit_mut_variants
                    }
                });
            },
            Data::Struct(fields) => {
                for (field, ty) in fields {
                    let id = Ident::new(field, Span::call_site());
                    let ref_toks = Own(quote!(node.#id));
                    let visit_mut_field = visit(ty, &s.feats, defs, &ref_toks).unwrap_or_else(|| noop_visit(&ref_toks));
                    visit_mut_impl.extend(quote! {
                        #visit_mut_field;
                    });
                }
            },
            Data::Private => {
                if ty == "Ident" {
                    visit_mut_impl.extend(quote! {
                        let mut span = node.span();
                        v.visit_span_mut(&mut span);
                        node.set_span(span);
                    });
                }
            },
        }
        traits.extend(quote! {
            fn #visit_mut_fn(&mut self, i: &mut #ty) {
                #visit_mut_fn(self, i);
            }
        });
        impls.extend(quote! {
            pub fn #visit_mut_fn<V>(v: &mut V, node: &mut #ty)
            where
                V: Visitor + ?Sized,
            {
                #visit_mut_impl
            }
        });
    }
    fn visit(ty: &Type, features: &Features, defs: &Defs, name: &Operand) -> Option<Stream> {
        match ty {
            Type::Box(t) => {
                let name = name.owned_tokens();
                visit(t, features, defs, &Own(quote!(*#name)))
            },
            Type::Vec(t) => {
                let operand = Borrow(quote!(it));
                let val = visit(t, features, defs, &operand)?;
                let name = name.ref_mut_tokens();
                Some(quote! {
                    for it in #name {
                        #val;
                    }
                })
            },
            Type::Punct(p) => {
                let operand = Borrow(quote!(it));
                let val = visit(&p.typ, features, defs, &operand)?;
                let name = name.ref_mut_tokens();
                Some(quote! {
                    for mut el in Punctuated::pairs_mut(#name) {
                        let it = el.value_mut();
                        #val;
                    }
                })
            },
            Type::Option(t) => {
                let it = Borrow(quote!(it));
                let val = visit(t, features, defs, &it)?;
                let name = name.ref_mut_tokens();
                Some(quote! {
                    if let Some(it) = #name {
                        #val;
                    }
                })
            },
            Type::Tuple(t) => {
                let mut code = Stream::new();
                for (i, elem) in t.iter().enumerate() {
                    let name = name.tokens();
                    let i = Index::from(i);
                    let it = Own(quote!((#name).#i));
                    let val = visit(elem, features, defs, &it).unwrap_or_else(|| noop_visit(&it));
                    code.extend(val);
                    code.extend(quote!(;));
                }
                Some(code)
            },
            Type::Syn(t) => {
                fn requires_full(features: &Features) -> bool {
                    features.any.contains("full") && features.any.len() == 1
                }
                let mut res = simple_visit(t, name);
                let target = defs.nodes.iter().find(|ty| ty.ident == *t).unwrap();
                if requires_full(&target.feats) && !requires_full(features) {
                    res = quote!(full!(#res));
                }
                Some(res)
            },
            Type::Ext(t) if TERM_TYPES.contains(&&t[..]) => Some(simple_visit(t, name)),
            Type::Ext(_) | Type::Std(_) | Type::Token(_) | Type::Group(_) => None,
        }
    }
    fn simple_visit(item: &str, name: &Operand) -> Stream {
        let ident = under_name(item);
        let method = format_ident!("visit_{}_mut", ident);
        let name = name.ref_mut_tokens();
        quote! {
            v.#method(#name)
        }
    }
    fn noop_visit(name: &Operand) -> Stream {
        let name = name.tokens();
        quote! {
            skip!(#name)
        }
    }
}

fn main() -> anyhow::Result<()> {
    fn generate(xs: &Defs) -> Result<()> {
        let mut y = serde_json::to_string_pretty(&xs)?;
        y.push('\n');
        let xs2: Defs = serde_json::from_str(&y)?;
        assert_eq!(*xs, xs2);
        let x = get_rel_path("syn.json");
        fs::write(x, y)?;
        Ok(())
    }
    color_backtrace::install();
    let ys = parse::parse()?;
    clone::generate(&ys)?;
    debug::generate(&ys)?;
    eq::generate(&ys)?;
    fold::generate(&ys)?;
    generate(&ys)?;
    hash::generate(&ys)?;
    visit::generate(&ys)?;
    visit_mut::generate(&ys)?;
    snapshot::generate(&ys)?;
    Ok(())
}
