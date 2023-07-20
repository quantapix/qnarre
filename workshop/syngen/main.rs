#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_like_matches_macro,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

extern crate proc_macro as pm;

mod cfg {
    use json::Features;
    use syn::pm2::Stream;
    use syn::quote::quote;
    pub fn features(features: &Features) -> Stream {
        let features = &features.any;
        match features.len() {
            0 => quote!(),
            1 => quote!(#[cfg(feature = #(#features)*)]),
            _ => quote!(#[cfg(any(#(feature = #features),*))]),
        }
    }
}
mod clone {
    use crate::{cfg, file, lookup};
    use anyhow::Result;
    use json::{Data, Definitions, Node, Type};
    use syn::pm2::{Ident, Span, Stream};
    use syn::quote::{format_ident, quote};
    const CLONE_SRC: &str = "src/gen/clone.rs";
    fn expand_impl_body(defs: &Definitions, node: &Node) -> Stream {
        let type_name = &node.ident;
        let ident = Ident::new(type_name, Span::call_site());
        match &node.data {
            Data::Enum(variants) if variants.is_empty() => quote!(match *self {}),
            Data::Enum(variants) => {
                let arms = variants.iter().map(|(variant_name, fields)| {
                    let variant = Ident::new(variant_name, Span::call_site());
                    if fields.is_empty() {
                        quote! {
                            #ident::#variant => #ident::#variant,
                        }
                    } else {
                        let mut pats = Vec::new();
                        let mut clones = Vec::new();
                        for i in 0..fields.len() {
                            let pat = format_ident!("v{}", i);
                            clones.push(quote!(#pat.clone()));
                            pats.push(pat);
                        }
                        let mut cfg = None;
                        if node.ident == "Expr" {
                            if let Type::Syn(ty) = &fields[0] {
                                if !lookup::node(defs, ty).features.any.contains("derive") {
                                    cfg = Some(quote!(#[cfg(feature = "full")]));
                                }
                            }
                        }
                        quote! {
                            #cfg
                            #ident::#variant(#(#pats),*) => #ident::#variant(#(#clones),*),
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
            Data::Struct(fields) => {
                let fields = fields.keys().map(|f| {
                    let ident = Ident::new(f, Span::call_site());
                    quote! {
                        #ident: self.#ident.clone(),
                    }
                });
                quote!(#ident { #(#fields)* })
            },
            Data::Private => unreachable!(),
        }
    }
    fn expand_impl(defs: &Definitions, node: &Node) -> Stream {
        let manual_clone = node.data == Data::Private || node.ident == "Lifetime";
        if manual_clone {
            return Stream::new();
        }
        let ident = Ident::new(&node.ident, Span::call_site());
        let cfg_features = cfg::features(&node.features);
        let copy = node.ident == "AttrStyle"
            || node.ident == "BinOp"
            || node.ident == "RangeLimits"
            || node.ident == "TraitBoundModifier"
            || node.ident == "UnOp";
        if copy {
            return quote! {
                #cfg_features
                #[cfg_attr(doc_cfg, doc(cfg(feature = "clone-impls")))]
                impl Copy for #ident {}
                #cfg_features
                #[cfg_attr(doc_cfg, doc(cfg(feature = "clone-impls")))]
                impl Clone for #ident {
                    fn clone(&self) -> Self {
                        *self
                    }
                }
            };
        }
        let body = expand_impl_body(defs, node);
        quote! {
            #cfg_features
            #[cfg_attr(doc_cfg, doc(cfg(feature = "clone-impls")))]
            impl Clone for #ident {
                fn clone(&self) -> Self {
                    #body
                }
            }
        }
    }
    pub fn generate(defs: &Definitions) -> Result<()> {
        let mut impls = Stream::new();
        for node in &defs.types {
            impls.extend(expand_impl(defs, node));
        }
        file::write(
            CLONE_SRC,
            quote! {
                #![allow(clippy::clone_on_copy, clippy::expl_impl_clone_on_copy)]
                use crate::*;
                #impls
            },
        )?;
        Ok(())
    }
}
mod debug {
    use crate::{cfg, file, lookup};
    use anyhow::Result;
    use json::{Data, Definitions, Node, Type};
    use std::collections::BTreeSet as Set;
    use syn::pm2::{Ident, Span, Stream};
    use syn::quote::{format_ident, quote};
    const DEBUG_SRC: &str = "src/gen/debug.rs";
    fn syntax_tree_enum<'a>(enum_name: &str, variant_name: &str, fields: &'a [Type]) -> Option<&'a str> {
        if fields.len() != 1 {
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
        match &fields[0] {
            Type::Syn(ty)
                if WHITELIST.contains(&(enum_name, variant_name)) || enum_name.to_owned() + variant_name == *ty =>
            {
                Some(ty)
            },
            _ => None,
        }
    }
    fn expand_impl_body(defs: &Definitions, node: &Node, syntax_tree_variants: &Set<&str>) -> Stream {
        let type_name = &node.ident;
        let ident = Ident::new(type_name, Span::call_site());
        let is_syntax_tree_variant = syntax_tree_variants.contains(type_name.as_str());
        let body = match &node.data {
            Data::Enum(variants) if variants.is_empty() => quote!(match *self {}),
            Data::Enum(variants) => {
                assert!(!is_syntax_tree_variant);
                let arms = variants.iter().map(|(variant_name, fields)| {
                    let variant = Ident::new(variant_name, Span::call_site());
                    if fields.is_empty() {
                        quote! {
                            #ident::#variant => formatter.write_str(#variant_name),
                        }
                    } else {
                        let mut cfg = None;
                        if node.ident == "Expr" {
                            if let Type::Syn(ty) = &fields[0] {
                                if !lookup::node(defs, ty).features.any.contains("derive") {
                                    cfg = Some(quote!(#[cfg(feature = "full")]));
                                }
                            }
                        }
                        if syntax_tree_enum(type_name, variant_name, fields).is_some() {
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
                let nonexhaustive = if node.ident == "Expr" {
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
    fn expand_impl(defs: &Definitions, node: &Node, syntax_tree_variants: &Set<&str>) -> Stream {
        let manual_debug = node.data == Data::Private || node.ident == "LitBool";
        if manual_debug {
            return Stream::new();
        }
        let ident = Ident::new(&node.ident, Span::call_site());
        let cfg_features = cfg::features(&node.features);
        let body = expand_impl_body(defs, node, syntax_tree_variants);
        let formatter = match &node.data {
            Data::Enum(variants) if variants.is_empty() => quote!(_formatter),
            _ => quote!(formatter),
        };
        quote! {
            #cfg_features
            #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
            impl Debug for #ident {
                fn fmt(&self, #formatter: &mut fmt::Formatter) -> fmt::Result {
                    #body
                }
            }
        }
    }
    pub fn generate(defs: &Definitions) -> Result<()> {
        let mut syntax_tree_variants = Set::new();
        for node in &defs.types {
            if let Data::Enum(variants) = &node.data {
                let enum_name = &node.ident;
                for (variant_name, fields) in variants {
                    if let Some(inner) = syntax_tree_enum(enum_name, variant_name, fields) {
                        syntax_tree_variants.insert(inner);
                    }
                }
            }
        }
        let mut impls = Stream::new();
        for node in &defs.types {
            impls.extend(expand_impl(defs, node, &syntax_tree_variants));
        }
        file::write(
            DEBUG_SRC,
            quote! {
                use crate::*;
                use std::fmt::{self, Debug};
                #impls
            },
        )?;
        Ok(())
    }
}
mod eq {
    use crate::{cfg, file, lookup};
    use anyhow::Result;
    use json::{Data, Definitions, Node, Type};
    use syn::pm2::{Ident, Span, Stream};
    use syn::quote::{format_ident, quote};
    const EQ_SRC: &str = "src/gen/eq.rs";
    fn always_eq(field_type: &Type) -> bool {
        match field_type {
            Type::Ext(ty) => ty == "Span",
            Type::Token(_) | Type::Group(_) => true,
            Type::Box(inner) => always_eq(inner),
            Type::Tuple(inner) => inner.iter().all(always_eq),
            _ => false,
        }
    }
    fn expand_impl_body(defs: &Definitions, node: &Node) -> Stream {
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
                                if !lookup::node(defs, ty).features.any.contains("derive") {
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
    fn expand_impl(defs: &Definitions, node: &Node) -> Stream {
        if node.ident == "Member" || node.ident == "Index" || node.ident == "Lifetime" {
            return Stream::new();
        }
        let ident = Ident::new(&node.ident, Span::call_site());
        let cfg_features = cfg::features(&node.features);
        let eq = quote! {
            #cfg_features
            #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
            impl Eq for #ident {}
        };
        let manual_partial_eq = node.data == Data::Private;
        if manual_partial_eq {
            return eq;
        }
        let body = expand_impl_body(defs, node);
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
    pub fn generate(defs: &Definitions) -> Result<()> {
        let mut impls = Stream::new();
        for node in &defs.types {
            impls.extend(expand_impl(defs, node));
        }
        file::write(
            EQ_SRC,
            quote! {
                #[cfg(any(feature = "derive", feature = "full"))]
                use crate::tt::StreamHelper;
                use crate::*;
                #impls
            },
        )?;
        Ok(())
    }
}
mod file {
    use crate::workspace_path;
    use anyhow::Result;
    use std::fs;
    use std::io::Write;
    use std::path::Path;
    use syn::pm2::Stream;
    pub fn write(relative_to_workspace_root: impl AsRef<Path>, content: Stream) -> Result<()> {
        let mut formatted = Vec::new();
        writeln!(formatted, "// This file is @generated by syn-internal-codegen.")?;
        writeln!(formatted, "// It is not intended for manual editing.")?;
        writeln!(formatted)?;
        let syntax_tree: syn::File = syn::parse2(content).unwrap();
        let pretty = prettyplease::unparse(&syntax_tree);
        write!(formatted, "{}", pretty)?;
        let path = workspace_path::get(relative_to_workspace_root);
        if path.is_file() && fs::read(&path)? == formatted {
            return Ok(());
        }
        fs::write(path, formatted)?;
        Ok(())
    }
}
mod fold {
    use crate::{file, full, gen};
    use anyhow::Result;
    use json::{Data, Definitions, Features, Node, Type};
    use syn::pm2::{Ident, Span, Stream};
    use syn::quote::{format_ident, quote};
    use syn::Index;
    const FOLD_SRC: &str = "src/gen/fold.rs";
    fn simple_visit(item: &str, name: &Stream) -> Stream {
        let ident = gen::under_name(item);
        let method = format_ident!("fold_{}", ident);
        quote! {
            f.#method(#name)
        }
    }
    fn visit(ty: &Type, features: &Features, defs: &Definitions, name: &Stream) -> Option<Stream> {
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
            Type::Punctuated(p) => {
                let operand = quote!(it);
                let val = visit(&p.element, features, defs, &operand)?;
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
                let target = defs.types.iter().find(|ty| ty.ident == *t).unwrap();
                if requires_full(&target.features) && !requires_full(features) {
                    res = quote!(full!(#res));
                }
                Some(res)
            },
            Type::Ext(t) if gen::TERMINAL_TYPES.contains(&&t[..]) => Some(simple_visit(t, name)),
            Type::Ext(_) | Type::Std(_) | Type::Token(_) | Type::Group(_) => None,
        }
    }
    fn node(traits: &mut Stream, impls: &mut Stream, s: &Node, defs: &Definitions) {
        let under_name = gen::under_name(&s.ident);
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
                            fold_fields.extend(visit(ty, &s.features, defs, &owned_binding).unwrap_or(owned_binding));
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
                    let fold = visit(ty, &s.features, defs, &ref_toks).unwrap_or(ref_toks);
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
        let fold_span_only = s.data == Data::Private && !gen::TERMINAL_TYPES.contains(&s.ident.as_str());
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
    pub fn generate(defs: &Definitions) -> Result<()> {
        let (traits, impls) = gen::traverse(defs, node);
        let full_macro = full::get_macro();
        file::write(
            FOLD_SRC,
            quote! {
                // Unreachable code is generated sometimes without the full feature.
                #![allow(unreachable_code, unused_variables)]
                #![allow(clippy::match_wildcard_for_single_variants, clippy::needless_match)]
                #[cfg(any(feature = "full", feature = "derive"))]
                use crate::gen::helper::fold::*;
                use crate::*;
                use syn::pm2::Span;
                #full_macro
                /// Syntax tree traversal to transform the nodes of an owned syntax tree.
                ///
                /// See the [module documentation] for details.
                ///
                /// [module documentation]: self
                pub trait Fold {
                    #traits
                }
                #impls
            },
        )?;
        Ok(())
    }
}
mod full {
    use syn::pm2::Stream;
    use syn::quote::quote;
    pub fn get_macro() -> Stream {
        quote! {
            #[cfg(feature = "full")]
            macro_rules! full {
                ($e:expr) => {
                    $e
                };
            }
            #[cfg(all(feature = "derive", not(feature = "full")))]
            macro_rules! full {
                ($e:expr) => {
                    unreachable!()
                };
            }
        }
    }
}
mod gen {
    use crate::cfg;
    use inflections::Inflect;
    use json::{Data, Definitions, Features, Node};
    use syn::pm2::{Ident, Span, Stream};
    pub const TERMINAL_TYPES: &[&str] = &["Span", "Ident"];
    pub fn under_name(name: &str) -> Ident {
        Ident::new(&name.to_snake_case(), Span::call_site())
    }
    pub fn traverse(defs: &Definitions, node: fn(&mut Stream, &mut Stream, &Node, &Definitions)) -> (Stream, Stream) {
        let mut types = defs.types.clone();
        for &terminal in TERMINAL_TYPES {
            types.push(Node {
                ident: terminal.to_owned(),
                features: Features::default(),
                data: Data::Private,
                exhaustive: true,
            });
        }
        types.sort_by(|a, b| a.ident.cmp(&b.ident));
        let mut traits = Stream::new();
        let mut impls = Stream::new();
        for s in types {
            let features = cfg::features(&s.features);
            traits.extend(features.clone());
            impls.extend(features);
            node(&mut traits, &mut impls, &s, defs);
        }
        (traits, impls)
    }
}
mod hash {
    use crate::{cfg, file, lookup};
    use anyhow::Result;
    use json::{Data, Definitions, Node, Type};
    use syn::pm2::{Ident, Span, Stream};
    use syn::quote::{format_ident, quote};
    const HASH_SRC: &str = "src/gen/hash.rs";
    fn skip(field_type: &Type) -> bool {
        match field_type {
            Type::Ext(ty) => ty == "Span",
            Type::Token(_) | Type::Group(_) => true,
            Type::Box(inner) => skip(inner),
            Type::Tuple(inner) => inner.iter().all(skip),
            _ => false,
        }
    }
    fn expand_impl_body(defs: &Definitions, node: &Node) -> Stream {
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
                                if !lookup::node(defs, ty).features.any.contains("derive") {
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
    fn expand_impl(defs: &Definitions, node: &Node) -> Stream {
        let manual_hash =
            node.data == Data::Private || node.ident == "Member" || node.ident == "Index" || node.ident == "Lifetime";
        if manual_hash {
            return Stream::new();
        }
        let ident = Ident::new(&node.ident, Span::call_site());
        let cfg_features = cfg::features(&node.features);
        let body = expand_impl_body(defs, node);
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
    pub fn generate(defs: &Definitions) -> Result<()> {
        let mut impls = Stream::new();
        for node in &defs.types {
            impls.extend(expand_impl(defs, node));
        }
        file::write(
            HASH_SRC,
            quote! {
                #[cfg(any(feature = "derive", feature = "full"))]
                use crate::tt::StreamHelper;
                use crate::*;
                use std::hash::{Hash, Hasher};
                #impls
            },
        )?;
        Ok(())
    }
}
mod json {
    use crate::workspace_path;
    use anyhow::Result;
    use std::fs;
    pub fn generate(defs: &Definitions) -> Result<()> {
        let mut j = serde_json::to_string_pretty(&defs)?;
        j.push('\n');
        let check: Definitions = serde_json::from_str(&j)?;
        assert_eq!(*defs, check);
        let json_path = workspace_path::get("syn.json");
        fs::write(json_path, j)?;
        Ok(())
    }
    use indexmap::IndexMap;
    use semver::Version;
    use serde::de::{Deserialize, Deserializer};
    use serde_derive::{Deserialize, Serialize};
    use std::collections::{BTreeMap, BTreeSet};

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct Definitions {
        pub version: Version,
        pub types: Vec<Node>,
        pub tokens: BTreeMap<String, String>,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct Node {
        pub ident: String,
        pub features: Features,
        #[serde(flatten, skip_serializing_if = "is_private", deserialize_with = "private_if_absent")]
        pub data: Data,
        #[serde(skip_serializing_if = "is_true", default = "bool_true")]
        pub exhaustive: bool,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub enum Data {
        Private,
        #[serde(rename = "fields")]
        Struct(Fields),
        #[serde(rename = "variants")]
        Enum(Variants),
    }

    pub type Fields = IndexMap<String, Type>;
    pub type Variants = IndexMap<String, Vec<Type>>;

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Type {
        Syn(String),
        Std(String),
        #[serde(rename = "proc_macro2")]
        Ext(String),
        Token(String),
        Group(String),
        Punctuated(Punctuated),
        Option(Box<Type>),
        Box(Box<Type>),
        Vec(Box<Type>),
        Tuple(Vec<Type>),
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct Punctuated {
        pub element: Box<Type>,
        pub punct: String,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Features {
        pub any: BTreeSet<String>,
    }

    fn is_private(data: &Data) -> bool {
        match data {
            Data::Private => true,
            Data::Struct(_) | Data::Enum(_) => false,
        }
    }
    fn private_if_absent<'de, D>(deserializer: D) -> Result<Data, D::Error>
    where
        D: Deserializer<'de>,
    {
        let option = Option::deserialize(deserializer)?;
        Ok(option.unwrap_or(Data::Private))
    }
    fn is_true(b: &bool) -> bool {
        *b
    }
    fn bool_true() -> bool {
        true
    }
}
mod lookup {
    use json::{Definitions, Node};
    pub fn node<'a>(defs: &'a Definitions, name: &str) -> &'a Node {
        for node in &defs.types {
            if node.ident == name {
                return node;
            }
        }
        panic!("not found: {}", name)
    }
}
mod operand {
    use syn::pm2::Stream;
    use syn::quote::quote;
    pub enum Operand {
        Borrowed(Stream),
        Owned(Stream),
    }
    pub use self::Operand::*;
    impl Operand {
        pub fn tokens(&self) -> &Stream {
            match self {
                Borrowed(n) | Owned(n) => n,
            }
        }
        pub fn ref_tokens(&self) -> Stream {
            match self {
                Borrowed(n) => n.clone(),
                Owned(n) => quote!(&#n),
            }
        }
        pub fn ref_mut_tokens(&self) -> Stream {
            match self {
                Borrowed(n) => n.clone(),
                Owned(n) => quote!(&mut #n),
            }
        }
        pub fn owned_tokens(&self) -> Stream {
            match self {
                Borrowed(n) => quote!(*#n),
                Owned(n) => n.clone(),
            }
        }
    }
}
mod parse {
    use crate::{version, workspace_path};
    use anyhow::{bail, Result};
    use indexmap::IndexMap;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use syn::parse::{Error, Parser};
    use syn::quote::quote;
    use syn::{
        parse_quote, Attribute, Data, DataEnum, DataStruct, DeriveInput, Fields, GenericArgument, Ident, Item,
        PathArguments, TypeMacro, TypePath, TypeTuple, UseTree, Visibility,
    };
    use syn_codegen as types;
    use thiserror::Error;
    const SYN_CRATE_ROOT: &str = "src/lib.rs";
    const TOKEN_SRC: &str = "src/token.rs";
    const IGNORED_MODS: &[&str] = &["fold", "visit", "visit_mut"];
    const EXTRA_TYPES: &[&str] = &["Lifetime"];
    struct Lookup {
        items: BTreeMap<Ident, AstItem>,
        // "+" => "Add"
        tokens: BTreeMap<String, String>,
        // "PatLit" => "ExprLit"
        aliases: BTreeMap<Ident, Ident>,
    }
    /// Parse the contents of `src` and return a list of AST types.
    pub fn parse() -> Result<types::Definitions> {
        let tokens = load_token_file(TOKEN_SRC)?;
        let mut lookup = Lookup {
            items: BTreeMap::new(),
            tokens,
            aliases: BTreeMap::new(),
        };
        load_file(SYN_CRATE_ROOT, &[], &mut lookup)?;
        let version = version::get()?;
        let types = lookup
            .items
            .values()
            .map(|item| introspect_item(item, &lookup))
            .collect();
        let tokens = lookup.tokens.into_iter().map(|(name, ty)| (ty, name)).collect();
        Ok(types::Definitions { version, types, tokens })
    }
    /// Data extracted from syn source
    pub struct AstItem {
        ast: DeriveInput,
        features: Vec<Attribute>,
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
                let last = path.segments.last().unwrap();
                let string = last.ident.to_string();
                match string.as_str() {
                    "Option" => {
                        let nested = introspect_type(first_arg(&last.arguments), lookup);
                        types::Type::Option(Box::new(nested))
                    },
                    "Punctuated" => {
                        let nested = introspect_type(first_arg(&last.arguments), lookup);
                        let punct = match introspect_type(last_arg(&last.arguments), lookup) {
                            types::Type::Token(s) => s,
                            _ => panic!(),
                        };
                        types::Type::Punctuated(types::Punctuated {
                            element: Box::new(nested),
                            punct,
                        })
                    },
                    "Vec" => {
                        let nested = introspect_type(first_arg(&last.arguments), lookup);
                        types::Type::Vec(Box::new(nested))
                    },
                    "Box" => {
                        let nested = introspect_type(first_arg(&last.arguments), lookup);
                        types::Type::Box(Box::new(nested))
                    },
                    "Brace" | "Bracket" | "Paren" | "Group" => types::Type::Group(string),
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
            syn::Type::Macro(TypeMacro { mac }) if mac.path.segments.last().unwrap().ident == "Token" => {
                let content = mac.tokens.to_string();
                let ty = lookup.tokens.get(&content).unwrap().to_string();
                types::Type::Token(ty)
            },
            _ => panic!("{}", quote!(#item).to_string()),
        }
    }
    fn introspect_features(attrs: &[Attribute]) -> types::Features {
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
    fn is_non_exhaustive(attrs: &[Attribute]) -> bool {
        for attr in attrs {
            if attr.path().is_ident("non_exhaustive") {
                return true;
            }
        }
        false
    }
    fn is_doc_hidden(attrs: &[Attribute]) -> bool {
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
        use syn::parse::{ParseStream, Result};
        use syn::pm2::Stream;
        use syn::quote::quote;
        use syn::{
            braced, bracketed, parenthesized, parse_quote, token, Attribute, Expr, Ident, Lit, LitStr, Path, Token,
        };
        use syn_codegen as types;
        fn peek_tag(input: ParseStream, tag: &str) -> bool {
            let ahead = input.fork();
            ahead.parse::<Token![#]>().is_ok() && ahead.parse::<Ident>().map(|ident| ident == tag).unwrap_or(false)
        }
        // Parses #full - returns #[cfg(feature = "full")] if it is present, and
        // nothing otherwise.
        fn full(input: ParseStream) -> Vec<Attribute> {
            if peek_tag(input, "full") {
                input.parse::<Token![#]>().unwrap();
                input.parse::<Ident>().unwrap();
                vec![parse_quote!(#[cfg(feature = "full")])]
            } else {
                vec![]
            }
        }
        // Parses a simple AstStruct without the `pub struct` prefix.
        fn ast_struct_inner(input: ParseStream) -> Result<AstItem> {
            let ident: Ident = input.parse()?;
            let features = full(input);
            let rest: Stream = input.parse()?;
            Ok(AstItem {
                ast: syn::parse2(quote! {
                    pub struct #ident #rest
                })?,
                features,
            })
        }
        pub fn ast_struct(input: ParseStream) -> Result<AstItem> {
            input.call(Attribute::parse_outer)?;
            input.parse::<Token![pub]>()?;
            input.parse::<Token![struct]>()?;
            let res = input.call(ast_struct_inner)?;
            Ok(res)
        }
        fn no_visit(input: ParseStream) -> bool {
            if peek_tag(input, "no_visit") {
                input.parse::<Token![#]>().unwrap();
                input.parse::<Ident>().unwrap();
                true
            } else {
                false
            }
        }
        pub fn ast_enum(input: ParseStream) -> Result<Option<AstItem>> {
            let attrs = input.call(Attribute::parse_outer)?;
            input.parse::<Token![pub]>()?;
            input.parse::<Token![enum]>()?;
            let ident: Ident = input.parse()?;
            let no_visit = no_visit(input);
            let rest: Stream = input.parse()?;
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
        // A single variant of an ast_enum_of_structs!
        struct EosVariant {
            attrs: Vec<Attribute>,
            name: Ident,
            member: Option<Path>,
        }
        fn eos_variant(input: ParseStream) -> Result<EosVariant> {
            let attrs = input.call(Attribute::parse_outer)?;
            let variant: Ident = input.parse()?;
            let member = if input.peek(token::Paren) {
                let content;
                parenthesized!(content in input);
                let path: Path = content.parse()?;
                Some(path)
            } else {
                None
            };
            input.parse::<Token![,]>()?;
            Ok(EosVariant {
                attrs,
                name: variant,
                member,
            })
        }
        pub fn ast_enum_of_structs(input: ParseStream) -> Result<AstItem> {
            let attrs = input.call(Attribute::parse_outer)?;
            input.parse::<Token![pub]>()?;
            input.parse::<Token![enum]>()?;
            let ident: Ident = input.parse()?;
            let content;
            braced!(content in input);
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
            syn::custom_kw!(hidden);
            syn::custom_kw!(macro_rules);
            syn::custom_kw!(Token);
        }
        pub fn parse_token_macro(input: ParseStream) -> Result<BTreeMap<String, String>> {
            let mut tokens = BTreeMap::new();
            while !input.is_empty() {
                let pattern;
                bracketed!(pattern in input);
                let token = pattern.parse::<Stream>()?.to_string();
                input.parse::<Token![=>]>()?;
                let expansion;
                braced!(expansion in input);
                input.parse::<Token![;]>()?;
                expansion.parse::<Token![$]>()?;
                let path: Path = expansion.parse()?;
                let ty = path.segments.last().unwrap().ident.to_string();
                tokens.insert(token, ty.to_string());
            }
            Ok(tokens)
        }
        fn parse_feature(input: ParseStream) -> Result<String> {
            let i: Ident = input.parse()?;
            assert_eq!(i, "feature");
            input.parse::<Token![=]>()?;
            let s = input.parse::<LitStr>()?;
            Ok(s.value())
        }
        pub fn parse_features(input: ParseStream) -> Result<types::Features> {
            let mut features = BTreeSet::new();
            let i: Ident = input.fork().parse()?;
            if i == "any" {
                input.parse::<Ident>()?;
                let nested;
                parenthesized!(nested in input);
                while !nested.is_empty() {
                    features.insert(parse_feature(&nested)?);
                    if !nested.is_empty() {
                        nested.parse::<Token![,]>()?;
                    }
                }
            } else if i == "feature" {
                features.insert(parse_feature(input)?);
                assert!(input.is_empty());
            } else {
                panic!("{:?}", i);
            }
            Ok(types::Features { any: features })
        }
        pub fn path_attr(attrs: &[Attribute]) -> Result<Option<&LitStr>> {
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
    fn clone_features(features: &[Attribute]) -> Vec<Attribute> {
        features.iter().map(|attr| parse_quote!(#attr)).collect()
    }
    fn get_features(attrs: &[Attribute], base: &[Attribute]) -> Vec<Attribute> {
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
        features: &[Attribute],
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
        features: &[Attribute],
        lookup: &mut Lookup,
    ) -> Result<()> {
        let relative_to_workspace_root = relative_to_workspace_root.as_ref();
        let parent = relative_to_workspace_root.parent().expect("no parent path");
        // Parse the file
        let src = fs::read_to_string(workspace_path::get(relative_to_workspace_root))?;
        let file = syn::parse_file(&src)?;
        // Collect all of the interesting AstItems declared in this file or submodules.
        'items: for item in file.items {
            match item {
                Item::Mod(item) => {
                    // Don't inspect inline modules.
                    if item.content.is_some() {
                        continue;
                    }
                    // We don't want to try to load the generated rust files and
                    // parse them, so we ignore them here.
                    for name in IGNORED_MODS {
                        if item.ident == name {
                            continue 'items;
                        }
                    }
                    // Lookup any #[cfg()] attributes on the module and add them to
                    // the feature set.
                    //
                    // The derive module is weird because it is built with either
                    // `full` or `derive` but exported only under `derive`.
                    let features = if item.ident == "derive" {
                        vec![parse_quote!(#[cfg(feature = "derive")])]
                    } else {
                        get_features(&item.attrs, features)
                    };
                    // Look up the submodule file, and recursively parse it.
                    // Only handles same-directory .rs file submodules for now.
                    let filename = if let Some(filename) = parsing::path_attr(&item.attrs)? {
                        filename.value()
                    } else {
                        format!("{}.rs", item.ident)
                    };
                    let path = parent.join(filename);
                    load_file(path, &features, lookup)?;
                },
                Item::Macro(item) => {
                    // Lookip any #[cfg()] attributes directly on the macro
                    // invocation, and add them to the feature set.
                    let features = get_features(&item.attrs, features);
                    // Try to parse the AstItem declaration out of the item.
                    let tts = item.mac.tokens.clone();
                    let found = if item.mac.path.is_ident("ast_struct") {
                        Some(parsing::ast_struct.parse2(tts)?)
                    } else if item.mac.path.is_ident("ast_enum") {
                        parsing::ast_enum.parse2(tts)?
                    } else if item.mac.path.is_ident("ast_enum_of_structs") {
                        Some(parsing::ast_enum_of_structs.parse2(tts)?)
                    } else {
                        continue;
                    };
                    // Record our features on the parsed AstItems.
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
                                ast: DeriveInput {
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
        let path = workspace_path::get(relative_to_workspace_root);
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
mod snapshot {
    use crate::operand::{Borrowed, Operand, Owned};
    use crate::{file, lookup};
    use anyhow::Result;
    use json::{Data, Definitions, Node, Type};
    use syn::pm2::{Ident, Span, Stream};
    use syn::quote::{format_ident, quote};
    use syn::Index;
    const TESTS_DEBUG_SRC: &str = "tests/debug/gen.rs";
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
            Type::Punctuated(ty) => {
                let element = rust_type(&ty.element);
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
    fn is_printable(ty: &Type) -> bool {
        match ty {
            Type::Ext(name) => name != "Span",
            Type::Box(ty) => is_printable(ty),
            Type::Tuple(ty) => ty.iter().any(is_printable),
            Type::Token(_) | Type::Group(_) => false,
            Type::Syn(_) | Type::Std(_) | Type::Punctuated(_) | Type::Option(_) | Type::Vec(_) => true,
        }
    }
    fn format_field(val: &Operand, ty: &Type) -> Option<Stream> {
        if !is_printable(ty) {
            return None;
        }
        let format = match ty {
            Type::Option(ty) => {
                if let Some(format) = format_field(&Borrowed(quote!(_val)), ty) {
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
                        let inner = Owned(quote!(#val.#index));
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
    fn syntax_tree_enum<'a>(outer: &str, inner: &str, fields: &'a [Type]) -> Option<&'a str> {
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
    fn expand_impl_body(defs: &Definitions, node: &Node, name: &str, val: &Operand) -> Stream {
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
                    } else if let Some(inner) = syntax_tree_enum(name, v, fields) {
                        let format = expand_impl_body(defs, lookup::node(defs, inner), &path, &Borrowed(quote!(_val)));
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
                            format_field(&Borrowed(val), ty).map(|format| {
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
                            let format = format_field(&Borrowed(val), ty)?;
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
                        Some(if let Some(format) = format_field(&Owned(quote!(self.0)), ty) {
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
                        let inner = Owned(quote!(#val.#ident));
                        let format = format_field(&inner, ty)?;
                        let mut call = quote! {
                            formatter.field(#f, #format);
                        };
                        if let Type::Vec(_) | Type::Punctuated(_) = ty {
                            call = quote! {
                                if !#val.#ident.is_empty() {
                                    #call
                                }
                            };
                        } else if let Type::Syn(inner) = ty {
                            for node in &defs.types {
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
    fn expand_impl(defs: &Definitions, node: &Node) -> Stream {
        let ident = Ident::new(&node.ident, Span::call_site());
        let body = expand_impl_body(defs, node, &node.ident, &Owned(quote!(self.value)));
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
    pub fn generate(defs: &Definitions) -> Result<()> {
        let mut impls = Stream::new();
        for node in &defs.types {
            impls.extend(expand_impl(defs, node));
        }
        file::write(
            TESTS_DEBUG_SRC,
            quote! {
                #![allow(clippy::match_wildcard_for_single_variants)]
                use super::{Lite, Present};
                use ref_cast::RefCast;
                use std::fmt::{self, Debug, Display};
                #impls
            },
        )?;
        Ok(())
    }
}
mod version {
    use crate::workspace_path;
    use anyhow::Result;
    use semver::Version;
    use serde_derive::Deserialize;
    use std::fs;
    pub fn get() -> Result<Version> {
        let syn_cargo_toml = workspace_path::get("Cargo.toml");
        let manifest = fs::read_to_string(syn_cargo_toml)?;
        let parsed: Manifest = toml::from_str(&manifest)?;
        Ok(parsed.package.version)
    }
    #[derive(Debug, Deserialize)]
    struct Manifest {
        package: Package,
    }
    #[derive(Debug, Deserialize)]
    struct Package {
        version: Version,
    }
}
mod visit {
    use crate::operand::{Borrowed, Operand, Owned};
    use crate::{file, full, gen};
    use anyhow::Result;
    use json::{Data, Definitions, Features, Node, Type};
    use syn::pm2::{Ident, Span, Stream};
    use syn::quote::{format_ident, quote};
    use syn::Index;
    const VISIT_SRC: &str = "src/gen/visit.rs";
    fn simple_visit(item: &str, name: &Operand) -> Stream {
        let ident = gen::under_name(item);
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
    fn visit(ty: &Type, features: &Features, defs: &Definitions, name: &Operand) -> Option<Stream> {
        match ty {
            Type::Box(t) => {
                let name = name.owned_tokens();
                visit(t, features, defs, &Owned(quote!(*#name)))
            },
            Type::Vec(t) => {
                let operand = Borrowed(quote!(it));
                let val = visit(t, features, defs, &operand)?;
                let name = name.ref_tokens();
                Some(quote! {
                    for it in #name {
                        #val;
                    }
                })
            },
            Type::Punctuated(p) => {
                let operand = Borrowed(quote!(it));
                let val = visit(&p.element, features, defs, &operand)?;
                let name = name.ref_tokens();
                Some(quote! {
                    for el in Punctuated::pairs(#name) {
                        let it = el.value();
                        #val;
                    }
                })
            },
            Type::Option(t) => {
                let it = Borrowed(quote!(it));
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
                    let it = Owned(quote!((#name).#i));
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
                let target = defs.types.iter().find(|ty| ty.ident == *t).unwrap();
                if requires_full(&target.features) && !requires_full(features) {
                    res = quote!(full!(#res));
                }
                Some(res)
            },
            Type::Ext(t) if gen::TERMINAL_TYPES.contains(&&t[..]) => Some(simple_visit(t, name)),
            Type::Ext(_) | Type::Std(_) | Type::Token(_) | Type::Group(_) => None,
        }
    }
    fn node(traits: &mut Stream, impls: &mut Stream, s: &Node, defs: &Definitions) {
        let under_name = gen::under_name(&s.ident);
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
                            let borrowed_binding = Borrowed(quote!(#binding));
                            visit_fields.extend(
                                visit(ty, &s.features, defs, &borrowed_binding)
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
                    let ref_toks = Owned(quote!(node.#id));
                    let visit_field = visit(ty, &s.features, defs, &ref_toks).unwrap_or_else(|| noop_visit(&ref_toks));
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
    pub fn generate(defs: &Definitions) -> Result<()> {
        let (traits, impls) = gen::traverse(defs, node);
        let full_macro = full::get_macro();
        file::write(
            VISIT_SRC,
            quote! {
                #![allow(unused_variables)]
                #[cfg(any(feature = "full", feature = "derive"))]
                use crate::punctuated::Punctuated;
                use crate::*;
                use syn::pm2::Span;
                #full_macro
                macro_rules! skip {
                    ($($tt:tt)*) => {};
                }
                /// Syntax tree traversal to walk a shared borrow of a syntax tree.
                ///
                /// See the [module documentation] for details.
                ///
                /// [module documentation]: self
                pub trait Visit<'ast> {
                    #traits
                }
                #impls
            },
        )?;
        Ok(())
    }
}
mod visit_mut {
    use crate::operand::{Borrowed, Operand, Owned};
    use crate::{file, full, gen};
    use anyhow::Result;
    use json::{Data, Definitions, Features, Node, Type};
    use syn::pm2::{Ident, Span, Stream};
    use syn::quote::{format_ident, quote};
    use syn::Index;
    const VISIT_MUT_SRC: &str = "src/gen/visit_mut.rs";
    fn simple_visit(item: &str, name: &Operand) -> Stream {
        let ident = gen::under_name(item);
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
    fn visit(ty: &Type, features: &Features, defs: &Definitions, name: &Operand) -> Option<Stream> {
        match ty {
            Type::Box(t) => {
                let name = name.owned_tokens();
                visit(t, features, defs, &Owned(quote!(*#name)))
            },
            Type::Vec(t) => {
                let operand = Borrowed(quote!(it));
                let val = visit(t, features, defs, &operand)?;
                let name = name.ref_mut_tokens();
                Some(quote! {
                    for it in #name {
                        #val;
                    }
                })
            },
            Type::Punctuated(p) => {
                let operand = Borrowed(quote!(it));
                let val = visit(&p.element, features, defs, &operand)?;
                let name = name.ref_mut_tokens();
                Some(quote! {
                    for mut el in Punctuated::pairs_mut(#name) {
                        let it = el.value_mut();
                        #val;
                    }
                })
            },
            Type::Option(t) => {
                let it = Borrowed(quote!(it));
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
                    let it = Owned(quote!((#name).#i));
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
                let target = defs.types.iter().find(|ty| ty.ident == *t).unwrap();
                if requires_full(&target.features) && !requires_full(features) {
                    res = quote!(full!(#res));
                }
                Some(res)
            },
            Type::Ext(t) if gen::TERMINAL_TYPES.contains(&&t[..]) => Some(simple_visit(t, name)),
            Type::Ext(_) | Type::Std(_) | Type::Token(_) | Type::Group(_) => None,
        }
    }
    fn node(traits: &mut Stream, impls: &mut Stream, s: &Node, defs: &Definitions) {
        let under_name = gen::under_name(&s.ident);
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
                            let borrowed_binding = Borrowed(quote!(#binding));
                            visit_mut_fields.extend(
                                visit(ty, &s.features, defs, &borrowed_binding)
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
                    let ref_toks = Owned(quote!(node.#id));
                    let visit_mut_field =
                        visit(ty, &s.features, defs, &ref_toks).unwrap_or_else(|| noop_visit(&ref_toks));
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
                V: VisitMut + ?Sized,
            {
                #visit_mut_impl
            }
        });
    }
    pub fn generate(defs: &Definitions) -> Result<()> {
        let (traits, impls) = gen::traverse(defs, node);
        let full_macro = full::get_macro();
        file::write(
            VISIT_MUT_SRC,
            quote! {
                #![allow(unused_variables)]
                #[cfg(any(feature = "full", feature = "derive"))]
                use crate::punctuated::Punctuated;
                use crate::*;
                use syn::pm2::Span;
                #full_macro
                macro_rules! skip {
                    ($($tt:tt)*) => {};
                }
                /// Syntax tree traversal to mutate an exclusive borrow of a syntax tree in
                /// place.
                ///
                /// See the [module documentation] for details.
                ///
                /// [module documentation]: self
                pub trait VisitMut {
                    #traits
                }
                #impls
            },
        )?;
        Ok(())
    }
}
mod workspace_path {
    use std::path::{Path, PathBuf};
    pub fn get(relative_to_workspace_root: impl AsRef<Path>) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        assert!(path.pop());
        path.push(relative_to_workspace_root);
        path
    }
}
fn main() -> anyhow::Result<()> {
    color_backtrace::install();
    let defs = parse::parse()?;
    clone::generate(&defs)?;
    debug::generate(&defs)?;
    eq::generate(&defs)?;
    hash::generate(&defs)?;
    json::generate(&defs)?;
    fold::generate(&defs)?;
    visit::generate(&defs)?;
    visit_mut::generate(&defs)?;
    snapshot::generate(&defs)?;
    Ok(())
}
