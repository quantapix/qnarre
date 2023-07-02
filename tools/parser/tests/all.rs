use expect_test::expect_file;
use rayon::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
};
use syntax::{ast, fuzz, SourceFile, SyntaxErr};
use test_utils::{bench, bench_fixture, project_root};

#[cfg(not(feature = "in-rust-tree"))]
mod ast_src {
    pub struct KindsSrc<'a> {
        pub punct: &'a [(&'a str, &'a str)],
        pub keywords: &'a [&'a str],
        pub contextual_keywords: &'a [&'a str],
        pub literals: &'a [&'a str],
        pub tokens: &'a [&'a str],
        pub nodes: &'a [&'a str],
    }

    pub const KINDS_SRC: KindsSrc<'_> = KindsSrc {
        punct: &[
            (";", "SEMICOLON"),
            (",", "COMMA"),
            ("(", "L_PAREN"),
            (")", "R_PAREN"),
            ("{", "L_CURLY"),
            ("}", "R_CURLY"),
            ("[", "L_BRACK"),
            ("]", "R_BRACK"),
            ("<", "L_ANGLE"),
            (">", "R_ANGLE"),
            ("@", "AT"),
            ("#", "POUND"),
            ("~", "TILDE"),
            ("?", "QUESTION"),
            ("$", "DOLLAR"),
            ("&", "AMP"),
            ("|", "PIPE"),
            ("+", "PLUS"),
            ("*", "STAR"),
            ("/", "SLASH"),
            ("^", "CARET"),
            ("%", "PERCENT"),
            ("_", "UNDERSCORE"),
            (".", "DOT"),
            ("..", "DOT2"),
            ("...", "DOT3"),
            ("..=", "DOT2EQ"),
            (":", "COLON"),
            ("::", "COLON2"),
            ("=", "EQ"),
            ("==", "EQ2"),
            ("=>", "FAT_ARROW"),
            ("!", "BANG"),
            ("!=", "NEQ"),
            ("-", "MINUS"),
            ("->", "THIN_ARROW"),
            ("<=", "LTEQ"),
            (">=", "GTEQ"),
            ("+=", "PLUSEQ"),
            ("-=", "MINUSEQ"),
            ("|=", "PIPEEQ"),
            ("&=", "AMPEQ"),
            ("^=", "CARETEQ"),
            ("/=", "SLASHEQ"),
            ("*=", "STAREQ"),
            ("%=", "PERCENTEQ"),
            ("&&", "AMP2"),
            ("||", "PIPE2"),
            ("<<", "SHL"),
            (">>", "SHR"),
            ("<<=", "SHLEQ"),
            (">>=", "SHREQ"),
        ],
        keywords: &[
            "as", "async", "await", "box", "break", "const", "continue", "crate", "do", "dyn", "else", "enum",
            "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "macro", "match", "mod", "move", "mut",
            "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true", "try", "type",
            "unsafe", "use", "where", "while", "yield",
        ],
        contextual_keywords: &["auto", "default", "existential", "union", "raw", "macro_rules", "yeet"],
        literals: &[
            "INT_NUMBER",
            "FLOAT_NUMBER",
            "CHAR",
            "BYTE",
            "STRING",
            "BYTE_STRING",
            "C_STRING",
        ],
        tokens: &["ERROR", "IDENT", "WHITESPACE", "LIFETIME_IDENT", "COMMENT", "SHEBANG"],
        nodes: &[
            "SOURCE_FILE",
            "STRUCT",
            "UNION",
            "ENUM",
            "FN",
            "RET_TYPE",
            "EXTERN_CRATE",
            "MODULE",
            "USE",
            "STATIC",
            "CONST",
            "TRAIT",
            "TRAIT_ALIAS",
            "IMPL",
            "TYPE_ALIAS",
            "MACRO_CALL",
            "MACRO_RULES",
            "MACRO_ARM",
            "TOKEN_TREE",
            "MACRO_DEF",
            "PAREN_TYPE",
            "TUPLE_TYPE",
            "MACRO_TYPE",
            "NEVER_TYPE",
            "PATH_TYPE",
            "PTR_TYPE",
            "ARRAY_TYPE",
            "SLICE_TYPE",
            "REF_TYPE",
            "INFER_TYPE",
            "FN_PTR_TYPE",
            "FOR_TYPE",
            "IMPL_TRAIT_TYPE",
            "DYN_TRAIT_TYPE",
            "OR_PAT",
            "PAREN_PAT",
            "REF_PAT",
            "BOX_PAT",
            "IDENT_PAT",
            "WILDCARD_PAT",
            "REST_PAT",
            "PATH_PAT",
            "RECORD_PAT",
            "RECORD_PAT_FIELD_LIST",
            "RECORD_PAT_FIELD",
            "TUPLE_STRUCT_PAT",
            "TUPLE_PAT",
            "SLICE_PAT",
            "RANGE_PAT",
            "LITERAL_PAT",
            "MACRO_PAT",
            "CONST_BLOCK_PAT",
            "TUPLE_EXPR",
            "ARRAY_EXPR",
            "PAREN_EXPR",
            "PATH_EXPR",
            "CLOSURE_EXPR",
            "IF_EXPR",
            "WHILE_EXPR",
            "LOOP_EXPR",
            "FOR_EXPR",
            "CONTINUE_EXPR",
            "BREAK_EXPR",
            "LABEL",
            "BLOCK_EXPR",
            "STMT_LIST",
            "RETURN_EXPR",
            "YIELD_EXPR",
            "YEET_EXPR",
            "LET_EXPR",
            "UNDERSCORE_EXPR",
            "MACRO_EXPR",
            "MATCH_EXPR",
            "MATCH_ARM_LIST",
            "MATCH_ARM",
            "MATCH_GUARD",
            "RECORD_EXPR",
            "RECORD_EXPR_FIELD_LIST",
            "RECORD_EXPR_FIELD",
            "BOX_EXPR",
            "CALL_EXPR",
            "INDEX_EXPR",
            "METHOD_CALL_EXPR",
            "FIELD_EXPR",
            "AWAIT_EXPR",
            "TRY_EXPR",
            "CAST_EXPR",
            "REF_EXPR",
            "PREFIX_EXPR",
            "RANGE_EXPR",
            "BIN_EXPR",
            "EXTERN_BLOCK",
            "EXTERN_ITEM_LIST",
            "VARIANT",
            "RECORD_FIELD_LIST",
            "RECORD_FIELD",
            "TUPLE_FIELD_LIST",
            "TUPLE_FIELD",
            "VARIANT_LIST",
            "ITEM_LIST",
            "ASSOC_ITEM_LIST",
            "ATTR",
            "META",
            "USE_TREE",
            "USE_TREE_LIST",
            "PATH",
            "PATH_SEGMENT",
            "LITERAL",
            "RENAME",
            "VISIBILITY",
            "WHERE_CLAUSE",
            "WHERE_PRED",
            "ABI",
            "NAME",
            "NAME_REF",
            "LET_STMT",
            "LET_ELSE",
            "EXPR_STMT",
            "GENERIC_PARAM_LIST",
            "GENERIC_PARAM",
            "LIFETIME_PARAM",
            "TYPE_PARAM",
            "RETURN_TYPE_ARG",
            "CONST_PARAM",
            "GENERIC_ARG_LIST",
            "LIFETIME",
            "LIFETIME_ARG",
            "TYPE_ARG",
            "ASSOC_TYPE_ARG",
            "CONST_ARG",
            "PARAM_LIST",
            "PARAM",
            "SELF_PARAM",
            "ARG_LIST",
            "TYPE_BOUND",
            "TYPE_BOUND_LIST",
            "MACRO_ITEMS",
            "MACRO_STMTS",
        ],
    };

    #[derive(Default, Debug)]
    pub struct AstSrc {
        pub tokens: Vec<String>,
        pub nodes: Vec<NodeSrc>,
        pub enums: Vec<EnumSrc>,
    }

    #[derive(Debug)]
    pub struct NodeSrc {
        pub doc: Vec<String>,
        pub name: String,
        pub traits: Vec<String>,
        pub fields: Vec<Field>,
    }

    #[derive(Debug, Eq, PartialEq)]
    pub enum Field {
        Token(String),
        Node {
            name: String,
            ty: String,
            cardinality: Cardinality,
        },
    }

    #[derive(Debug, Eq, PartialEq)]
    pub enum Cardinality {
        Optional,
        Many,
    }

    #[derive(Debug)]
    pub struct EnumSrc {
        pub doc: Vec<String>,
        pub name: String,
        pub traits: Vec<String>,
        pub variants: Vec<String>,
    }
}
#[cfg(not(feature = "in-rust-tree"))]
mod srcgen_ast {
    use super::ast_src::{AstSrc, Cardinality, EnumSrc, Field, KindsSrc, NodeSrc, KINDS_SRC};
    use itertools::Itertools;
    use proc_macro2::{Punct, Spacing};
    use quote::{format_ident, quote};
    use std::{
        collections::{BTreeSet, HashSet},
        fmt::Write,
    };
    use ungrammar::{Grammar, Rule};

    #[test]
    fn sourcegen_ast() {
        let syntax_kinds = generate_syntax_kinds(KINDS_SRC);
        let syntax_kinds_file = sourcegen::project_root().join("crates/parser/src/syntax_kind/generated.rs");
        sourcegen::ensure_file_contents(syntax_kinds_file.as_path(), &syntax_kinds);
        let grammar = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/rust.ungram"))
            .parse()
            .unwrap();
        let ast = lower(&grammar);
        let ast_tokens = generate_tokens(&ast);
        let ast_tokens_file = sourcegen::project_root().join("crates/syntax/src/ast/generated/tokens.rs");
        sourcegen::ensure_file_contents(ast_tokens_file.as_path(), &ast_tokens);
        let ast_nodes = generate_nodes(KINDS_SRC, &ast);
        let ast_nodes_file = sourcegen::project_root().join("crates/syntax/src/ast/generated/nodes.rs");
        sourcegen::ensure_file_contents(ast_nodes_file.as_path(), &ast_nodes);
    }
    fn generate_tokens(grammar: &AstSrc) -> String {
        let tokens = grammar.tokens.iter().map(|token| {
            let name = format_ident!("{}", token);
            let kind = format_ident!("{}", to_upper_snake_case(token));
            quote! {
                #[derive(Debug, Clone, PartialEq, Eq, Hash)]
                pub struct #name {
                    pub syntax: syntax::Token,
                }
                impl std::fmt::Display for #name {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        std::fmt::Display::fmt(&self.syntax, f)
                    }
                }
                impl ast::Token for #name {
                    fn can_cast(kind: SyntaxKind) -> bool { kind == #kind }
                    fn cast(syntax: syntax::Token) -> Option<Self> {
                        if Self::can_cast(syntax.kind()) { Some(Self { syntax }) } else { None }
                    }
                    fn syntax(&self) -> &syntax::Token { &self.syntax }
                }
            }
        });
        sourcegen::add_preamble(
            "sourcegen_ast",
            sourcegen::reformat(
                quote! {
                    use crate::{SyntaxKind::{self, *}, syntax::Token, ast};
                    #(#tokens)*
                }
                .to_string(),
            ),
        )
        .replace("#[derive", "\n#[derive")
    }
    fn generate_nodes(kinds: KindsSrc<'_>, grammar: &AstSrc) -> String {
        let (node_defs, node_boilerplate_impls): (Vec<_>, Vec<_>) = grammar
            .nodes
            .iter()
            .map(|node| {
                let name = format_ident!("{}", node.name);
                let kind = format_ident!("{}", to_upper_snake_case(&node.name));
                let traits = node
                    .traits
                    .iter()
                    .filter(|trait_name| {
                        node.name != "ForExpr" && node.name != "WhileExpr" || trait_name.as_str() != "HasLoopBody"
                    })
                    .map(|trait_name| {
                        let trait_name = format_ident!("{}", trait_name);
                        quote!(impl ast::#trait_name for #name {})
                    });
                let methods = node.fields.iter().map(|field| {
                    let method_name = field.method_name();
                    let ty = field.ty();
                    if field.is_many() {
                        quote! {
                            pub fn #method_name(&self) -> ast::Children<#ty> {
                                ast::children(&self.syntax)
                            }
                        }
                    } else if let Some(token_kind) = field.token_kind() {
                        quote! {
                            pub fn #method_name(&self) -> Option<#ty> {
                                ast::token(&self.syntax, #token_kind)
                            }
                        }
                    } else {
                        quote! {
                            pub fn #method_name(&self) -> Option<#ty> {
                                ast::child(&self.syntax)
                            }
                        }
                    }
                });
                (
                    quote! {
                        #[pretty_doc_comment_placeholder_workaround]
                        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
                        pub struct #name {
                            pub syntax: crate::Node,
                        }
                        #(#traits)*
                        impl #name {
                            #(#methods)*
                        }
                    },
                    quote! {
                        impl ast::Node for #name {
                            fn can_cast(kind: SyntaxKind) -> bool {
                                kind == #kind
                            }
                            fn cast(syntax: crate::Node) -> Option<Self> {
                                if Self::can_cast(syntax.kind()) { Some(Self { syntax }) } else { None }
                            }
                            fn syntax(&self) -> &crate::Node { &self.syntax }
                        }
                    },
                )
            })
            .unzip();
        let (enum_defs, enum_boilerplate_impls): (Vec<_>, Vec<_>) = grammar
            .enums
            .iter()
            .map(|en| {
                let variants: Vec<_> = en.variants.iter().map(|var| format_ident!("{}", var)).collect();
                let name = format_ident!("{}", en.name);
                let kinds: Vec<_> = variants
                    .iter()
                    .map(|name| format_ident!("{}", to_upper_snake_case(&name.to_string())))
                    .collect();
                let traits = en.traits.iter().map(|trait_name| {
                    let trait_name = format_ident!("{}", trait_name);
                    quote!(impl ast::#trait_name for #name {})
                });
                let ast_node = if en.name == "Stmt" {
                    quote! {}
                } else {
                    quote! {
                        impl ast::Node for #name {
                            fn can_cast(kind: SyntaxKind) -> bool {
                                matches!(kind, #(#kinds)|*)
                            }
                            fn cast(syntax: crate::Node) -> Option<Self> {
                                let res = match syntax.kind() {
                                    #(
                                    #kinds => #name::#variants(#variants { syntax }),
                                    )*
                                    _ => return None,
                                };
                                Some(res)
                            }
                            fn syntax(&self) -> &crate::Node {
                                match self {
                                    #(
                                    #name::#variants(it) => &it.syntax,
                                    )*
                                }
                            }
                        }
                    }
                };
                (
                    quote! {
                        #[pretty_doc_comment_placeholder_workaround]
                        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
                        pub enum #name {
                            #(#variants(#variants),)*
                        }
                        #(#traits)*
                    },
                    quote! {
                        #(
                            impl From<#variants> for #name {
                                fn from(node: #variants) -> #name {
                                    #name::#variants(node)
                                }
                            }
                        )*
                        #ast_node
                    },
                )
            })
            .unzip();
        let (any_node_defs, any_node_boilerplate_impls): (Vec<_>, Vec<_>) = grammar
            .nodes
            .iter()
            .flat_map(|node| node.traits.iter().map(move |t| (t, node)))
            .into_group_map()
            .into_iter()
            .sorted_by_key(|(k, _)| *k)
            .map(|(trait_name, nodes)| {
                let name = format_ident!("Any{}", trait_name);
                let trait_name = format_ident!("{}", trait_name);
                let kinds: Vec<_> = nodes
                    .iter()
                    .map(|name| format_ident!("{}", to_upper_snake_case(&name.name.to_string())))
                    .collect();
                (
                    quote! {
                        #[pretty_doc_comment_placeholder_workaround]
                        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
                        pub struct #name {
                            pub syntax: crate::Node,
                        }
                        impl ast::#trait_name for #name {}
                    },
                    quote! {
                        impl #name {
                            #[inline]
                            pub fn new<T: ast::#trait_name>(node: T) -> #name {
                                #name {
                                    syntax: node.syntax().clone()
                                }
                            }
                        }
                        impl ast::Node for #name {
                            fn can_cast(kind: SyntaxKind) -> bool {
                                matches!(kind, #(#kinds)|*)
                            }
                            fn cast(syntax: crate::Node) -> Option<Self> {
                                Self::can_cast(syntax.kind()).then_some(#name { syntax })
                            }
                            fn syntax(&self) -> &crate::Node {
                                &self.syntax
                            }
                        }
                    },
                )
            })
            .unzip();
        let enum_names = grammar.enums.iter().map(|x| &x.name);
        let node_names = grammar.nodes.iter().map(|x| &x.name);
        let display_impls = enum_names
            .chain(node_names.clone())
            .map(|x| format_ident!("{}", x))
            .map(|x| {
                quote! {
                    impl std::fmt::Display for #x {
                        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                            std::fmt::Display::fmt(self.syntax(), f)
                        }
                    }
                }
            });
        let defined_nodes: HashSet<_> = node_names.collect();
        for node in kinds
            .nodes
            .iter()
            .map(|x| to_pascal_case(x))
            .filter(|x| !defined_nodes.iter().any(|&x2| x2 == x))
        {
            drop(node)
        }
        let ast = quote! {
            #![allow(non_snake_case)]
            use crate::{
                crate::Node, syntax::Token, SyntaxKind::{self, *},
                ast,
                T,
            };
            #(#node_defs)*
            #(#enum_defs)*
            #(#any_node_defs)*
            #(#node_boilerplate_impls)*
            #(#enum_boilerplate_impls)*
            #(#any_node_boilerplate_impls)*
            #(#display_impls)*
        };
        let ast = ast.to_string().replace("T ! [", "T![");
        let mut res = String::with_capacity(ast.len() * 2);
        let mut docs = grammar
            .nodes
            .iter()
            .map(|x| &x.doc)
            .chain(grammar.enums.iter().map(|x| &x.doc));
        for chunk in ast.split("# [pretty_doc_comment_placeholder_workaround] ") {
            res.push_str(chunk);
            if let Some(doc) = docs.next() {
                write_doc_comment(doc, &mut res);
            }
        }
        let res = sourcegen::add_preamble("sourcegen_ast", sourcegen::reformat(res));
        res.replace("#[derive", "\n#[derive")
    }
    fn write_doc_comment(contents: &[String], dest: &mut String) {
        for line in contents {
            writeln!(dest, "///{line}").unwrap();
        }
    }
    fn generate_syntax_kinds(grammar: KindsSrc<'_>) -> String {
        let (single_byte_tokens_values, single_byte_tokens): (Vec<_>, Vec<_>) = grammar
            .punct
            .iter()
            .filter(|(token, _name)| token.len() == 1)
            .map(|(token, name)| (token.chars().next().unwrap(), format_ident!("{}", name)))
            .unzip();
        let punctuation_values = grammar.punct.iter().map(|(token, _name)| {
            if "{}[]()".contains(token) {
                let c = token.chars().next().unwrap();
                quote! { #c }
            } else {
                let cs = token.chars().map(|c| Punct::new(c, Spacing::Joint));
                quote! { #(#cs)* }
            }
        });
        let punctuation = grammar
            .punct
            .iter()
            .map(|(_token, name)| format_ident!("{}", name))
            .collect::<Vec<_>>();
        let x = |&name| match name {
            "Self" => format_ident!("SELF_TYPE_KW"),
            name => format_ident!("{}_KW", to_upper_snake_case(name)),
        };
        let full_keywords_values = grammar.keywords;
        let full_keywords = full_keywords_values.iter().map(x);
        let contextual_keywords_values = &grammar.contextual_keywords;
        let contextual_keywords = contextual_keywords_values.iter().map(x);
        let all_keywords_values = grammar
            .keywords
            .iter()
            .chain(grammar.contextual_keywords.iter())
            .copied()
            .collect::<Vec<_>>();
        let all_keywords_idents = all_keywords_values.iter().map(|kw| format_ident!("{}", kw));
        let all_keywords = all_keywords_values.iter().map(x).collect::<Vec<_>>();
        let literals = grammar
            .literals
            .iter()
            .map(|name| format_ident!("{}", name))
            .collect::<Vec<_>>();
        let tokens = grammar
            .tokens
            .iter()
            .map(|name| format_ident!("{}", name))
            .collect::<Vec<_>>();
        let nodes = grammar
            .nodes
            .iter()
            .map(|name| format_ident!("{}", name))
            .collect::<Vec<_>>();
        let ast = quote! {
            #![allow(bad_style, missing_docs, unreachable_pub)]
            #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
            #[repr(u16)]
            pub enum SyntaxKind {
                #[doc(hidden)]
                TOMBSTONE,
                #[doc(hidden)]
                EOF,
                #(#punctuation,)*
                #(#all_keywords,)*
                #(#literals,)*
                #(#tokens,)*
                #(#nodes,)*
                #[doc(hidden)]
                __LAST,
            }
            use self::SyntaxKind::*;
            impl SyntaxKind {
                pub fn is_keyword(self) -> bool {
                    matches!(self, #(#all_keywords)|*)
                }
                pub fn is_punct(self) -> bool {
                    matches!(self, #(#punctuation)|*)
                }
                pub fn is_literal(self) -> bool {
                    matches!(self, #(#literals)|*)
                }
                pub fn from_keyword(ident: &str) -> Option<SyntaxKind> {
                    let kw = match ident {
                        #(#full_keywords_values => #full_keywords,)*
                        _ => return None,
                    };
                    Some(kw)
                }
                pub fn from_contextual_keyword(ident: &str) -> Option<SyntaxKind> {
                    let kw = match ident {
                        #(#contextual_keywords_values => #contextual_keywords,)*
                        _ => return None,
                    };
                    Some(kw)
                }
                pub fn from_char(c: char) -> Option<SyntaxKind> {
                    let tok = match c {
                        #(#single_byte_tokens_values => #single_byte_tokens,)*
                        _ => return None,
                    };
                    Some(tok)
                }
            }
            #[macro_export]
            macro_rules! T {
                #([#punctuation_values] => { $crate::SyntaxKind::#punctuation };)*
                #([#all_keywords_idents] => { $crate::SyntaxKind::#all_keywords };)*
                [lifetime_ident] => { $crate::SyntaxKind::LIFETIME_IDENT };
                [ident] => { $crate::SyntaxKind::IDENT };
                [shebang] => { $crate::SyntaxKind::SHEBANG };
            }
            pub use T;
        };
        sourcegen::add_preamble("sourcegen_ast", sourcegen::reformat(ast.to_string()))
    }
    fn to_upper_snake_case(s: &str) -> String {
        let mut buf = String::with_capacity(s.len());
        let mut prev = false;
        for c in s.chars() {
            if c.is_ascii_uppercase() && prev {
                buf.push('_')
            }
            prev = true;
            buf.push(c.to_ascii_uppercase());
        }
        buf
    }
    fn to_lower_snake_case(s: &str) -> String {
        let mut buf = String::with_capacity(s.len());
        let mut prev = false;
        for c in s.chars() {
            if c.is_ascii_uppercase() && prev {
                buf.push('_')
            }
            prev = true;
            buf.push(c.to_ascii_lowercase());
        }
        buf
    }
    fn to_pascal_case(s: &str) -> String {
        let mut buf = String::with_capacity(s.len());
        let mut prev_is_underscore = true;
        for c in s.chars() {
            if c == '_' {
                prev_is_underscore = true;
            } else if prev_is_underscore {
                buf.push(c.to_ascii_uppercase());
                prev_is_underscore = false;
            } else {
                buf.push(c.to_ascii_lowercase());
            }
        }
        buf
    }
    fn pluralize(s: &str) -> String {
        format!("{s}s")
    }
    impl Field {
        fn is_many(&self) -> bool {
            matches!(
                self,
                Field::Node {
                    cardinality: Cardinality::Many,
                    ..
                }
            )
        }
        fn token_kind(&self) -> Option<proc_macro2::TokenStream> {
            match self {
                Field::Token(token) => {
                    let token: proc_macro2::TokenStream = token.parse().unwrap();
                    Some(quote! { T![#token] })
                },
                _ => None,
            }
        }
        fn method_name(&self) -> proc_macro2::Ident {
            match self {
                Field::Token(name) => {
                    let name = match name.as_str() {
                        ";" => "semicolon",
                        "->" => "thin_arrow",
                        "'{'" => "l_curly",
                        "'}'" => "r_curly",
                        "'('" => "l_paren",
                        "')'" => "r_paren",
                        "'['" => "l_brack",
                        "']'" => "r_brack",
                        "<" => "l_angle",
                        ">" => "r_angle",
                        "=" => "eq",
                        "!" => "excl",
                        "*" => "star",
                        "&" => "amp",
                        "-" => "minus",
                        "_" => "underscore",
                        "." => "dot",
                        ".." => "dotdot",
                        "..." => "dotdotdot",
                        "..=" => "dotdoteq",
                        "=>" => "fat_arrow",
                        "@" => "at",
                        ":" => "colon",
                        "::" => "coloncolon",
                        "#" => "pound",
                        "?" => "question_mark",
                        "," => "comma",
                        "|" => "pipe",
                        "~" => "tilde",
                        _ => name,
                    };
                    format_ident!("{}_token", name)
                },
                Field::Node { name, .. } => {
                    if name == "type" {
                        format_ident!("ty")
                    } else {
                        format_ident!("{}", name)
                    }
                },
            }
        }
        fn ty(&self) -> proc_macro2::Ident {
            match self {
                Field::Token(_) => format_ident!("syntax::Token"),
                Field::Node { ty, .. } => format_ident!("{}", ty),
            }
        }
    }
    fn lower(grammar: &Grammar) -> AstSrc {
        let mut res = AstSrc {
            tokens: "Whitespace Comment String ByteString CString IntNumber FloatNumber Char Byte Ident"
                .split_ascii_whitespace()
                .map(|x| x.to_string())
                .collect::<Vec<_>>(),
            ..Default::default()
        };
        let nodes = grammar.iter().collect::<Vec<_>>();
        for &node in &nodes {
            let name = grammar[node].name.clone();
            let rule = &grammar[node].rule;
            match lower_enum(grammar, rule) {
                Some(variants) => {
                    let enum_src = EnumSrc {
                        doc: Vec::new(),
                        name,
                        traits: Vec::new(),
                        variants,
                    };
                    res.enums.push(enum_src);
                },
                None => {
                    let mut fields = Vec::new();
                    lower_rule(&mut fields, grammar, None, rule);
                    res.nodes.push(NodeSrc {
                        doc: Vec::new(),
                        name,
                        traits: Vec::new(),
                        fields,
                    });
                },
            }
        }
        deduplicate_fields(&mut res);
        extract_enums(&mut res);
        extract_struct_traits(&mut res);
        extract_enum_traits(&mut res);
        res
    }
    fn lower_enum(grammar: &Grammar, rule: &Rule) -> Option<Vec<String>> {
        let alternatives = match rule {
            Rule::Alt(it) => it,
            _ => return None,
        };
        let mut variants = Vec::new();
        for alternative in alternatives {
            match alternative {
                Rule::Node(it) => variants.push(grammar[*it].name.clone()),
                Rule::Token(it) if grammar[*it].name == ";" => (),
                _ => return None,
            }
        }
        Some(variants)
    }
    fn lower_rule(acc: &mut Vec<Field>, grammar: &Grammar, label: Option<&String>, rule: &Rule) {
        if lower_comma_list(acc, grammar, label, rule) {
            return;
        }
        match rule {
            Rule::Node(node) => {
                let ty = grammar[*node].name.clone();
                let name = label.cloned().unwrap_or_else(|| to_lower_snake_case(&ty));
                let field = Field::Node {
                    name,
                    ty,
                    cardinality: Cardinality::Optional,
                };
                acc.push(field);
            },
            Rule::Token(token) => {
                assert!(label.is_none());
                let mut name = grammar[*token].name.clone();
                if name != "int_number" && name != "string" {
                    if "[]{}()".contains(&name) {
                        name = format!("'{name}'");
                    }
                    let field = Field::Token(name);
                    acc.push(field);
                }
            },
            Rule::Rep(inner) => {
                if let Rule::Node(node) = &**inner {
                    let ty = grammar[*node].name.clone();
                    let name = label.cloned().unwrap_or_else(|| pluralize(&to_lower_snake_case(&ty)));
                    let field = Field::Node {
                        name,
                        ty,
                        cardinality: Cardinality::Many,
                    };
                    acc.push(field);
                    return;
                }
                panic!("unhandled rule: {rule:?}")
            },
            Rule::Labeled { label: l, rule } => {
                assert!(label.is_none());
                let manually_implemented = matches!(
                    l.as_str(),
                    "lhs"
                        | "rhs"
                        | "then_branch"
                        | "else_branch"
                        | "start"
                        | "end"
                        | "op"
                        | "index"
                        | "base"
                        | "value"
                        | "trait"
                        | "self_ty"
                        | "iterable"
                        | "condition"
                );
                if manually_implemented {
                    return;
                }
                lower_rule(acc, grammar, Some(l), rule);
            },
            Rule::Seq(rules) | Rule::Alt(rules) => {
                for rule in rules {
                    lower_rule(acc, grammar, label, rule)
                }
            },
            Rule::Opt(rule) => lower_rule(acc, grammar, label, rule),
        }
    }
    fn lower_comma_list(acc: &mut Vec<Field>, grammar: &Grammar, label: Option<&String>, rule: &Rule) -> bool {
        let rule = match rule {
            Rule::Seq(it) => it,
            _ => return false,
        };
        let (node, repeat, trailing_comma) = match rule.as_slice() {
            [Rule::Node(node), Rule::Rep(repeat), Rule::Opt(trailing_comma)] => (node, repeat, trailing_comma),
            _ => return false,
        };
        let repeat = match &**repeat {
            Rule::Seq(it) => it,
            _ => return false,
        };
        match repeat.as_slice() {
            [comma, Rule::Node(n)] if comma == &**trailing_comma && n == node => (),
            _ => return false,
        }
        let ty = grammar[*node].name.clone();
        let name = label.cloned().unwrap_or_else(|| pluralize(&to_lower_snake_case(&ty)));
        let field = Field::Node {
            name,
            ty,
            cardinality: Cardinality::Many,
        };
        acc.push(field);
        true
    }
    fn deduplicate_fields(ast: &mut AstSrc) {
        for node in &mut ast.nodes {
            let mut i = 0;
            'outer: while i < node.fields.len() {
                for j in 0..i {
                    let f1 = &node.fields[i];
                    let f2 = &node.fields[j];
                    if f1 == f2 {
                        node.fields.remove(i);
                        continue 'outer;
                    }
                }
                i += 1;
            }
        }
    }
    fn extract_enums(ast: &mut AstSrc) {
        for node in &mut ast.nodes {
            for enm in &ast.enums {
                let mut to_remove = Vec::new();
                for (i, field) in node.fields.iter().enumerate() {
                    let ty = field.ty().to_string();
                    if enm.variants.iter().any(|x| x == &ty) {
                        to_remove.push(i);
                    }
                }
                if to_remove.len() == enm.variants.len() {
                    node.remove_field(to_remove);
                    let ty = enm.name.clone();
                    let name = to_lower_snake_case(&ty);
                    node.fields.push(Field::Node {
                        name,
                        ty,
                        cardinality: Cardinality::Optional,
                    });
                }
            }
        }
    }
    fn extract_struct_traits(ast: &mut AstSrc) {
        let traits: &[(&str, &[&str])] = &[
            ("HasAttrs", &["attrs"]),
            ("HasName", &["name"]),
            ("HasVisibility", &["visibility"]),
            ("HasGenericParams", &["generic_param_list", "where_clause"]),
            ("HasTypeBounds", &["type_bound_list", "colon_token"]),
            ("HasModuleItem", &["items"]),
            ("HasLoopBody", &["label", "loop_body"]),
            ("HasArgList", &["arg_list"]),
        ];
        for node in &mut ast.nodes {
            for (name, methods) in traits {
                extract_struct_trait(node, name, methods);
            }
        }
        let nodes_with_doc_comments = [
            "SourceFile",
            "Fn",
            "Struct",
            "Union",
            "RecordField",
            "TupleField",
            "Enum",
            "Variant",
            "Trait",
            "TraitAlias",
            "Module",
            "Static",
            "Const",
            "TypeAlias",
            "Impl",
            "ExternBlock",
            "ExternCrate",
            "MacroCall",
            "MacroRules",
            "MacroDef",
            "Use",
        ];
        for node in &mut ast.nodes {
            if nodes_with_doc_comments.contains(&&*node.name) {
                node.traits.push("HasDocComments".into());
            }
        }
    }
    fn extract_struct_trait(node: &mut NodeSrc, trait_name: &str, methods: &[&str]) {
        let mut to_remove = Vec::new();
        for (i, field) in node.fields.iter().enumerate() {
            let method_name = field.method_name().to_string();
            if methods.iter().any(|&it| it == method_name) {
                to_remove.push(i);
            }
        }
        if to_remove.len() == methods.len() {
            node.traits.push(trait_name.to_string());
            node.remove_field(to_remove);
        }
    }
    fn extract_enum_traits(ast: &mut AstSrc) {
        for enm in &mut ast.enums {
            if enm.name == "Stmt" {
                continue;
            }
            let nodes = &ast.nodes;
            let mut variant_traits = enm
                .variants
                .iter()
                .map(|var| nodes.iter().find(|x| &x.name == var).unwrap())
                .map(|node| node.traits.iter().cloned().collect::<BTreeSet<_>>());
            let mut enum_traits = match variant_traits.next() {
                Some(it) => it,
                None => continue,
            };
            for traits in variant_traits {
                enum_traits = enum_traits.intersection(&traits).cloned().collect();
            }
            enm.traits = enum_traits.into_iter().collect();
        }
    }
    impl NodeSrc {
        fn remove_field(&mut self, to_remove: Vec<usize>) {
            to_remove.into_iter().rev().for_each(|idx| {
                self.fields.remove(idx);
            });
        }
    }
}

#[test]
fn parse_smoke_test() {
    let code = r##"
fn main() {
    println!("Hello, world!")
}
    "##;

    let parse = SourceFile::parse(code);
    assert!(parse.ok().is_ok());
}

#[test]
fn benchmark_parser() {
    if std::env::var("RUN_SLOW_BENCHES").is_err() {
        return;
    }

    let data = bench_fixture::glorious_old_parser();
    let tree = {
        let _b = bench("parsing");
        let p = SourceFile::parse(&data);
        assert!(p.errors.is_empty());
        assert_eq!(p.tree().syntax.text_range().len(), 352474.into());
        p.tree()
    };

    {
        let _b = bench("tree traversal");
        let fn_names = tree
            .syntax()
            .descendants()
            .filter_map(ast::Fn::cast)
            .filter_map(|f| f.name())
            .count();
        assert_eq!(fn_names, 268);
    }
}

#[test]
fn validate_tests() {
    dir_tests(&test_data_dir(), &["parser/validate"], "rast", |text, path| {
        let parse = SourceFile::parse(text);
        let errors = parse.errors();
        assert_errors_are_present(errors, path);
        parse.debug_dump()
    });
}

#[test]
fn parser_fuzz_tests() {
    for (_, text) in collect_rust_files(&test_data_dir(), &["parser/fuzz-failures"]) {
        fuzz::check_parser(&text)
    }
}

#[test]
fn reparse_fuzz_tests() {
    for (_, text) in collect_rust_files(&test_data_dir(), &["reparse/fuzz-failures"]) {
        let check = fuzz::CheckReparse::from_data(text.as_bytes()).unwrap();
        check.run();
    }
}

#[test]
fn self_hosting_parsing() {
    let crates_dir = project_root().join("crates");

    let mut files = ::sourcegen::list_rust_files(&crates_dir);
    files.retain(|path| !path.components().any(|component| component.as_os_str() == "test_data"));

    assert!(
        files.len() > 100,
        "self_hosting_parsing found too few files - is it running in the right directory?"
    );

    let errors = files
        .into_par_iter()
        .filter_map(|file| {
            let text = read_text(&file);
            match SourceFile::parse(&text).ok() {
                Ok(_) => None,
                Err(err) => Some((file, err)),
            }
        })
        .collect::<Vec<_>>();

    if !errors.is_empty() {
        let errors = errors
            .into_iter()
            .map(|(path, err)| format!("{}: {:?}\n", path.display(), err[0]))
            .collect::<String>();
        panic!("Parsing errors:\n{errors}\n");
    }
}

fn test_data_dir() -> PathBuf {
    project_root().join("crates/syntax/test_data")
}

fn assert_errors_are_present(errors: &[SyntaxErr], path: &Path) {
    assert!(
        !errors.is_empty(),
        "There should be errors in the file {:?}",
        path.display()
    );
}

fn dir_tests<F>(test_data_dir: &Path, paths: &[&str], outfile_extension: &str, f: F)
where
    F: Fn(&str, &Path) -> String,
{
    for (path, input_code) in collect_rust_files(test_data_dir, paths) {
        let actual = f(&input_code, &path);
        let path = path.with_extension(outfile_extension);
        expect_file![path].assert_eq(&actual)
    }
}

fn collect_rust_files(root_dir: &Path, paths: &[&str]) -> Vec<(PathBuf, String)> {
    paths
        .iter()
        .flat_map(|path| {
            let path = root_dir.to_owned().join(path);
            rust_files_in_dir(&path).into_iter()
        })
        .map(|path| {
            let text = read_text(&path);
            (path, text)
        })
        .collect()
}

fn rust_files_in_dir(dir: &Path) -> Vec<PathBuf> {
    let mut acc = Vec::new();
    for file in fs::read_dir(dir).unwrap() {
        let file = file.unwrap();
        let path = file.path();
        if path.extension().unwrap_or_default() == "rs" {
            acc.push(path);
        }
    }
    acc.sort();
    acc
}

fn read_text(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("File at {path:?} should be valid"))
        .replace("\r\n", "\n")
}
