use expect_test::expect_file;
use itertools::Itertools;
use parser::syntax::{ast, fuzz, SourceFile, SyntaxErr};
use proc_macro2::{Punct, Spacing};
use quote::{format_ident, quote};
use rayon::prelude::*;
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fmt::{self, Write},
    fs, iter, mem,
    path::{Path, PathBuf},
};
use test_utils::{bench, bench_fixture, project_root};
use ungrammar::{Grammar, Rule};
use xshell::{cmd, Shell};

#[test]
fn self_parsing() {
    let x = project_root().join("crates");
    let mut ys = list_files(&x);
    ys.retain(|x| !x.components().any(|x| x.as_os_str() == "test_data"));
    assert!(ys.len() > 100);
    let ys = ys
        .into_par_iter()
        .filter_map(|x| {
            let y = read_text(&x);
            match SourceFile::parse(&y).ok() {
                Ok(_) => None,
                Err(err) => Some((x, err)),
            }
        })
        .collect::<Vec<_>>();
    if !ys.is_empty() {
        let ys = ys
            .into_iter()
            .map(|(k, v)| format!("{}: {:?}\n", k.display(), v[0]))
            .collect::<String>();
        panic!("Parsing errors:\n{ys}\n");
    }
}
fn list_files(x: &Path) -> Vec<PathBuf> {
    fn list_all(x: &Path) -> Vec<PathBuf> {
        let mut ys = Vec::new();
        let mut xs = vec![x.to_path_buf()];
        while let Some(x) = xs.pop() {
            for x in x.read_dir().unwrap() {
                let x = x.unwrap();
                let y = x.path();
                let is_hidden = y
                    .file_name()
                    .unwrap_or_default()
                    .to_str()
                    .unwrap_or_default()
                    .starts_with('.');
                if !is_hidden {
                    let x = x.file_type().unwrap();
                    if x.is_dir() {
                        xs.push(y);
                    } else if x.is_file() {
                        ys.push(y);
                    }
                }
            }
        }
        ys
    }
    let mut ys = list_all(x);
    ys.retain(|x| {
        x.file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or_default()
            .ends_with(".rs")
    });
    ys
}

#[derive(Debug)]
struct Test {
    name: String,
    text: String,
    ok: bool,
}

#[derive(Default, Debug)]
struct Tests {
    ok: HashMap<String, Test>,
    err: HashMap<String, Test>,
}

#[test]
fn parser_tests() {
    fn all_from(x: &Path) -> Tests {
        let mut y = Tests::default();
        let grammar = x.parent().unwrap().join("grammar.rs");
        fn process(y: &mut Tests, x: &Path) {
            let x = fs::read_to_string(x).unwrap();
            for x in collect_tests(&x) {
                if x.ok {
                    if let Some(x) = y.ok.insert(x.name.clone(), x) {
                        panic!("Duplicate test: {}", x.name);
                    }
                } else if let Some(x) = y.err.insert(x.name.clone(), x) {
                    panic!("Duplicate test: {}", x.name);
                }
            }
        }
        for x in list_files(x) {
            process(&mut y, x.as_path());
        }
        process(&mut y, &grammar);
        return y;
    }
    fn install(x: &HashMap<String, Test>, into: &str) {
        let into = project_root().join(into);
        if !into.is_dir() {
            fs::create_dir_all(&into).unwrap();
        }
        let ys = existing(&into, true);
        for y in ys.keys().filter(|&y| !x.contains_key(y)) {
            panic!("Test is deleted: {y}");
        }
        let mut idx = ys.len() + 1;
        for (k, v) in x {
            let path = match ys.get(k) {
                Some((path, _)) => path.clone(),
                None => {
                    let x = format!("{idx:04}_{k}.rs");
                    idx += 1;
                    into.join(x)
                },
            };
            ensure_contents(&path, &v.text);
        }
    }
    fn existing(x: &Path, ok: bool) -> HashMap<String, (PathBuf, Test)> {
        let mut y = HashMap::default();
        for x in fs::read_dir(x).unwrap() {
            let x = x.unwrap().path();
            if x.extension().unwrap_or_default() != "rs" {
                continue;
            }
            let k = {
                let x = x.file_name().unwrap().to_str().unwrap();
                x[5..x.len() - 3].to_string()
            };
            let text = fs::read_to_string(&x).unwrap();
            let test = Test {
                name: k.clone(),
                text,
                ok,
            };
            if let Some(x) = y.insert(k, (x, test)) {
                println!("Duplicate test: {x:?}");
            }
        }
        y
    }
    let x = project_root().join(Path::new("crates/parser/src/grammar"));
    let y = all_from(&x);
    install(&y.ok, "crates/parser/test_data/parser/inline/ok");
    install(&y.err, "crates/parser/test_data/parser/inline/err");
}

pub fn project_root() -> PathBuf {
    let y = env!("CARGO_MANIFEST_DIR");
    let y = PathBuf::from(y).parent().unwrap().parent().unwrap().to_owned();
    assert!(y.join("triagebot.toml").exists());
    y
}

fn ensure_contents(p: &Path, x: &str) {
    if let Ok(y) = fs::read_to_string(p) {
        if normalize(&y) == normalize(x) {
            return;
        }
    }
    let p2 = p.strip_prefix(project_root()).unwrap_or(p);
    eprintln!("\n{} was not up-to-date, updating\n", p2.display());
    eprintln!("NOTE: run `cargo test` locally and commit the updated files\n");
    if let Some(x) = p.parent() {
        let _ = fs::create_dir_all(x);
    }
    fs::write(p, x).unwrap();
    fn normalize(x: &str) -> String {
        x.replace("\r\n", "\n")
    }
}

#[derive(Clone)]
pub struct CommentBlock {
    pub id: String,
    pub line: usize,
    pub texts: Vec<String>,
    is_doc: bool,
}
impl CommentBlock {
    pub fn extract(tag: &str, x: &str) -> Vec<CommentBlock> {
        assert!(tag.starts_with(char::is_uppercase));
        let tag = format!("{tag}:");
        let mut ys = CommentBlock::untagged(x);
        ys.retain_mut(|x| {
            let first = x.texts.remove(0);
            let Some(id) = first.strip_prefix(&tag) else {
                return false;
            };
            if x.is_doc {
                panic!("Use plain (non-doc) comments with tags like {tag}:\n    {first}");
            }
            x.id = id.trim().to_string();
            true
        });
        ys
    }
    pub fn untagged(x: &str) -> Vec<CommentBlock> {
        let mut ys = Vec::new();
        let xs = x.lines().map(str::trim_start);
        let dummy = CommentBlock {
            id: String::new(),
            line: 0,
            texts: Vec::new(),
            is_doc: false,
        };
        let mut y = dummy.clone();
        for (i, x) in xs.enumerate() {
            match x.strip_prefix("//") {
                Some(mut x) => {
                    if let Some('/' | '!') = x.chars().next() {
                        x = &x[1..];
                        y.is_doc = true;
                    }
                    if let Some(' ') = x.chars().next() {
                        x = &x[1..];
                    }
                    y.texts.push(x.to_string());
                },
                None => {
                    if !y.texts.is_empty() {
                        let y = mem::replace(&mut y, dummy.clone());
                        ys.push(y);
                    }
                    y.line = i + 2;
                },
            }
        }
        if !y.texts.is_empty() {
            ys.push(y);
        }
        ys
    }
}

fn collect_tests(x: &str) -> Vec<Test> {
    let mut ys = Vec::new();
    for x in CommentBlock::untagged(x) {
        let first = &x.texts[0];
        let (name, ok) = if let Some(name) = first.strip_prefix("test ") {
            (name.to_string(), true)
        } else if let Some(name) = first.strip_prefix("test_err ") {
            (name.to_string(), false)
        } else {
            continue;
        };
        let text: String = x.texts[1..]
            .iter()
            .cloned()
            .chain(iter::once(String::new()))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.trim().is_empty() && text.ends_with('\n'));
        ys.push(Test { name, text, ok })
    }
    ys
}

#[derive(Debug)]
pub struct Location {
    pub file: PathBuf,
    pub line: usize,
}
impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = self.file.strip_prefix(project_root()).unwrap().display().to_string();
        let p = p.replace('\\', "/");
        let n = self.file.file_name().unwrap();
        write!(
            f,
            "https://github.com/rust-lang/rust-analyzer/blob/master/{}#L{}[{}]",
            p,
            self.line,
            n.to_str().unwrap()
        )
    }
}

fn add_preamble(gen: &'static str, mut y: String) -> String {
    let x = format!("//! Generated by `{gen}`, do not edit by hand.\n\n");
    y.insert_str(0, &x);
    y
}

fn reformat(x: String) -> String {
    fn ensure_rustfmt(x: &Shell) {
        let y = cmd!(x, "rustup run stable rustfmt --version")
            .read()
            .unwrap_or_default();
        if !y.contains("stable") {
            panic!("Failed to run rustfmt from toolchain 'stable'.",);
        }
    }
    let sh = Shell::new().unwrap();
    ensure_rustfmt(&sh);
    let rustfmt_toml = project_root().join("rustfmt.toml");
    let mut y = cmd!(
        sh,
        "rustup run stable rustfmt --config-path {rustfmt_toml} --config fn_single_line=true"
    )
    .stdin(x)
    .read()
    .unwrap();
    if !y.ends_with('\n') {
        y.push('\n');
    }
    y
}

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
        "as", "async", "await", "box", "break", "const", "continue", "crate", "do", "dyn", "else", "enum", "extern",
        "false", "fn", "for", "if", "impl", "in", "let", "loop", "macro", "match", "mod", "move", "mut", "pub", "ref",
        "return", "self", "Self", "static", "struct", "super", "trait", "true", "try", "type", "unsafe", "use",
        "where", "while", "yield",
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

#[test]
fn gen_ast() {
    let x = gen_kinds(KINDS_SRC);
    let y = project_root().join("crates/parser/src/syntax_kind/generated.rs");
    ensure_contents(y.as_path(), &x);
    let g = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/rust.ungram"))
        .parse()
        .unwrap();
    let g = lower(&g);
    let x = gen_toks(&g);
    let y = project_root().join("src/syntax/ast/token_gen.rs");
    ensure_contents(y.as_path(), &x);
    let x = gen_nodes(KINDS_SRC, &g);
    let y = project_root().join("crates/syntax/src/ast/generated/nodes.rs");
    ensure_contents(y.as_path(), &x);
    fn gen_kinds(grammar: KindsSrc<'_>) -> String {
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
        add_preamble("sourcegen_ast", reformat(ast.to_string()))
    }
    fn gen_toks(x: &AstSrc) -> String {
        let ys = x.tokens.iter().map(|x| {
            let n = format_ident!("{}", x);
            let k = format_ident!("{}", to_upper_snake_case(x));
            quote! {
                #[derive(Debug, Clone, PartialEq, Eq, Hash)]
                pub struct #n {
                    pub syntax: syntax::Token,
                }
                impl fmt::Display for #n {
                    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        fmt::Display::fmt(&self.syntax, f)
                    }
                }
                impl ast::Token for #n {
                    fn can_cast(x: SyntaxKind) -> bool { x == #k }
                    fn cast(syntax: syntax::Token) -> Option<Self> {
                        if Self::can_cast(syntax.kind()) { Some(Self { syntax }) } else { None }
                    }
                    fn syntax(&self) -> &syntax::Token { &self.syntax }
                }
            }
        });
        add_preamble(
            "gen_ast",
            reformat(
                quote! {
                    use crate::{
                        syntax::{self, ast},
                        SyntaxKind::{self, *},
                    };
                    use std::fmt;
                    #(#ys)*
                }
                .to_string(),
            ),
        )
        .replace("#[derive", "\n#[derive")
    }
    fn gen_nodes(kinds: KindsSrc<'_>, grammar: &AstSrc) -> String {
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
        let res = add_preamble("sourcegen_ast", reformat(res));
        res.replace("#[derive", "\n#[derive")
    }
}
fn write_doc_comment(contents: &[String], dest: &mut String) {
    for line in contents {
        writeln!(dest, "///{line}").unwrap();
    }
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
