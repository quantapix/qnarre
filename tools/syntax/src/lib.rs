#![forbid(
    // missing_debug_implementations,
    unconditional_recursion,
    future_incompatible,
    // missing_docs,
)]
#![warn(unused_lifetimes)]
#[allow(unused)]

macro_rules! _eprintln {
    ($($tt:tt)*) => {{
        if $crate::is_ci() {
            panic!("Forgot to remove debug-print?")
        }
        std::eprintln!($($tt)*)
    }}
}

#[macro_export]
macro_rules! eprintln {
    ($($tt:tt)*) => { _eprintln!($($tt)*) };
}

#[macro_export]
macro_rules! format_to {
    ($buf:expr) => ();
    ($buf:expr, $lit:literal $($arg:tt)*) => {
        { use ::std::fmt::Write as _; let _ = ::std::write!($buf, $lit $($arg)*); }
    };
}

#[macro_export]
macro_rules! impl_from {
    ($($variant:ident $(($($sub_variant:ident),*))?),* for $enum:ident) => {
        $(
            impl From<$variant> for $enum {
                fn from(it: $variant) -> $enum {
                    $enum::$variant(it)
                }
            }
            $($(
                impl From<$sub_variant> for $enum {
                    fn from(it: $sub_variant) -> $enum {
                        $enum::$variant($variant::$sub_variant(it))
                    }
                }
            )*)?
        )*
    };
    ($($variant:ident$(<$V:ident>)?),* for $enum:ident) => {
        $(
            impl$(<$V>)? From<$variant$(<$V>)?> for $enum$(<$V>)? {
                fn from(it: $variant$(<$V>)?) -> $enum$(<$V>)? {
                    $enum::$variant(it)
                }
            }
        )*
    }
}


use std::marker::PhantomData;
use text_edit::Indel;
use triomphe::Arc;

pub use crate::{
    ast::{AstNode, AstToken},
    ptr::{AstPtr, SyntaxNodePtr},
    syntax_error::SyntaxError,
    syntax_node::{
        PreorderWithTokens, RustLanguage, SyntaxElement, SyntaxElementChildren, SyntaxNode, SyntaxNodeChildren,
        SyntaxToken, SyntaxTreeBuilder,
    },
    token_text::TokenText,
};
pub use parser::{SyntaxKind, T};
pub use rowan::{
    api::Preorder, Direction, GreenNode, NodeOrToken, SyntaxText, TextRange, TextSize, TokenAtOffset, WalkEvent,
};
pub use smol_str::SmolStr;

pub mod algo;
pub mod ast;
pub mod rowan;
#[doc(hidden)]
pub mod fuzz {
    use crate::{validation, AstNode, SourceFile, TextRange};
    use std::str::{self, FromStr};
    use text_edit::Indel;

    fn check_file_invariants(file: &SourceFile) {
        let root = file.syntax();
        validation::validate_block_structure(root);
    }
    pub fn check_parser(text: &str) {
        let file = SourceFile::parse(text);
        check_file_invariants(&file.tree());
    }
    #[derive(Debug, Clone)]
    pub struct CheckReparse {
        text: String,
        edit: Indel,
        edited_text: String,
    }
    impl CheckReparse {
        pub fn from_data(data: &[u8]) -> Option<Self> {
            const PREFIX: &str = "fn main(){\n\t";
            const SUFFIX: &str = "\n}";
            let data = str::from_utf8(data).ok()?;
            let mut lines = data.lines();
            let delete_start = usize::from_str(lines.next()?).ok()? + PREFIX.len();
            let delete_len = usize::from_str(lines.next()?).ok()?;
            let insert = lines.next()?.to_string();
            let text = lines.collect::<Vec<_>>().join("\n");
            let text = format!("{PREFIX}{text}{SUFFIX}");
            text.get(delete_start..delete_start.checked_add(delete_len)?)?;
            let delete = TextRange::at(delete_start.try_into().unwrap(), delete_len.try_into().unwrap());
            let edited_text = format!(
                "{}{}{}",
                &text[..delete_start],
                &insert,
                &text[delete_start + delete_len..]
            );
            let edit = Indel { insert, delete };
            Some(CheckReparse {
                text,
                edit,
                edited_text,
            })
        }
        pub fn run(&self) {
            let parse = SourceFile::parse(&self.text);
            let new_parse = parse.reparse(&self.edit);
            check_file_invariants(&new_parse.tree());
            assert_eq!(&new_parse.tree().syntax().text().to_string(), &self.edited_text);
            let full_reparse = SourceFile::parse(&self.edited_text);
            for (a, b) in new_parse
                .tree()
                .syntax()
                .descendants()
                .zip(full_reparse.tree().syntax().descendants())
            {
                if (a.kind(), a.text_range()) != (b.kind(), b.text_range()) {
                    eprint!("original:\n{:#?}", parse.tree().syntax());
                    eprint!("reparsed:\n{:#?}", new_parse.tree().syntax());
                    eprint!("full reparse:\n{:#?}", full_reparse.tree().syntax());
                    assert_eq!(
                        format!("{a:?}"),
                        format!("{b:?}"),
                        "different syntax tree produced by the full reparse"
                    );
                }
            }
        }
    }
}
pub mod hacks {
    use crate::{ast, AstNode};
    pub fn parse_expr_from_str(s: &str) -> Option<ast::Expr> {
        let s = s.trim();
        let file = ast::SourceFile::parse(&format!("const _: () = {s};"));
        let expr = file.syntax_node().descendants().find_map(ast::Expr::cast)?;
        if expr.syntax().text() != s {
            return None;
        }
        Some(expr)
    }
}
mod parsing {
    mod reparsing {
        use parser::Reparser;
        use text_edit::Indel;

        use crate::{
            parsing::build_tree,
            syntax_node::{GreenNode, GreenToken, NodeOrToken, SyntaxElement, SyntaxNode},
            SyntaxError,
            SyntaxKind::*,
            TextRange, TextSize, T,
        };

        pub fn incremental_reparse(
            node: &SyntaxNode,
            edit: &Indel,
            errors: Vec<SyntaxError>,
        ) -> Option<(GreenNode, Vec<SyntaxError>, TextRange)> {
            if let Some((green, new_errors, old_range)) = reparse_token(node, edit) {
                return Some((green, merge_errors(errors, new_errors, old_range, edit), old_range));
            }
            if let Some((green, new_errors, old_range)) = reparse_block(node, edit) {
                return Some((green, merge_errors(errors, new_errors, old_range, edit), old_range));
            }
            None
        }
        fn reparse_token(root: &SyntaxNode, edit: &Indel) -> Option<(GreenNode, Vec<SyntaxError>, TextRange)> {
            let prev_token = root.covering_element(edit.delete).as_token()?.clone();
            let prev_token_kind = prev_token.kind();
            match prev_token_kind {
                WHITESPACE | COMMENT | IDENT | STRING | BYTE_STRING | C_STRING => {
                    if prev_token_kind == WHITESPACE || prev_token_kind == COMMENT {
                        let deleted_range = edit.delete - prev_token.text_range().start();
                        if prev_token.text()[deleted_range].contains('\n') {
                            return None;
                        }
                    }
                    let mut new_text = get_text_after_edit(prev_token.clone().into(), edit);
                    let (new_token_kind, new_err) = parser::LexedStr::single_token(&new_text)?;
                    if new_token_kind != prev_token_kind || (new_token_kind == IDENT && is_contextual_kw(&new_text)) {
                        return None;
                    }
                    if let Some(next_char) = root.text().char_at(prev_token.text_range().end()) {
                        new_text.push(next_char);
                        let token_with_next_char = parser::LexedStr::single_token(&new_text);
                        if let Some((_kind, _error)) = token_with_next_char {
                            return None;
                        }
                        new_text.pop();
                    }
                    let new_token = GreenToken::new(rowan::SyntaxKind(prev_token_kind.into()), &new_text);
                    let range = TextRange::up_to(TextSize::of(&new_text));
                    Some((
                        prev_token.replace_with(new_token),
                        new_err.into_iter().map(|msg| SyntaxError::new(msg, range)).collect(),
                        prev_token.text_range(),
                    ))
                },
                _ => None,
            }
        }
        fn reparse_block(root: &SyntaxNode, edit: &Indel) -> Option<(GreenNode, Vec<SyntaxError>, TextRange)> {
            let (node, reparser) = find_reparsable_node(root, edit.delete)?;
            let text = get_text_after_edit(node.clone().into(), edit);
            let lexed = parser::LexedStr::new(text.as_str());
            let parser_input = lexed.to_input();
            if !is_balanced(&lexed) {
                return None;
            }
            let tree_traversal = reparser.parse(&parser_input);
            let (green, new_parser_errors, _eof) = build_tree(lexed, tree_traversal);
            Some((node.replace_with(green), new_parser_errors, node.text_range()))
        }
        fn get_text_after_edit(element: SyntaxElement, edit: &Indel) -> String {
            let edit = Indel::replace(edit.delete - element.text_range().start(), edit.insert.clone());
            let mut text = match element {
                NodeOrToken::Token(token) => token.text().to_string(),
                NodeOrToken::Node(node) => node.text().to_string(),
            };
            edit.apply(&mut text);
            text
        }
        fn is_contextual_kw(text: &str) -> bool {
            matches!(text, "auto" | "default" | "union")
        }
        fn find_reparsable_node(node: &SyntaxNode, range: TextRange) -> Option<(SyntaxNode, Reparser)> {
            let node = node.covering_element(range);
            node.ancestors().find_map(|node| {
                let first_child = node.first_child_or_token().map(|it| it.kind());
                let parent = node.parent().map(|it| it.kind());
                Reparser::for_node(node.kind(), first_child, parent).map(|r| (node, r))
            })
        }
        fn is_balanced(lexed: &parser::LexedStr<'_>) -> bool {
            if lexed.is_empty() || lexed.kind(0) != T!['{'] || lexed.kind(lexed.len() - 1) != T!['}'] {
                return false;
            }
            let mut balance = 0usize;
            for i in 1..lexed.len() - 1 {
                match lexed.kind(i) {
                    T!['{'] => balance += 1,
                    T!['}'] => {
                        balance = match balance.checked_sub(1) {
                            Some(b) => b,
                            None => return false,
                        }
                    },
                    _ => (),
                }
            }
            balance == 0
        }
        fn merge_errors(
            old_errors: Vec<SyntaxError>,
            new_errors: Vec<SyntaxError>,
            range_before_reparse: TextRange,
            edit: &Indel,
        ) -> Vec<SyntaxError> {
            let mut res = Vec::new();
            for old_err in old_errors {
                let old_err_range = old_err.range();
                if old_err_range.end() <= range_before_reparse.start() {
                    res.push(old_err);
                } else if old_err_range.start() >= range_before_reparse.end() {
                    let inserted_len = TextSize::of(&edit.insert);
                    res.push(old_err.with_range((old_err_range + inserted_len) - edit.delete.len()));
                }
            }
            res.extend(new_errors.into_iter().map(|new_err| {
                let offsetted_range = new_err.range() + range_before_reparse.start();
                new_err.with_range(offsetted_range)
            }));
            res
        }
        #[cfg(test)]
        mod tests {
            use super::*;
            use crate::{AstNode, Parse, SourceFile};
            use test_utils::{assert_eq_text, extract_range};
            fn do_check(before: &str, replace_with: &str, reparsed_len: u32) {
                let (range, before) = extract_range(before);
                let edit = Indel::replace(range, replace_with.to_owned());
                let after = {
                    let mut after = before.clone();
                    edit.apply(&mut after);
                    after
                };
                let fully_reparsed = SourceFile::parse(&after);
                let incrementally_reparsed: Parse<SourceFile> = {
                    let before = SourceFile::parse(&before);
                    let (green, new_errors, range) =
                        incremental_reparse(before.tree().syntax(), &edit, before.errors.to_vec()).unwrap();
                    assert_eq!(range.len(), reparsed_len.into(), "reparsed fragment has wrong length");
                    Parse::new(green, new_errors)
                };
                assert_eq_text!(
                    &format!("{:#?}", fully_reparsed.tree().syntax()),
                    &format!("{:#?}", incrementally_reparsed.tree().syntax()),
                );
                assert_eq!(fully_reparsed.errors(), incrementally_reparsed.errors());
            }
            #[test]
            fn reparse_block_tests() {
                do_check(
                    r"
fn foo() {
    let x = foo + $0bar$0
}
",
                    "baz",
                    3,
                );
                do_check(
                    r"
fn foo() {
    let x = foo$0 + bar$0
}
",
                    "baz",
                    25,
                );
                do_check(
                    r"
struct Foo {
    f: foo$0$0
}
",
                    ",\n    g: (),",
                    14,
                );
                do_check(
                    r"
fn foo {
    let;
    1 + 1;
    $092$0;
}
",
                    "62",
                    31,
                );
                do_check(
                    r"
mod foo {
    fn $0$0
}
",
                    "bar",
                    11,
                );
                do_check(
                    r"
trait Foo {
    type $0Foo$0;
}
",
                    "Output",
                    3,
                );
                do_check(
                    r"
impl IntoIterator<Item=i32> for Foo {
    f$0$0
}
",
                    "n next(",
                    9,
                );
                do_check(r"use a::b::{foo,$0,bar$0};", "baz", 10);
                do_check(
                    r"
pub enum A {
    Foo$0$0
}
",
                    "\nBar;\n",
                    11,
                );
                do_check(
                    r"
foo!{a, b$0$0 d}
",
                    ", c[3]",
                    8,
                );
                do_check(
                    r"
fn foo() {
    vec![$0$0]
}
",
                    "123",
                    14,
                );
                do_check(
                    r"
extern {
    fn$0;$0
}
",
                    " exit(code: c_int)",
                    11,
                );
            }
            #[test]
            fn reparse_token_tests() {
                do_check(
                    r"$0$0
fn foo() -> i32 { 1 }
",
                    "\n\n\n   \n",
                    1,
                );
                do_check(
                    r"
fn foo() -> $0$0 {}
",
                    "  \n",
                    2,
                );
                do_check(
                    r"
fn $0foo$0() -> i32 { 1 }
",
                    "bar",
                    3,
                );
                do_check(
                    r"
fn foo$0$0foo() {  }
",
                    "bar",
                    6,
                );
                do_check(
                    r"
fn foo /* $0$0 */ () {}
",
                    "some comment",
                    6,
                );
                do_check(
                    r"
fn baz $0$0 () {}
",
                    "    \t\t\n\n",
                    2,
                );
                do_check(
                    r"
fn baz $0$0 () {}
",
                    "    \t\t\n\n",
                    2,
                );
                do_check(
                    r"
/// foo $0$0omment
mod { }
",
                    "c",
                    14,
                );
                do_check(
                    r#"
fn -> &str { "Hello$0$0" }
"#,
                    ", world",
                    7,
                );
                do_check(
                    r#"
fn -> &str { // "Hello$0$0"
"#,
                    ", world",
                    10,
                );
                do_check(
                    r##"
fn -> &str { r#"Hello$0$0"#
"##,
                    ", world",
                    10,
                );
                do_check(
                    r"
#[derive($0Copy$0)]
enum Foo {
}
",
                    "Clone",
                    4,
                );
            }
            #[test]
            fn reparse_str_token_with_error_unchanged() {
                do_check(r#""$0Unclosed$0 string literal"#, "Still unclosed", 24);
            }
            #[test]
            fn reparse_str_token_with_error_fixed() {
                do_check(r#""unterminated$0$0"#, "\"", 13);
            }
            #[test]
            fn reparse_block_with_error_in_middle_unchanged() {
                do_check(
                    r#"fn main() {
                if {}
                32 + 4$0$0
                return
                if {}
            }"#,
                    "23",
                    105,
                )
            }
            #[test]
            fn reparse_block_with_error_in_middle_fixed() {
                do_check(
                    r#"fn main() {
                if {}
                32 + 4$0$0
                return
                if {}
            }"#,
                    ";",
                    105,
                )
            }
        }
    }
    pub use crate::parsing::reparsing::incremental_reparse;
    use crate::{syntax_node::GreenNode, SyntaxError, SyntaxTreeBuilder};
    use rowan::TextRange;
    pub fn parse_text(text: &str) -> (GreenNode, Vec<SyntaxError>) {
        let lexed = parser::LexedStr::new(text);
        let parser_input = lexed.to_input();
        let parser_output = parser::TopEntryPoint::SourceFile.parse(&parser_input);
        let (node, errors, _eof) = build_tree(lexed, parser_output);
        (node, errors)
    }
    pub fn build_tree(
        lexed: parser::LexedStr<'_>,
        parser_output: parser::Output,
    ) -> (GreenNode, Vec<SyntaxError>, bool) {
        let mut builder = SyntaxTreeBuilder::default();
        let is_eof = lexed.intersperse_trivia(&parser_output, &mut |step| match step {
            parser::StrStep::Token { kind, text } => builder.token(kind, text),
            parser::StrStep::Enter { kind } => builder.start_node(kind),
            parser::StrStep::Exit => builder.finish_node(),
            parser::StrStep::Error { msg, pos } => builder.error(msg.to_string(), pos.try_into().unwrap()),
        });
        let (node, mut errors) = builder.finish_raw();
        for (i, err) in lexed.errors() {
            let text_range = lexed.text_range(i);
            let text_range = TextRange::new(text_range.start.try_into().unwrap(), text_range.end.try_into().unwrap());
            errors.push(SyntaxError::new(err, text_range))
        }
        (node, errors, is_eof)
    }
}
mod ptr {
    use crate::{syntax_node::RustLanguage, AstNode, SyntaxNode};
    use rowan::TextRange;
    use std::{
        hash::{Hash, Hasher},
        marker::PhantomData,
    };
    pub type SyntaxNodePtr = rowan::ast::SyntaxNodePtr<RustLanguage>;
    #[derive(Debug)]
    pub struct AstPtr<N: AstNode> {
        raw: SyntaxNodePtr,
        _ty: PhantomData<fn() -> N>,
    }
    impl<N: AstNode> Clone for AstPtr<N> {
        fn clone(&self) -> AstPtr<N> {
            AstPtr {
                raw: self.raw.clone(),
                _ty: PhantomData,
            }
        }
    }
    impl<N: AstNode> Eq for AstPtr<N> {}
    impl<N: AstNode> PartialEq for AstPtr<N> {
        fn eq(&self, other: &AstPtr<N>) -> bool {
            self.raw == other.raw
        }
    }
    impl<N: AstNode> Hash for AstPtr<N> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.raw.hash(state);
        }
    }
    impl<N: AstNode> AstPtr<N> {
        pub fn new(node: &N) -> AstPtr<N> {
            AstPtr {
                raw: SyntaxNodePtr::new(node.syntax()),
                _ty: PhantomData,
            }
        }
        pub fn to_node(&self, root: &SyntaxNode) -> N {
            let syntax_node = self.raw.to_node(root);
            N::cast(syntax_node).unwrap()
        }
        pub fn syntax_node_ptr(&self) -> SyntaxNodePtr {
            self.raw.clone()
        }
        pub fn text_range(&self) -> TextRange {
            self.raw.text_range()
        }
        pub fn cast<U: AstNode>(self) -> Option<AstPtr<U>> {
            if !U::can_cast(self.raw.kind()) {
                return None;
            }
            Some(AstPtr {
                raw: self.raw,
                _ty: PhantomData,
            })
        }
        pub fn upcast<M: AstNode>(self) -> AstPtr<M>
        where
            N: Into<M>,
        {
            AstPtr {
                raw: self.raw,
                _ty: PhantomData,
            }
        }
        pub fn try_from_raw(raw: SyntaxNodePtr) -> Option<AstPtr<N>> {
            N::can_cast(raw.kind()).then_some(AstPtr { raw, _ty: PhantomData })
        }
    }
    impl<N: AstNode> From<AstPtr<N>> for SyntaxNodePtr {
        fn from(ptr: AstPtr<N>) -> SyntaxNodePtr {
            ptr.raw
        }
    }
    #[test]
    fn test_local_syntax_ptr() {
        use crate::{ast, AstNode, SourceFile};
        let file = SourceFile::parse("struct Foo { f: u32, }").ok().unwrap();
        let field = file.syntax().descendants().find_map(ast::RecordField::cast).unwrap();
        let ptr = SyntaxNodePtr::new(field.syntax());
        let field_syntax = ptr.to_node(file.syntax());
        assert_eq!(field.syntax(), &field_syntax);
    }
}
mod syntax_error {
    use crate::{TextRange, TextSize};
    use std::fmt;
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct SyntaxError(String, TextRange);
    impl SyntaxError {
        pub fn new(message: impl Into<String>, range: TextRange) -> Self {
            Self(message.into(), range)
        }
        pub fn new_at_offset(message: impl Into<String>, offset: TextSize) -> Self {
            Self(message.into(), TextRange::empty(offset))
        }
        pub fn range(&self) -> TextRange {
            self.1
        }
        pub fn with_range(mut self, range: TextRange) -> Self {
            self.1 = range;
            self
        }
    }
    impl fmt::Display for SyntaxError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.fmt(f)
        }
    }
}
mod syntax_node {
    use crate::{Parse, SyntaxError, SyntaxKind, TextSize};
    pub use rowan::{GreenNode, GreenToken, NodeOrToken};
    use rowan::{GreenNodeBuilder, Language};
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum RustLanguage {}
    impl Language for RustLanguage {
        type Kind = SyntaxKind;
        fn kind_from_raw(raw: rowan::SyntaxKind) -> SyntaxKind {
            SyntaxKind::from(raw.0)
        }
        fn kind_to_raw(kind: SyntaxKind) -> rowan::SyntaxKind {
            rowan::SyntaxKind(kind.into())
        }
    }
    pub type SyntaxNode = rowan::SyntaxNode<RustLanguage>;
    pub type SyntaxToken = rowan::SyntaxToken<RustLanguage>;
    pub type SyntaxElement = rowan::SyntaxElement<RustLanguage>;
    pub type SyntaxNodeChildren = rowan::SyntaxNodeChildren<RustLanguage>;
    pub type SyntaxElementChildren = rowan::SyntaxElementChildren<RustLanguage>;
    pub type PreorderWithTokens = rowan::api::PreorderWithTokens<RustLanguage>;
    #[derive(Default)]
    pub struct SyntaxTreeBuilder {
        errors: Vec<SyntaxError>,
        inner: GreenNodeBuilder<'static>,
    }
    impl SyntaxTreeBuilder {
        pub fn finish_raw(self) -> (GreenNode, Vec<SyntaxError>) {
            let green = self.inner.finish();
            (green, self.errors)
        }
        pub fn finish(self) -> Parse<SyntaxNode> {
            let (green, errors) = self.finish_raw();
            #[allow(clippy::overly_complex_bool_expr)]
            if cfg!(debug_assertions) && false {
                let node = SyntaxNode::new_root(green.clone());
                crate::validation::validate_block_structure(&node);
            }
            Parse::new(green, errors)
        }
        pub fn token(&mut self, kind: SyntaxKind, text: &str) {
            let kind = RustLanguage::kind_to_raw(kind);
            self.inner.token(kind, text);
        }
        pub fn start_node(&mut self, kind: SyntaxKind) {
            let kind = RustLanguage::kind_to_raw(kind);
            self.inner.start_node(kind);
        }
        pub fn finish_node(&mut self) {
            self.inner.finish_node();
        }
        pub fn error(&mut self, error: String, text_pos: TextSize) {
            self.errors.push(SyntaxError::new_at_offset(error, text_pos));
        }
    }
}
pub mod ted {
    use crate::{
        ast::{self, edit::IndentLevel, make, AstNode},
        SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken,
    };
    use parser::T;
    use std::{mem, ops::RangeInclusive};
    pub trait Element {
        fn syntax_element(self) -> SyntaxElement;
    }
    impl<E: Element + Clone> Element for &'_ E {
        fn syntax_element(self) -> SyntaxElement {
            self.clone().syntax_element()
        }
    }
    impl Element for SyntaxElement {
        fn syntax_element(self) -> SyntaxElement {
            self
        }
    }
    impl Element for SyntaxNode {
        fn syntax_element(self) -> SyntaxElement {
            self.into()
        }
    }
    impl Element for SyntaxToken {
        fn syntax_element(self) -> SyntaxElement {
            self.into()
        }
    }
    #[derive(Debug)]
    pub struct Position {
        repr: PositionRepr,
    }
    #[derive(Debug)]
    enum PositionRepr {
        FirstChild(SyntaxNode),
        After(SyntaxElement),
    }
    impl Position {
        pub fn after(elem: impl Element) -> Position {
            let repr = PositionRepr::After(elem.syntax_element());
            Position { repr }
        }
        pub fn before(elem: impl Element) -> Position {
            let elem = elem.syntax_element();
            let repr = match elem.prev_sibling_or_token() {
                Some(it) => PositionRepr::After(it),
                None => PositionRepr::FirstChild(elem.parent().unwrap()),
            };
            Position { repr }
        }
        pub fn first_child_of(node: &(impl Into<SyntaxNode> + Clone)) -> Position {
            let repr = PositionRepr::FirstChild(node.clone().into());
            Position { repr }
        }
        pub fn last_child_of(node: &(impl Into<SyntaxNode> + Clone)) -> Position {
            let node = node.clone().into();
            let repr = match node.last_child_or_token() {
                Some(it) => PositionRepr::After(it),
                None => PositionRepr::FirstChild(node),
            };
            Position { repr }
        }
    }
    pub fn insert(position: Position, elem: impl Element) {
        insert_all(position, vec![elem.syntax_element()]);
    }
    pub fn insert_raw(position: Position, elem: impl Element) {
        insert_all_raw(position, vec![elem.syntax_element()]);
    }
    pub fn insert_all(position: Position, mut elements: Vec<SyntaxElement>) {
        if let Some(first) = elements.first() {
            if let Some(ws) = ws_before(&position, first) {
                elements.insert(0, ws.into());
            }
        }
        if let Some(last) = elements.last() {
            if let Some(ws) = ws_after(&position, last) {
                elements.push(ws.into());
            }
        }
        insert_all_raw(position, elements);
    }
    pub fn insert_all_raw(position: Position, elements: Vec<SyntaxElement>) {
        let (parent, index) = match position.repr {
            PositionRepr::FirstChild(parent) => (parent, 0),
            PositionRepr::After(child) => (child.parent().unwrap(), child.index() + 1),
        };
        parent.splice_children(index..index, elements);
    }
    pub fn remove(elem: impl Element) {
        elem.syntax_element().detach();
    }
    pub fn remove_all(range: RangeInclusive<SyntaxElement>) {
        replace_all(range, Vec::new());
    }
    pub fn remove_all_iter(range: impl IntoIterator<Item = SyntaxElement>) {
        let mut it = range.into_iter();
        if let Some(mut first) = it.next() {
            match it.last() {
                Some(mut last) => {
                    if first.index() > last.index() {
                        mem::swap(&mut first, &mut last);
                    }
                    remove_all(first..=last);
                },
                None => remove(first),
            }
        }
    }
    pub fn replace(old: impl Element, new: impl Element) {
        replace_with_many(old, vec![new.syntax_element()]);
    }
    pub fn replace_with_many(old: impl Element, new: Vec<SyntaxElement>) {
        let old = old.syntax_element();
        replace_all(old.clone()..=old, new);
    }
    pub fn replace_all(range: RangeInclusive<SyntaxElement>, new: Vec<SyntaxElement>) {
        let start = range.start().index();
        let end = range.end().index();
        let parent = range.start().parent().unwrap();
        parent.splice_children(start..end + 1, new);
    }
    pub fn append_child(node: &(impl Into<SyntaxNode> + Clone), child: impl Element) {
        let position = Position::last_child_of(node);
        insert(position, child);
    }
    pub fn append_child_raw(node: &(impl Into<SyntaxNode> + Clone), child: impl Element) {
        let position = Position::last_child_of(node);
        insert_raw(position, child);
    }
    fn ws_before(position: &Position, new: &SyntaxElement) -> Option<SyntaxToken> {
        let prev = match &position.repr {
            PositionRepr::FirstChild(_) => return None,
            PositionRepr::After(it) => it,
        };
        if prev.kind() == T!['{'] && new.kind() == SyntaxKind::USE {
            if let Some(item_list) = prev.parent().and_then(ast::ItemList::cast) {
                let mut indent = IndentLevel::from_element(&item_list.syntax().clone().into());
                indent.0 += 1;
                return Some(make::tokens::whitespace(&format!("\n{indent}")));
            }
        }
        if prev.kind() == T!['{'] && ast::Stmt::can_cast(new.kind()) {
            if let Some(stmt_list) = prev.parent().and_then(ast::StmtList::cast) {
                let mut indent = IndentLevel::from_element(&stmt_list.syntax().clone().into());
                indent.0 += 1;
                return Some(make::tokens::whitespace(&format!("\n{indent}")));
            }
        }
        ws_between(prev, new)
    }
    fn ws_after(position: &Position, new: &SyntaxElement) -> Option<SyntaxToken> {
        let next = match &position.repr {
            PositionRepr::FirstChild(parent) => parent.first_child_or_token()?,
            PositionRepr::After(sibling) => sibling.next_sibling_or_token()?,
        };
        ws_between(new, &next)
    }
    fn ws_between(left: &SyntaxElement, right: &SyntaxElement) -> Option<SyntaxToken> {
        if left.kind() == SyntaxKind::WHITESPACE || right.kind() == SyntaxKind::WHITESPACE {
            return None;
        }
        if right.kind() == T![;] || right.kind() == T![,] {
            return None;
        }
        if left.kind() == T![<] || right.kind() == T![>] {
            return None;
        }
        if left.kind() == T![&] && right.kind() == SyntaxKind::LIFETIME {
            return None;
        }
        if right.kind() == SyntaxKind::GENERIC_ARG_LIST {
            return None;
        }
        if right.kind() == SyntaxKind::USE {
            let mut indent = IndentLevel::from_element(left);
            if left.kind() == SyntaxKind::USE {
                indent.0 = IndentLevel::from_element(right).0.max(indent.0);
            }
            return Some(make::tokens::whitespace(&format!("\n{indent}")));
        }
        Some(make::tokens::single_space())
    }
}
mod token_text {
    use rowan::GreenToken;
    use smol_str::SmolStr;
    use std::{cmp::Ordering, fmt, ops};
    pub struct TokenText<'a>(pub Repr<'a>);
    pub enum Repr<'a> {
        Borrowed(&'a str),
        Owned(GreenToken),
    }
    impl<'a> TokenText<'a> {
        pub fn borrowed(text: &'a str) -> Self {
            TokenText(Repr::Borrowed(text))
        }
        pub fn owned(green: GreenToken) -> Self {
            TokenText(Repr::Owned(green))
        }
        pub fn as_str(&self) -> &str {
            match &self.0 {
                &Repr::Borrowed(it) => it,
                Repr::Owned(green) => green.text(),
            }
        }
    }
    impl ops::Deref for TokenText<'_> {
        type Target = str;
        fn deref(&self) -> &str {
            self.as_str()
        }
    }
    impl AsRef<str> for TokenText<'_> {
        fn as_ref(&self) -> &str {
            self.as_str()
        }
    }
    impl From<TokenText<'_>> for String {
        fn from(token_text: TokenText<'_>) -> Self {
            token_text.as_str().into()
        }
    }
    impl From<TokenText<'_>> for SmolStr {
        fn from(token_text: TokenText<'_>) -> Self {
            SmolStr::new(token_text.as_str())
        }
    }
    impl PartialEq<&'_ str> for TokenText<'_> {
        fn eq(&self, other: &&str) -> bool {
            self.as_str() == *other
        }
    }
    impl PartialEq<TokenText<'_>> for &'_ str {
        fn eq(&self, other: &TokenText<'_>) -> bool {
            other == self
        }
    }
    impl PartialEq<String> for TokenText<'_> {
        fn eq(&self, other: &String) -> bool {
            self.as_str() == other.as_str()
        }
    }
    impl PartialEq<TokenText<'_>> for String {
        fn eq(&self, other: &TokenText<'_>) -> bool {
            other == self
        }
    }
    impl PartialEq for TokenText<'_> {
        fn eq(&self, other: &TokenText<'_>) -> bool {
            self.as_str() == other.as_str()
        }
    }
    impl Eq for TokenText<'_> {}
    impl Ord for TokenText<'_> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.as_str().cmp(other.as_str())
        }
    }
    impl PartialOrd for TokenText<'_> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl fmt::Display for TokenText<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Display::fmt(self.as_str(), f)
        }
    }
    impl fmt::Debug for TokenText<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Debug::fmt(self.as_str(), f)
        }
    }
}
pub mod utils {
    use crate::{ast, match_ast, AstNode, SyntaxKind};
    use itertools::Itertools;
    pub fn path_to_string_stripping_turbo_fish(path: &ast::Path) -> String {
        path.syntax()
            .children()
            .filter_map(|node| {
                match_ast! {
                    match node {
                        ast::PathSegment(it) => {
                            Some(it.name_ref()?.to_string())
                        },
                        ast::Path(it) => {
                            Some(path_to_string_stripping_turbo_fish(&it))
                        },
                        _ => None,
                    }
                }
            })
            .join("::")
    }
    pub fn is_raw_identifier(name: &str) -> bool {
        let is_keyword = SyntaxKind::from_keyword(name).is_some();
        is_keyword && !matches!(name, "self" | "crate" | "super" | "Self")
    }
    #[cfg(test)]
    mod tests {
        use super::path_to_string_stripping_turbo_fish;
        use crate::ast::make;
        #[test]
        fn turbofishes_are_stripped() {
            assert_eq!(
                "Vec",
                path_to_string_stripping_turbo_fish(&make::path_from_text("Vec::<i32>")),
            );
            assert_eq!(
                "Vec::new",
                path_to_string_stripping_turbo_fish(&make::path_from_text("Vec::<i32>::new")),
            );
            assert_eq!(
                "Vec::new",
                path_to_string_stripping_turbo_fish(&make::path_from_text("Vec::new()")),
            );
        }
    }
}
mod validation;

#[derive(Debug, PartialEq, Eq)]
pub struct Parse<T> {
    green: GreenNode,
    errors: Arc<Vec<SyntaxError>>,
    _ty: PhantomData<fn() -> T>,
}
impl<T> Parse<T> {
    fn new(green: GreenNode, errors: Vec<SyntaxError>) -> Parse<T> {
        Parse {
            green,
            errors: Arc::new(errors),
            _ty: PhantomData,
        }
    }
    pub fn syntax_node(&self) -> SyntaxNode {
        SyntaxNode::new_root(self.green.clone())
    }
    pub fn errors(&self) -> &[SyntaxError] {
        &self.errors
    }
}
impl<T: AstNode> Parse<T> {
    pub fn to_syntax(self) -> Parse<SyntaxNode> {
        Parse {
            green: self.green,
            errors: self.errors,
            _ty: PhantomData,
        }
    }
    pub fn tree(&self) -> T {
        T::cast(self.syntax_node()).unwrap()
    }
    pub fn ok(self) -> Result<T, Arc<Vec<SyntaxError>>> {
        if self.errors.is_empty() {
            Ok(self.tree())
        } else {
            Err(self.errors)
        }
    }
}
impl Parse<SyntaxNode> {
    pub fn cast<N: AstNode>(self) -> Option<Parse<N>> {
        if N::cast(self.syntax_node()).is_some() {
            Some(Parse {
                green: self.green,
                errors: self.errors,
                _ty: PhantomData,
            })
        } else {
            None
        }
    }
}
impl Parse<SourceFile> {
    pub fn debug_dump(&self) -> String {
        let mut buf = format!("{:#?}", self.tree().syntax());
        for err in self.errors.iter() {
            format_to!(buf, "error {:?}: {}\n", err.range(), err);
        }
        buf
    }
    pub fn reparse(&self, indel: &Indel) -> Parse<SourceFile> {
        self.incremental_reparse(indel)
            .unwrap_or_else(|| self.full_reparse(indel))
    }
    fn incremental_reparse(&self, indel: &Indel) -> Option<Parse<SourceFile>> {
        parsing::incremental_reparse(self.tree().syntax(), indel, self.errors.to_vec()).map(
            |(green_node, errors, _reparsed_range)| Parse {
                green: green_node,
                errors: Arc::new(errors),
                _ty: PhantomData,
            },
        )
    }
    fn full_reparse(&self, indel: &Indel) -> Parse<SourceFile> {
        let mut text = self.tree().syntax().text().to_string();
        indel.apply(&mut text);
        SourceFile::parse(&text)
    }
}
impl<T> Clone for Parse<T> {
    fn clone(&self) -> Parse<T> {
        Parse {
            green: self.green.clone(),
            errors: self.errors.clone(),
            _ty: PhantomData,
        }
    }
}
pub use crate::ast::SourceFile;
impl SourceFile {
    pub fn parse(text: &str) -> Parse<SourceFile> {
        let (green, mut errors) = parsing::parse_text(text);
        let root = SyntaxNode::new_root(green.clone());
        errors.extend(validation::validate(&root));
        assert_eq!(root.kind(), SyntaxKind::SOURCE_FILE);
        Parse {
            green,
            errors: Arc::new(errors),
            _ty: PhantomData,
        }
    }
}
#[macro_export]
macro_rules! match_ast {
    (match $node:ident { $($tt:tt)* }) => { $crate::match_ast!(match ($node) { $($tt)* }) };
    (match ($node:expr) {
        $( $( $path:ident )::+ ($it:pat) => $res:expr, )*
        _ => $catch_all:expr $(,)?
    }) => {{
        $( if let Some($it) = $($path::)+cast($node.clone()) { $res } else )*
        { $catch_all }
    }};
}
#[test]
fn api_walkthrough() {
    use ast::{HasModuleItem, HasName};
    let source_code = "
        fn foo() {
            1 + 1
        }
    ";
    let parse = SourceFile::parse(source_code);
    assert!(parse.errors().is_empty());
    let file: SourceFile = parse.tree();
    let mut func = None;
    for item in file.items() {
        match item {
            ast::Item::Fn(f) => func = Some(f),
            _ => unreachable!(),
        }
    }
    let func: ast::Fn = func.unwrap();
    let name: Option<ast::Name> = func.name();
    let name = name.unwrap();
    assert_eq!(name.text(), "foo");
    let body: ast::BlockExpr = func.body().unwrap();
    let stmt_list: ast::StmtList = body.stmt_list().unwrap();
    let expr: ast::Expr = stmt_list.tail_expr().unwrap();
    let bin_expr: &ast::BinExpr = match &expr {
        ast::Expr::BinExpr(e) => e,
        _ => unreachable!(),
    };
    let expr_syntax: &SyntaxNode = expr.syntax();
    assert!(expr_syntax == bin_expr.syntax());
    let _expr: ast::Expr = match ast::Expr::cast(expr_syntax.clone()) {
        Some(e) => e,
        None => unreachable!(),
    };
    assert_eq!(expr_syntax.kind(), SyntaxKind::BIN_EXPR);
    assert_eq!(expr_syntax.text_range(), TextRange::new(32.into(), 37.into()));
    let text: SyntaxText = expr_syntax.text();
    assert_eq!(text.to_string(), "1 + 1");
    assert_eq!(expr_syntax.parent().as_ref(), Some(stmt_list.syntax()));
    assert_eq!(
        stmt_list.syntax().first_child_or_token().map(|it| it.kind()),
        Some(T!['{'])
    );
    assert_eq!(
        expr_syntax.next_sibling_or_token().map(|it| it.kind()),
        Some(SyntaxKind::WHITESPACE)
    );
    let f = expr_syntax.ancestors().find_map(ast::Fn::cast);
    assert_eq!(f, Some(func));
    assert!(expr_syntax
        .siblings_with_tokens(Direction::Next)
        .any(|it| it.kind() == T!['}']));
    assert_eq!(expr_syntax.descendants_with_tokens().count(), 8,);
    let mut buf = String::new();
    let mut indent = 0;
    for event in expr_syntax.preorder_with_tokens() {
        match event {
            WalkEvent::Enter(node) => {
                let text = match &node {
                    NodeOrToken::Node(it) => it.text().to_string(),
                    NodeOrToken::Token(it) => it.text().to_string(),
                };
                format_to!(buf, "{:indent$}{:?} {:?}\n", " ", text, node.kind(), indent = indent);
                indent += 2;
            },
            WalkEvent::Leave(_) => indent -= 2,
        }
    }
    assert_eq!(indent, 0);
    assert_eq!(
        buf.trim(),
        r#"
"1 + 1" BIN_EXPR
  "1" LITERAL
    "1" INT_NUMBER
  " " WHITESPACE
  "+" PLUS
  " " WHITESPACE
  "1" LITERAL
    "1" INT_NUMBER
"#
        .trim()
    );
    let exprs_cast: Vec<String> = file
        .syntax()
        .descendants()
        .filter_map(ast::Expr::cast)
        .map(|expr| expr.syntax().text().to_string())
        .collect();
    let mut exprs_visit = Vec::new();
    for node in file.syntax().descendants() {
        match_ast! {
            match node {
                ast::Expr(it) => {
                    let res = it.syntax().text().to_string();
                    exprs_visit.push(res);
                },
                _ => (),
            }
        }
    }
    assert_eq!(exprs_cast, exprs_visit);
}
#[cfg(test)]
mod tests;
