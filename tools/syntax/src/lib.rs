#![forbid(
    // missing_debug_implementations,
    unconditional_recursion,
    future_incompatible,
    // missing_docs,
)]
#![warn(unused_lifetimes)]

pub use crate::{
    core::{api, green, Direction, Text},
    ptr::{AstPtr, NodePtr},
    token_text::TokenText,
};
pub use smol_str::SmolStr;
use std::{fmt, marker::PhantomData};
use text_edit::Indel;
pub use text_size::{TextRange, TextSize};
use triomphe::Arc;

//pub use parser::{SyntaxKind, T};
pub mod tmp;
pub use tmp::*;
pub mod lexer;
pub mod text;

pub mod algo;
pub mod ast;
pub mod core;
#[doc(hidden)]
pub mod fuzz {
    use crate::{validation, SourceFile, TextRange};
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
    use crate::ast;
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
        use crate::{
            core::{green, NodeOrToken},
            parsing::build_tree,
            SyntaxErr,
            SyntaxKind::*,
            TextRange, TextSize, T,
        };
        use parser::Reparser;
        use text_edit::Indel;

        pub fn incremental_reparse(
            x: &crate::Node,
            edit: &Indel,
            es: Vec<SyntaxErr>,
        ) -> Option<(green::Node, Vec<SyntaxErr>, TextRange)> {
            if let Some((green, new_errors, old_range)) = reparse_token(x, edit) {
                return Some((green, merge_errors(es, new_errors, old_range, edit), old_range));
            }
            if let Some((green, new_errors, old_range)) = reparse_block(x, edit) {
                return Some((green, merge_errors(es, new_errors, old_range, edit), old_range));
            }
            None
        }
        fn reparse_token(root: &crate::Node, edit: &Indel) -> Option<(green::Node, Vec<SyntaxErr>, TextRange)> {
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
                    let new_token = green::Token::new(green::Kind(prev_token_kind.into()), &new_text);
                    let range = TextRange::up_to(TextSize::of(&new_text));
                    Some((
                        prev_token.replace_with(new_token),
                        new_err.into_iter().map(|x| SyntaxErr::new(x, range)).collect(),
                        prev_token.text_range(),
                    ))
                },
                _ => None,
            }
        }
        fn reparse_block(root: &crate::Node, edit: &Indel) -> Option<(green::Node, Vec<SyntaxErr>, TextRange)> {
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
        fn get_text_after_edit(element: crate::Elem, edit: &Indel) -> String {
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
        fn find_reparsable_node(node: &crate::Node, range: TextRange) -> Option<(crate::Node, Reparser)> {
            let node = node.covering_element(range);
            node.ancestors().find_map(|node| {
                let first_child = node.first_child_or_token().map(|x| x.kind());
                let parent = node.parent().map(|x| x.kind());
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
            old_errors: Vec<SyntaxErr>,
            new_errors: Vec<SyntaxErr>,
            range_before_reparse: TextRange,
            edit: &Indel,
        ) -> Vec<SyntaxErr> {
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
            use crate::{assert_eq_text, Parse, SourceFile};
            use test_utils::extract_range;
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
    use crate::{core::green, SyntaxErr, SyntaxTreeBuilder, TextRange};
    pub fn parse_text(text: &str) -> (green::Node, Vec<SyntaxErr>) {
        let lexed = parser::LexedStr::new(text);
        let parser_input = lexed.to_input();
        let parser_output = parser::TopEntryPoint::SourceFile.parse(&parser_input);
        let (node, errors, _eof) = build_tree(lexed, parser_output);
        (node, errors)
    }
    pub fn build_tree(
        lexed: parser::LexedStr<'_>,
        parser_output: parser::Output,
    ) -> (green::Node, Vec<SyntaxErr>, bool) {
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
            errors.push(SyntaxErr::new(err, text_range))
        }
        (node, errors, is_eof)
    }
}
mod ptr {
    use crate::{api, ast, core, Lang, TextRange};
    use std::{
        hash::{Hash, Hasher},
        marker::PhantomData,
    };
    pub type NodePtr = core::NodePtr<Lang>;
    #[derive(Debug)]
    pub struct AstPtr<N: ast::Node> {
        raw: NodePtr,
        _ty: PhantomData<fn() -> N>,
    }
    impl<N: ast::Node> Clone for AstPtr<N> {
        fn clone(&self) -> AstPtr<N> {
            AstPtr {
                raw: self.raw.clone(),
                _ty: PhantomData,
            }
        }
    }
    impl<N: ast::Node> Eq for AstPtr<N> {}
    impl<N: ast::Node> PartialEq for AstPtr<N> {
        fn eq(&self, other: &AstPtr<N>) -> bool {
            self.raw == other.raw
        }
    }
    impl<N: ast::Node> Hash for AstPtr<N> {
        fn hash<H: Hasher>(&self, x: &mut H) {
            self.raw.hash(x);
        }
    }
    impl<N: ast::Node> AstPtr<N> {
        pub fn new(node: &N) -> AstPtr<N> {
            AstPtr {
                raw: NodePtr::new(node.syntax()),
                _ty: PhantomData,
            }
        }
        pub fn to_node(&self, root: &crate::Node) -> N {
            let syntax_node = self.raw.to_node(root);
            N::cast(syntax_node).unwrap()
        }
        pub fn syntax_node_ptr(&self) -> NodePtr {
            self.raw.clone()
        }
        pub fn text_range(&self) -> TextRange {
            self.raw.text_range()
        }
        pub fn cast<U: ast::Node>(self) -> Option<AstPtr<U>> {
            if !U::can_cast(self.raw.kind()) {
                return None;
            }
            Some(AstPtr {
                raw: self.raw,
                _ty: PhantomData,
            })
        }
        pub fn upcast<M: ast::Node>(self) -> AstPtr<M>
        where
            N: Into<M>,
        {
            AstPtr {
                raw: self.raw,
                _ty: PhantomData,
            }
        }
        pub fn try_from_raw(raw: NodePtr) -> Option<AstPtr<N>> {
            N::can_cast(raw.kind()).then_some(AstPtr { raw, _ty: PhantomData })
        }
    }
    impl<N: ast::Node> From<AstPtr<N>> for NodePtr {
        fn from(ptr: AstPtr<N>) -> NodePtr {
            ptr.raw
        }
    }
    #[test]
    fn test_local_syntax_ptr() {
        use crate::{ast, SourceFile};
        let file = SourceFile::parse("struct Foo { f: u32, }").ok().unwrap();
        let field = file.syntax().descendants().find_map(ast::RecordField::cast).unwrap();
        let ptr = NodePtr::new(field.syntax());
        let field_syntax = ptr.to_node(file.syntax());
        assert_eq!(field.syntax(), &field_syntax);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SyntaxErr(String, TextRange);
impl SyntaxErr {
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
impl fmt::Display for SyntaxErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lang {}
impl api::Lang for Lang {
    type Kind = SyntaxKind;
    fn kind_from_raw(x: green::Kind) -> SyntaxKind {
        SyntaxKind::from(x.0)
    }
    fn kind_to_raw(x: SyntaxKind) -> green::Kind {
        green::Kind(x.into())
    }
}
pub type Node = api::Node<Lang>;
pub type Token = api::Token<Lang>;
pub type Elem = api::Elem<Lang>;
pub type NodeChildren = api::NodeChildren<Lang>;
pub type ElemChildren = api::ElemChildren<Lang>;
pub type PreorderWithToks = api::PreorderWithToks<Lang>;

#[derive(Default)]
pub struct SyntaxTreeBuilder {
    errs: Vec<SyntaxErr>,
    inner: green::NodeBuilder<'static>,
}
impl SyntaxTreeBuilder {
    pub fn finish_raw(self) -> (green::Node, Vec<SyntaxErr>) {
        let green = self.inner.finish();
        (green, self.errs)
    }
    pub fn finish(self) -> Parse<Node> {
        let (green, errs) = self.finish_raw();
        #[allow(clippy::overly_complex_bool_expr)]
        if cfg!(debug_assertions) && false {
            let node = api::Node::new_root(green.clone());
            crate::validation::validate_block_structure(&node);
        }
        Parse::new(green, errs)
    }
    pub fn token(&mut self, x: SyntaxKind, text: &str) {
        let y = Lang::kind_to_raw(x);
        self.inner.token(y, text);
    }
    pub fn start_node(&mut self, x: SyntaxKind) {
        let y = Lang::kind_to_raw(x);
        self.inner.start_node(y);
    }
    pub fn finish_node(&mut self) {
        self.inner.finish_node();
    }
    pub fn error(&mut self, err: String, pos: TextSize) {
        self.errs.push(SyntaxErr::new_at_offset(err, pos));
    }
}

pub mod ted {
    use crate::{
        ast::{self, edit::IndentLevel, make},
        SyntaxKind, T, *,
    };
    use std::{mem, ops::RangeInclusive};
    pub trait Elem {
        fn syntax_element(self) -> crate::Elem;
    }
    impl<E: Elem + Clone> Elem for &'_ E {
        fn syntax_element(self) -> crate::Elem {
            self.clone().syntax_element()
        }
    }
    impl Elem for crate::Elem {
        fn syntax_element(self) -> crate::Elem {
            self
        }
    }
    impl Elem for crate::Node {
        fn syntax_element(self) -> crate::Elem {
            self.into()
        }
    }
    impl Elem for crate::Token {
        fn syntax_element(self) -> crate::Elem {
            self.into()
        }
    }
    #[derive(Debug)]
    pub struct Pos {
        repr: PosRepr,
    }
    #[derive(Debug)]
    enum PosRepr {
        FirstChild(Node),
        After(Elem),
    }
    impl Pos {
        pub fn after(x: impl Elem) -> Pos {
            let repr = PosRepr::After(x.syntax_element());
            Pos { repr }
        }
        pub fn before(x: impl Elem) -> Pos {
            let x = x.syntax_element();
            let repr = match x.prev_sibling_or_token() {
                Some(x) => PosRepr::After(x),
                None => PosRepr::FirstChild(x.parent().unwrap()),
            };
            Pos { repr }
        }
        pub fn first_child_of(x: &(impl Into<crate::Node> + Clone)) -> Pos {
            let repr = PosRepr::FirstChild(x.clone().into());
            Pos { repr }
        }
        pub fn last_child_of(x: &(impl Into<crate::Node> + Clone)) -> Pos {
            let x = x.clone().into();
            let repr = match x.last_child_or_token() {
                Some(x) => PosRepr::After(x),
                None => PosRepr::FirstChild(x),
            };
            Pos { repr }
        }
    }
    pub fn insert(pos: Pos, x: impl Elem) {
        insert_all(pos, vec![x.syntax_element()]);
    }
    pub fn insert_raw(pos: Pos, x: impl Elem) {
        insert_all_raw(pos, vec![x.syntax_element()]);
    }
    pub fn insert_all(pos: Pos, mut xs: Vec<crate::Elem>) {
        if let Some(x) = xs.first() {
            if let Some(x) = ws_before(&pos, x) {
                xs.insert(0, x.into());
            }
        }
        if let Some(x) = xs.last() {
            if let Some(x) = ws_after(&pos, x) {
                xs.push(x.into());
            }
        }
        insert_all_raw(pos, xs);
    }
    pub fn insert_all_raw(pos: Pos, xs: Vec<crate::Elem>) {
        let (parent, index) = match pos.repr {
            PosRepr::FirstChild(parent) => (parent, 0),
            PosRepr::After(child) => (child.parent().unwrap(), child.index() + 1),
        };
        parent.splice_children(index..index, xs);
    }
    pub fn remove(x: impl Elem) {
        x.syntax_element().detach();
    }
    pub fn remove_all(range: RangeInclusive<crate::Elem>) {
        replace_all(range, Vec::new());
    }
    pub fn remove_all_iter(range: impl IntoIterator<Item = crate::Elem>) {
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
    pub fn replace(old: impl Elem, new: impl Elem) {
        replace_with_many(old, vec![new.syntax_element()]);
    }
    pub fn replace_with_many(old: impl Elem, new: Vec<crate::Elem>) {
        let old = old.syntax_element();
        replace_all(old.clone()..=old, new);
    }
    pub fn replace_all(range: RangeInclusive<crate::Elem>, new: Vec<crate::Elem>) {
        let start = range.start().index();
        let end = range.end().index();
        let parent = range.start().parent().unwrap();
        parent.splice_children(start..end + 1, new);
    }
    pub fn append_child(x: &(impl Into<crate::Node> + Clone), child: impl Elem) {
        let pos = Pos::last_child_of(x);
        insert(pos, child);
    }
    pub fn append_child_raw(x: &(impl Into<crate::Node> + Clone), child: impl Elem) {
        let pos = Pos::last_child_of(x);
        insert_raw(pos, child);
    }
    fn ws_before(pos: &Pos, new: &crate::Elem) -> Option<crate::Token> {
        let prev = match &pos.repr {
            PosRepr::FirstChild(_) => return None,
            PosRepr::After(x) => x,
        };
        if prev.kind() == T!['{'] && new.kind() == SyntaxKind::USE {
            if let Some(item_list) = prev.parent().and_then(ast::ItemList::cast) {
                let mut y = IndentLevel::from_element(&item_list.syntax().clone().into());
                y.0 += 1;
                return Some(make::tokens::whitespace(&format!("\n{indent}")));
            }
        }
        if prev.kind() == T!['{'] && ast::Stmt::can_cast(new.kind()) {
            if let Some(stmt_list) = prev.parent().and_then(ast::StmtList::cast) {
                let mut y = IndentLevel::from_element(&stmt_list.syntax().clone().into());
                y.0 += 1;
                return Some(make::tokens::whitespace(&format!("\n{indent}")));
            }
        }
        ws_between(prev, new)
    }
    fn ws_after(pos: &Pos, new: &crate::Elem) -> Option<crate::Token> {
        let next = match &pos.repr {
            PosRepr::FirstChild(x) => x.first_child_or_token()?,
            PosRepr::After(x) => x.next_sibling_or_token()?,
        };
        ws_between(new, &next)
    }
    fn ws_between(left: &crate::Elem, right: &crate::Elem) -> Option<crate::Token> {
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
            let mut y = IndentLevel::from_element(left);
            if left.kind() == SyntaxKind::USE {
                y.0 = IndentLevel::from_element(right).0.max(y.0);
            }
            return Some(make::tokens::whitespace(&format!("\n{indent}")));
        }
        Some(make::tokens::single_space())
    }
}
mod token_text {
    use crate::core::green;
    use smol_str::SmolStr;
    use std::{cmp::Ordering, fmt, ops};
    pub struct TokenText<'a>(pub Repr<'a>);
    pub enum Repr<'a> {
        Borrowed(&'a str),
        Owned(green::Token),
    }
    impl<'a> TokenText<'a> {
        pub fn borrowed(text: &'a str) -> Self {
            TokenText(Repr::Borrowed(text))
        }
        pub fn owned(green: green::Token) -> Self {
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
    use crate::{ast, match_ast, SyntaxKind};
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
    green: green::Node,
    errors: Arc<Vec<SyntaxErr>>,
    _ty: PhantomData<fn() -> T>,
}
impl<T> Parse<T> {
    fn new(green: green::Node, errors: Vec<SyntaxErr>) -> Parse<T> {
        Parse {
            green,
            errors: Arc::new(errors),
            _ty: PhantomData,
        }
    }
    pub fn syntax_node(&self) -> Node {
        api::Node::new_root(self.green.clone())
    }
    pub fn errors(&self) -> &[SyntaxErr] {
        &self.errors
    }
}
impl<T: ast::Node> Parse<T> {
    pub fn to_syntax(self) -> Parse<Node> {
        Parse {
            green: self.green,
            errors: self.errors,
            _ty: PhantomData,
        }
    }
    pub fn tree(&self) -> T {
        T::cast(self.syntax_node()).unwrap()
    }
    pub fn ok(self) -> Result<T, Arc<Vec<SyntaxErr>>> {
        if self.errors.is_empty() {
            Ok(self.tree())
        } else {
            Err(self.errors)
        }
    }
}
impl Parse<Node> {
    pub fn cast<N: ast::Node>(self) -> Option<Parse<N>> {
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
        let (green, mut errs) = parsing::parse_text(text);
        let root = api::Node::new_root(green.clone());
        errs.extend(validation::validate(&root));
        assert_eq!(root.kind(), SyntaxKind::SOURCE_FILE);
        Parse {
            green,
            errors: Arc::new(errs),
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
    use crate::core::{NodeOrToken, WalkEvent};
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
    let expr_syntax: &api::Node = expr.syntax();
    assert!(expr_syntax == bin_expr.syntax());
    let _expr: ast::Expr = match ast::Expr::cast(expr_syntax.clone()) {
        Some(e) => e,
        None => unreachable!(),
    };
    assert_eq!(expr_syntax.kind(), SyntaxKind::BIN_EXPR);
    assert_eq!(expr_syntax.text_range(), TextRange::new(32.into(), 37.into()));
    let text: Text = expr_syntax.text();
    assert_eq!(text.to_string(), "1 + 1");
    assert_eq!(expr_syntax.parent().as_ref(), Some(stmt_list.syntax()));
    assert_eq!(
        stmt_list.syntax().first_child_or_token().map(|x| x.kind()),
        Some(T!['{'])
    );
    assert_eq!(
        expr_syntax.next_sibling_or_token().map(|x| x.kind()),
        Some(SyntaxKind::WHITESPACE)
    );
    let f = expr_syntax.ancestors().find_map(ast::Fn::cast);
    assert_eq!(f, Some(func));
    assert!(expr_syntax
        .siblings_with_tokens(Direction::Next)
        .any(|x| x.kind() == T!['}']));
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
