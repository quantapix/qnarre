use crate::{
    syntax::{
        self, ast,
        core::{Direction, NodeOrToken},
        text::EditBuilder,
        Elem, TextRange, TextSize,
    },
    SyntaxKind,
};
use indexmap::IndexMap;
use itertools::Itertools;
use rustc_hash::FxHashMap;
use std::hash::BuildHasherDefault;

pub fn ancestors_at_offset(x: &syntax::Node, off: TextSize) -> impl Iterator<Item = syntax::Node> {
    x.token_at_offset(off)
        .map(|x| x.parent_ancestors())
        .kmerge_by(|x1, x2| x1.text_range().len() < x2.text_range().len())
}
pub fn find_node_at_offset<N: ast::Node>(x: &syntax::Node, off: TextSize) -> Option<N> {
    ancestors_at_offset(x, off).find_map(N::cast)
}
pub fn find_node_at_range<N: ast::Node>(x: &syntax::Node, range: TextRange) -> Option<N> {
    x.covering_element(range).ancestors().find_map(N::cast)
}
pub fn skip_trivia_token(mut x: super::Token, dir: Direction) -> Option<super::Token> {
    while x.kind().is_trivia() {
        x = match dir {
            Direction::Next => x.next_token()?,
            Direction::Prev => x.prev_token()?,
        }
    }
    Some(x)
}
pub fn skip_whitespace_token(mut x: super::Token, dir: Direction) -> Option<super::Token> {
    while x.kind() == SyntaxKind::WHITESPACE {
        x = match dir {
            Direction::Next => x.next_token()?,
            Direction::Prev => x.prev_token()?,
        }
    }
    Some(x)
}
pub fn non_trivia_sibling(x: Elem, dir: Direction) -> Option<Elem> {
    return match x {
        NodeOrToken::Node(x) => x.siblings_with_tokens(x).skip(1).find(not_trivia),
        NodeOrToken::Token(x) => x.siblings_with_tokens(x).skip(1).find(not_trivia),
    };
    fn not_trivia(x: &Elem) -> bool {
        match x {
            NodeOrToken::Node(_) => true,
            NodeOrToken::Token(x) => !x.kind().is_trivia(),
        }
    }
}
pub fn least_common_ancestor(u: &syntax::Node, v: &syntax::Node) -> Option<syntax::Node> {
    if u == v {
        return Some(u.clone());
    }
    let u_depth = u.ancestors().count();
    let v_depth = v.ancestors().count();
    let keep = u_depth.min(v_depth);
    let u_candidates = u.ancestors().skip(u_depth - keep);
    let v_candidates = v.ancestors().skip(v_depth - keep);
    let (res, _) = u_candidates.zip(v_candidates).find(|(x, y)| x == y)?;
    Some(res)
}
pub fn neighbor<T: ast::Node>(x: &T, dir: Direction) -> Option<T> {
    x.syntax().siblings(dir).skip(1).find_map(T::cast)
}
pub fn has_errors(x: &syntax::Node) -> bool {
    x.children().any(|x| x.kind() == SyntaxKind::ERROR)
}
type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<rustc_hash::FxHasher>>;
#[derive(Debug, Hash, PartialEq, Eq)]
enum TreeDiffInsertPos {
    After(Elem),
    AsFirstChild(Elem),
}
#[derive(Debug)]
pub struct TreeDiff {
    replacements: FxHashMap<Elem, Elem>,
    deletions: Vec<Elem>,
    insertions: FxIndexMap<TreeDiffInsertPos, Vec<Elem>>,
}
impl TreeDiff {
    pub fn into_text_edit(&self, builder: &mut EditBuilder) {
        // let _p = profile::span("into_text_edit");
        for (anchor, to) in &self.insertions {
            let offset = match anchor {
                TreeDiffInsertPos::After(x) => x.text_range().end(),
                TreeDiffInsertPos::AsFirstChild(x) => x.text_range().start(),
            };
            to.iter().for_each(|x| builder.insert(offset, x.to_string()));
        }
        for (from, to) in &self.replacements {
            builder.replace(from.text_range(), to.to_string());
        }
        for text_range in self.deletions.iter().map(Elem::text_range) {
            builder.delete(text_range);
        }
    }
    pub fn is_empty(&self) -> bool {
        self.replacements.is_empty() && self.deletions.is_empty() && self.insertions.is_empty()
    }
}
pub fn diff(from: &syntax::Node, to: &syntax::Node) -> TreeDiff {
    // let _p = profile::span("diff");
    let mut diff = TreeDiff {
        replacements: FxHashMap::default(),
        insertions: FxIndexMap::default(),
        deletions: Vec::new(),
    };
    let (from, to) = (from.clone().into(), to.clone().into());
    if !syntax_element_eq(&from, &to) {
        go(&mut diff, from, to);
    }
    return diff;
    fn syntax_element_eq(lhs: &Elem, rhs: &Elem) -> bool {
        lhs.kind() == rhs.kind()
            && lhs.text_range().len() == rhs.text_range().len()
            && match (&lhs, &rhs) {
                (NodeOrToken::Node(lhs), NodeOrToken::Node(rhs)) => lhs == rhs || lhs.text() == rhs.text(),
                (NodeOrToken::Token(lhs), NodeOrToken::Token(rhs)) => lhs.text() == rhs.text(),
                _ => false,
            }
    }
    fn go(diff: &mut TreeDiff, lhs: Elem, rhs: Elem) {
        let (lhs, rhs) = match lhs.as_node().zip(rhs.as_node()) {
            Some((lhs, rhs)) => (lhs, rhs),
            _ => {
                // cov_mark::hit!(diff_node_token_replace);
                diff.replacements.insert(lhs, rhs);
                return;
            },
        };
        let mut look_ahead_scratch = Vec::default();
        let mut rhs_children = rhs.children_with_tokens();
        let mut lhs_children = lhs.children_with_tokens();
        let mut last_lhs = None;
        loop {
            let lhs_child = lhs_children.next();
            match (lhs_child.clone(), rhs_children.next()) {
                (None, None) => break,
                (None, Some(element)) => {
                    let insert_pos = match last_lhs.clone() {
                        Some(prev) => {
                            // cov_mark::hit!(diff_insert);
                            TreeDiffInsertPos::After(prev)
                        },
                        None => {
                            // cov_mark::hit!(diff_insert_as_first_child);
                            TreeDiffInsertPos::AsFirstChild(lhs.clone().into())
                        },
                    };
                    diff.insertions.entry(insert_pos).or_insert_with(Vec::new).push(element);
                },
                (Some(element), None) => {
                    // cov_mark::hit!(diff_delete);
                    diff.deletions.push(element);
                },
                (Some(ref lhs_ele), Some(ref rhs_ele)) if syntax_element_eq(lhs_ele, rhs_ele) => {},
                (Some(lhs_ele), Some(rhs_ele)) => {
                    look_ahead_scratch.push(rhs_ele.clone());
                    let mut rhs_children_clone = rhs_children.clone();
                    let mut insert = false;
                    for rhs_child in &mut rhs_children_clone {
                        if syntax_element_eq(&lhs_ele, &rhs_child) {
                            // cov_mark::hit!(diff_insertions);
                            insert = true;
                            break;
                        }
                        look_ahead_scratch.push(rhs_child);
                    }
                    let drain = look_ahead_scratch.drain(..);
                    if insert {
                        let insert_pos = if let Some(prev) = last_lhs.clone().filter(|_| insert) {
                            TreeDiffInsertPos::After(prev)
                        } else {
                            // cov_mark::hit!(insert_first_child);
                            TreeDiffInsertPos::AsFirstChild(lhs.clone().into())
                        };
                        diff.insertions.entry(insert_pos).or_insert_with(Vec::new).extend(drain);
                        rhs_children = rhs_children_clone;
                    } else {
                        go(diff, lhs_ele, rhs_ele);
                    }
                },
            }
            last_lhs = lhs_child.or(last_lhs);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::syntax::{text::Edit, Elem, SyntaxKind};
    use expect_test::{expect, Expect};
    use itertools::Itertools;
    #[test]
    fn replace_node_token() {
        // cov_mark::check!(diff_node_token_replace);
        check_diff(
            r#"use node;"#,
            r#"ident"#,
            expect![[r#"
                insertions:

                replacements:
                Line 0: Token(USE_KW@0..3 "use") -> ident
                deletions:
                Line 1: " "
                Line 1: node
                Line 1: ;
            "#]],
        );
    }
    #[test]
    fn replace_parent() {
        // cov_mark::check!(diff_insert_as_first_child);
        check_diff(
            r#""#,
            r#"use foo::bar;"#,
            expect![[r#"
                insertions:
                Line 0: AsFirstChild(Node(SOURCE_FILE@0..0))
                -> use foo::bar;
                replacements:

                deletions:

            "#]],
        );
    }
    #[test]
    fn insert_last() {
        // cov_mark::check!(diff_insert);
        check_diff(
            r#"
use foo;
use bar;"#,
            r#"
use foo;
use bar;
use baz;"#,
            expect![[r#"
                insertions:
                Line 2: After(Node(USE@10..18))
                -> "\n"
                -> use baz;
                replacements:

                deletions:

            "#]],
        );
    }
    #[test]
    fn insert_middle() {
        check_diff(
            r#"
use foo;
use baz;"#,
            r#"
use foo;
use bar;
use baz;"#,
            expect![[r#"
                insertions:
                Line 2: After(Token(WHITESPACE@9..10 "\n"))
                -> use bar;
                -> "\n"
                replacements:

                deletions:

            "#]],
        )
    }
    #[test]
    fn insert_first() {
        check_diff(
            r#"
use bar;
use baz;"#,
            r#"
use foo;
use bar;
use baz;"#,
            expect![[r#"
                insertions:
                Line 0: After(Token(WHITESPACE@0..1 "\n"))
                -> use foo;
                -> "\n"
                replacements:

                deletions:

            "#]],
        )
    }
    #[test]
    fn first_child_insertion() {
        // cov_mark::check!(insert_first_child);
        check_diff(
            r#"fn main() {
        stdi
    }"#,
            r#"use foo::bar;
    fn main() {
        stdi
    }"#,
            expect![[r#"
                insertions:
                Line 0: AsFirstChild(Node(SOURCE_FILE@0..30))
                -> use foo::bar;
                -> "\n\n    "
                replacements:

                deletions:

            "#]],
        );
    }
    #[test]
    fn delete_last() {
        // cov_mark::check!(diff_delete);
        check_diff(
            r#"use foo;
            use bar;"#,
            r#"use foo;"#,
            expect![[r#"
                insertions:

                replacements:

                deletions:
                Line 1: "\n            "
                Line 2: use bar;
            "#]],
        );
    }
    #[test]
    fn delete_middle() {
        // cov_mark::check!(diff_insertions);
        check_diff(
            r#"
use expect_test::{expect, Expect};
use text_edit::TextEdit;
use crate::ast::Node;
"#,
            r#"
use expect_test::{expect, Expect};
use crate::ast::Node;
"#,
            expect![[r#"
                insertions:
                Line 1: After(Node(USE@1..35))
                -> "\n\n"
                -> use crate::ast::Node;
                replacements:

                deletions:
                Line 2: use text_edit::TextEdit;
                Line 3: "\n\n"
                Line 4: use crate::ast::Node;
                Line 5: "\n"
            "#]],
        )
    }
    #[test]
    fn delete_first() {
        check_diff(
            r#"
use text_edit::TextEdit;
use crate::ast::Node;
"#,
            r#"
use crate::ast::Node;
"#,
            expect![[r#"
                insertions:

                replacements:
                Line 2: Token(IDENT@5..14 "text_edit") -> crate
                Line 2: Token(IDENT@16..24 "TextEdit") -> ast::Node
                Line 2: Token(WHITESPACE@25..27 "\n\n") -> "\n"
                deletions:
                Line 3: use crate::ast::Node;
                Line 4: "\n"
            "#]],
        )
    }
    #[test]
    fn merge_use() {
        check_diff(
            r#"
use std::{
    fmt,
    hash::BuildHasherDefault,
    ops::{self, RangeInclusive},
};
"#,
            r#"
use std::fmt;
use std::hash::BuildHasherDefault;
use std::ops::{self, RangeInclusive};
"#,
            expect![[r#"
                insertions:
                Line 2: After(Node(PATH_SEGMENT@5..8))
                -> ::
                -> fmt
                Line 6: After(Token(WHITESPACE@86..87 "\n"))
                -> use std::hash::BuildHasherDefault;
                -> "\n"
                -> use std::ops::{self, RangeInclusive};
                -> "\n"
                replacements:
                Line 2: Token(IDENT@5..8 "std") -> std
                deletions:
                Line 2: ::
                Line 2: {
                    fmt,
                    hash::BuildHasherDefault,
                    ops::{self, RangeInclusive},
                }
            "#]],
        )
    }
    #[test]
    fn early_return_assist() {
        check_diff(
            r#"
fn main() {
    if let Ok(x) = Err(92) {
        foo(x);
    }
}
            "#,
            r#"
fn main() {
    let x = match Err(92) {
        Ok(x) => x,
        _ => return,
    };
    foo(x);
}
            "#,
            expect![[r#"
                insertions:
                Line 3: After(Node(BLOCK_EXPR@40..63))
                -> " "
                -> match Err(92) {
                        Ok(x) => x,
                        _ => return,
                    }
                -> ;
                Line 3: After(Node(IF_EXPR@17..63))
                -> "\n    "
                -> foo(x);
                replacements:
                Line 3: Token(IF_KW@17..19 "if") -> let
                Line 3: Token(LET_KW@20..23 "let") -> x
                Line 3: Node(BLOCK_EXPR@40..63) -> =
                deletions:
                Line 3: " "
                Line 3: Ok(x)
                Line 3: " "
                Line 3: =
                Line 3: " "
                Line 3: Err(92)
            "#]],
        )
    }
    fn check_diff(from: &str, to: &str, expected_diff: Expect) {
        let from_node = crate::syntax::SourceFile::parse(from).tree().syntax().clone();
        let to_node = crate::syntax::SourceFile::parse(to).tree().syntax().clone();
        let diff = super::diff(&from_node, &to_node);
        let line_number = |x: &Elem| from[..x.text_range().start().into()].lines().count();
        let fmt_syntax = |x: &Elem| match x.kind() {
            SyntaxKind::WHITESPACE => format!("{:?}", x.to_string()),
            _ => format!("{x}"),
        };
        let insertions = diff
            .insertions
            .iter()
            .format_with("\n", |(k, v), f| -> Result<(), std::fmt::Error> {
                f(&format!(
                    "Line {}: {:?}\n-> {}",
                    line_number(match k {
                        super::TreeDiffInsertPos::After(syn) => syn,
                        super::TreeDiffInsertPos::AsFirstChild(syn) => syn,
                    }),
                    k,
                    v.iter().format_with("\n-> ", |v, f| f(&fmt_syntax(v)))
                ))
            });
        let replacements = diff
            .replacements
            .iter()
            .sorted_by_key(|(syntax, _)| syntax.text_range().start())
            .format_with("\n", |(k, v), f| {
                f(&format!("Line {}: {k:?} -> {}", line_number(k), fmt_syntax(v)))
            });
        let deletions = diff
            .deletions
            .iter()
            .format_with("\n", |v, f| f(&format!("Line {}: {}", line_number(v), &fmt_syntax(v))));
        let actual =
            format!("insertions:\n\n{insertions}\n\nreplacements:\n\n{replacements}\n\ndeletions:\n\n{deletions}\n");
        expected_diff.assert_eq(&actual);
        let mut from = from.to_owned();
        let mut text_edit = Edit::builder();
        diff.into_text_edit(&mut text_edit);
        text_edit.finish().apply(&mut from);
        assert_eq!(&*from, to, "diff did not turn `from` to `to`");
    }
}
