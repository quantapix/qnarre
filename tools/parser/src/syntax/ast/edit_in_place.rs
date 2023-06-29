use super::HasName;
use crate::{
    syntax::{
        self,
        algo::{self, neighbor},
        ast::{self, edit::IndentLevel, make, HasGenericParams},
        core::{api, Direction},
        ted::{self, Pos},
    },
    SyntaxKind::{self, ATTR, COMMENT, WHITESPACE},
    T,
};
use std::iter::{empty, successors};

pub trait GenericParamsOwnerEdit: ast::HasGenericParams {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList;
    fn get_or_create_where_clause(&self) -> ast::WhereClause;
}
impl GenericParamsOwnerEdit for ast::Fn {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = if let Some(name) = self.name() {
                    Pos::after(name.syntax)
                } else if let Some(fn_token) = self.fn_token() {
                    Pos::after(fn_token)
                } else if let Some(param_list) = self.param_list() {
                    Pos::before(param_list.syntax)
                } else {
                    Pos::last_child_of(self.syntax())
                };
                create_generic_param_list(position)
            },
        }
    }
    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(ty) = self.ret_type() {
                Pos::after(ty.syntax())
            } else if let Some(param_list) = self.param_list() {
                Pos::after(param_list.syntax())
            } else {
                Pos::last_child_of(self.syntax())
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}
impl GenericParamsOwnerEdit for ast::Impl {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = match self.impl_token() {
                    Some(imp_token) => Pos::after(imp_token),
                    None => Pos::last_child_of(self.syntax()),
                };
                create_generic_param_list(position)
            },
        }
    }
    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let position = match self.assoc_item_list() {
                Some(items) => Pos::before(items.syntax()),
                None => Pos::last_child_of(self.syntax()),
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}
impl GenericParamsOwnerEdit for ast::Trait {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = if let Some(name) = self.name() {
                    Pos::after(name.syntax)
                } else if let Some(trait_token) = self.trait_token() {
                    Pos::after(trait_token)
                } else {
                    Pos::last_child_of(self.syntax())
                };
                create_generic_param_list(position)
            },
        }
    }
    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let position = match self.assoc_item_list() {
                Some(items) => Pos::before(items.syntax()),
                None => Pos::last_child_of(self.syntax()),
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}
impl GenericParamsOwnerEdit for ast::Struct {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = if let Some(name) = self.name() {
                    Pos::after(name.syntax)
                } else if let Some(struct_token) = self.struct_token() {
                    Pos::after(struct_token)
                } else {
                    Pos::last_child_of(self.syntax())
                };
                create_generic_param_list(position)
            },
        }
    }
    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let tfl = self.field_list().and_then(|fl| match fl {
                ast::FieldList::RecordFieldList(_) => None,
                ast::FieldList::TupleFieldList(it) => Some(it),
            });
            let position = if let Some(tfl) = tfl {
                Pos::after(tfl.syntax())
            } else if let Some(gpl) = self.generic_param_list() {
                Pos::after(gpl.syntax())
            } else if let Some(name) = self.name() {
                Pos::after(name.syntax())
            } else {
                Pos::last_child_of(self.syntax())
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}
impl GenericParamsOwnerEdit for ast::Enum {
    fn get_or_create_generic_param_list(&self) -> ast::GenericParamList {
        match self.generic_param_list() {
            Some(it) => it,
            None => {
                let position = if let Some(name) = self.name() {
                    Pos::after(name.syntax)
                } else if let Some(enum_token) = self.enum_token() {
                    Pos::after(enum_token)
                } else {
                    Pos::last_child_of(self.syntax())
                };
                create_generic_param_list(position)
            },
        }
    }
    fn get_or_create_where_clause(&self) -> ast::WhereClause {
        if self.where_clause().is_none() {
            let position = if let Some(gpl) = self.generic_param_list() {
                Pos::after(gpl.syntax())
            } else if let Some(name) = self.name() {
                Pos::after(name.syntax())
            } else {
                Pos::last_child_of(self.syntax())
            };
            create_where_clause(position);
        }
        self.where_clause().unwrap()
    }
}
fn create_where_clause(position: Pos) {
    let where_clause = make::where_clause(empty()).clone_for_update();
    ted::insert(position, where_clause.syntax());
}
fn create_generic_param_list(position: Pos) -> ast::GenericParamList {
    let gpl = make::generic_param_list(empty()).clone_for_update();
    ted::insert_raw(position, gpl.syntax());
    gpl
}
pub trait AttrsOwnerEdit: ast::HasAttrs {
    fn remove_attrs_and_docs(&self) {
        remove_attrs_and_docs(self.syntax());
        fn remove_attrs_and_docs(node: &syntax::Node) {
            let mut remove_next_ws = false;
            for child in node.children_with_tokens() {
                match child.kind() {
                    ATTR | COMMENT => {
                        remove_next_ws = true;
                        child.detach();
                        continue;
                    },
                    WHITESPACE if remove_next_ws => {
                        child.detach();
                    },
                    _ => (),
                }
                remove_next_ws = false;
            }
        }
    }
}
impl<T: ast::HasAttrs> AttrsOwnerEdit for T {}
impl ast::GenericParamList {
    pub fn add_generic_param(&self, generic_param: ast::GenericParam) {
        match self.generic_params().last() {
            Some(last_param) => {
                let position = Pos::after(last_param.syntax());
                let elements = vec![
                    make::token(T![,]).into(),
                    make::tokens::single_space().into(),
                    generic_param.syntax().clone().into(),
                ];
                ted::insert_all(position, elements);
            },
            None => {
                let after_l_angle = Pos::after(self.l_angle_token().unwrap());
                ted::insert(after_l_angle, generic_param.syntax());
            },
        }
    }
    pub fn remove_generic_param(&self, generic_param: ast::GenericParam) {
        if let Some(previous) = generic_param.syntax().prev_sibling() {
            if let Some(next_token) = previous.next_sibling_or_token() {
                ted::remove_all(next_token..=generic_param.syntax().clone().into());
            }
        } else if let Some(next) = generic_param.syntax().next_sibling() {
            if let Some(next_token) = next.prev_sibling_or_token() {
                ted::remove_all(generic_param.syntax().clone().into()..=next_token);
            }
        } else {
            ted::remove(generic_param.syntax());
        }
    }
    pub fn to_generic_args(&self) -> ast::GenericArgList {
        let args = self.generic_params().filter_map(|param| match param {
            ast::GenericParam::LifetimeParam(it) => {
                Some(ast::GenericArg::LifetimeArg(make::lifetime_arg(it.lifetime()?)))
            },
            ast::GenericParam::TypeParam(it) => {
                Some(ast::GenericArg::TypeArg(make::type_arg(make::ext::ty_name(it.name()?))))
            },
            ast::GenericParam::ConstParam(it) => {
                Some(ast::GenericArg::TypeArg(make::type_arg(make::ext::ty_name(it.name()?))))
            },
        });
        make::generic_arg_list(args)
    }
}
impl ast::WhereClause {
    pub fn add_predicate(&self, pred: ast::WherePred) {
        if let Some(pred) = self.predicates().last() {
            if !pred
                .syntax()
                .siblings_with_tokens(Direction::Next)
                .any(|x| x.kind() == T![,])
            {
                ted::append_child_raw(self.syntax(), make::token(T![,]));
            }
        }
        ted::append_child(self.syntax(), pred.syntax());
    }
}
impl ast::TypeParam {
    pub fn remove_default(&self) {
        if let Some((eq, last)) = self
            .syntax()
            .children_with_tokens()
            .find(|x| x.kind() == T![=])
            .zip(self.syntax().last_child_or_token())
        {
            ted::remove_all(eq..=last);
            if let Some(x) = self.syntax().last_token().filter(|x| x.kind() == WHITESPACE) {
                x.detach();
            }
        }
    }
}
impl ast::ConstParam {
    pub fn remove_default(&self) {
        if let Some((eq, last)) = self
            .syntax()
            .children_with_tokens()
            .find(|x| x.kind() == T![=])
            .zip(self.syntax().last_child_or_token())
        {
            ted::remove_all(eq..=last);
            if let Some(x) = self.syntax().last_token().filter(|x| x.kind() == WHITESPACE) {
                x.detach();
            }
        }
    }
}
pub trait Removable: ast::Node {
    fn remove(&self);
}
impl Removable for ast::TypeBoundList {
    fn remove(&self) {
        match self
            .syntax()
            .siblings_with_tokens(Direction::Prev)
            .find(|x| x.kind() == T![:])
        {
            Some(x) => ted::remove_all(x..=self.syntax().clone().into()),
            None => ted::remove(self.syntax()),
        }
    }
}
impl ast::PathSegment {
    pub fn get_or_create_generic_arg_list(&self) -> ast::GenericArgList {
        if self.generic_arg_list().is_none() {
            let arg_list = make::generic_arg_list(empty()).clone_for_update();
            ted::append_child(self.syntax(), arg_list.syntax());
        }
        self.generic_arg_list().unwrap()
    }
}
impl Removable for ast::UseTree {
    fn remove(&self) {
        for dir in [Direction::Next, Direction::Prev] {
            if let Some(next_use_tree) = neighbor(self, dir) {
                let separators = self
                    .syntax()
                    .siblings_with_tokens(dir)
                    .skip(1)
                    .take_while(|x| x.as_node() != Some(next_use_tree.syntax()));
                ted::remove_all_iter(separators);
                break;
            }
        }
        ted::remove(self.syntax());
    }
}
impl ast::UseTree {
    pub fn get_or_create_use_tree_list(&self) -> ast::UseTreeList {
        match self.use_tree_list() {
            Some(it) => it,
            None => {
                let position = Pos::last_child_of(self.syntax());
                let use_tree_list = make::use_tree_list(empty()).clone_for_update();
                let mut elements = Vec::with_capacity(2);
                if self.coloncolon_token().is_none() {
                    elements.push(make::token(T![::]).into());
                }
                elements.push(use_tree_list.syntax().clone().into());
                ted::insert_all_raw(position, elements);
                use_tree_list
            },
        }
    }
    pub fn split_prefix(&self, prefix: &ast::Path) {
        debug_assert_eq!(self.path(), Some(prefix.top_path()));
        let path = self.path().unwrap();
        if &path == prefix && self.use_tree_list().is_none() {
            if self.star_token().is_some() {
                self.coloncolon_token().map(ted::remove);
                ted::remove(prefix.syntax());
            } else {
                let self_suffix = make::path_unqualified(make::path_segment_self()).clone_for_update();
                ted::replace(path.syntax(), self_suffix.syntax());
            }
        } else if split_path_prefix(prefix).is_none() {
            return;
        }
        let subtree = self.clone_subtree().clone_for_update();
        ted::remove_all_iter(self.syntax().children_with_tokens());
        ted::insert(Pos::first_child_of(self.syntax()), prefix.syntax());
        self.get_or_create_use_tree_list().add_use_tree(subtree);
        fn split_path_prefix(prefix: &ast::Path) -> Option<()> {
            let parent = prefix.parent_path()?;
            let segment = parent.segment()?;
            if algo::has_errors(segment.syntax()) {
                return None;
            }
            for p in successors(parent.parent_path(), |x| x.parent_path()) {
                p.segment()?;
            }
            prefix.parent_path().and_then(|p| p.coloncolon_token()).map(ted::remove);
            ted::remove(prefix.syntax());
            Some(())
        }
    }
}
impl ast::UseTreeList {
    pub fn add_use_tree(&self, use_tree: ast::UseTree) {
        let (position, elements) = match self.use_trees().last() {
            Some(last_tree) => (
                Pos::after(last_tree.syntax()),
                vec![
                    make::token(T![,]).into(),
                    make::tokens::single_space().into(),
                    use_tree.syntax.into(),
                ],
            ),
            None => {
                let position = match self.l_curly_token() {
                    Some(l_curly) => Pos::after(l_curly),
                    None => Pos::last_child_of(self.syntax()),
                };
                (position, vec![use_tree.syntax.into()])
            },
        };
        ted::insert_all_raw(position, elements);
    }
}
impl Removable for ast::Use {
    fn remove(&self) {
        let next_ws = self
            .syntax()
            .next_sibling_or_token()
            .and_then(|x| x.into_token())
            .and_then(ast::Whitespace::cast);
        if let Some(next_ws) = next_ws {
            let ws_text = next_ws.syntax().text();
            if let Some(rest) = ws_text.strip_prefix('\n') {
                if rest.is_empty() {
                    ted::remove(next_ws.syntax());
                } else {
                    ted::replace(next_ws.syntax(), make::tokens::whitespace(rest));
                }
            }
        }
        ted::remove(self.syntax());
    }
}
impl ast::Impl {
    pub fn get_or_create_assoc_item_list(&self) -> ast::AssocItemList {
        if self.assoc_item_list().is_none() {
            let assoc_item_list = make::assoc_item_list().clone_for_update();
            ted::append_child(self.syntax(), assoc_item_list.syntax());
        }
        self.assoc_item_list().unwrap()
    }
}
impl ast::AssocItemList {
    pub fn add_item(&self, item: ast::AssocItem) {
        let (indent, position, whitespace) = match self.assoc_items().last() {
            Some(last_item) => (
                IndentLevel::from_node(last_item.syntax()),
                Pos::after(last_item.syntax()),
                "\n\n",
            ),
            None => match self.l_curly_token() {
                Some(l_curly) => {
                    normalize_ws_between_braces(self.syntax());
                    (IndentLevel::from_token(&l_curly) + 1, Pos::after(&l_curly), "\n")
                },
                None => (IndentLevel::single(), Pos::last_child_of(self.syntax()), "\n"),
            },
        };
        let elements: Vec<api::Elem<_>> = vec![
            make::tokens::whitespace(&format!("{whitespace}{indent}")).into(),
            item.syntax().clone().into(),
        ];
        ted::insert_all(position, elements);
    }
}
impl ast::Fn {
    pub fn get_or_create_body(&self) -> ast::BlockExpr {
        if self.body().is_none() {
            let body = make::ext::empty_block_expr().clone_for_update();
            match self.semicolon_token() {
                Some(semi) => {
                    ted::replace(semi, body.syntax());
                    ted::insert(Pos::before(body.syntax), make::tokens::single_space());
                },
                None => ted::append_child(self.syntax(), body.syntax()),
            }
        }
        self.body().unwrap()
    }
}
impl Removable for ast::MatchArm {
    fn remove(&self) {
        if let Some(sibling) = self.syntax().prev_sibling_or_token() {
            if sibling.kind() == SyntaxKind::WHITESPACE {
                ted::remove(sibling);
            }
        }
        if let Some(sibling) = self.syntax().next_sibling_or_token() {
            if sibling.kind() == T![,] {
                ted::remove(sibling);
            }
        }
        ted::remove(self.syntax());
    }
}
impl ast::MatchArmList {
    pub fn add_arm(&self, arm: ast::MatchArm) {
        normalize_ws_between_braces(self.syntax());
        let mut elements = Vec::new();
        let position = match self.arms().last() {
            Some(last_arm) => {
                if needs_comma(&last_arm) {
                    ted::append_child(last_arm.syntax(), make::token(SyntaxKind::COMMA));
                }
                Pos::after(last_arm.syntax().clone())
            },
            None => match self.l_curly_token() {
                Some(it) => Pos::after(it),
                None => Pos::last_child_of(self.syntax()),
            },
        };
        let indent = IndentLevel::from_node(self.syntax()) + 1;
        elements.push(make::tokens::whitespace(&format!("\n{indent}")).into());
        elements.push(arm.syntax().clone().into());
        if needs_comma(&arm) {
            ted::append_child(arm.syntax(), make::token(SyntaxKind::COMMA));
        }
        ted::insert_all(position, elements);
        fn needs_comma(arm: &ast::MatchArm) -> bool {
            arm.expr().map_or(false, |e| !e.is_block_like()) && arm.comma_token().is_none()
        }
    }
}
impl ast::RecordExprFieldList {
    pub fn add_field(&self, field: ast::RecordExprField) {
        let is_multiline = self.syntax().text().contains_char('\n');
        let whitespace = if is_multiline {
            let indent = IndentLevel::from_node(self.syntax()) + 1;
            make::tokens::whitespace(&format!("\n{indent}"))
        } else {
            make::tokens::single_space()
        };
        if is_multiline {
            normalize_ws_between_braces(self.syntax());
        }
        let position = match self.fields().last() {
            Some(last_field) => {
                let comma = get_or_insert_comma_after(last_field.syntax());
                Pos::after(comma)
            },
            None => match self.l_curly_token() {
                Some(it) => Pos::after(it),
                None => Pos::last_child_of(self.syntax()),
            },
        };
        ted::insert_all(position, vec![whitespace.into(), field.syntax().clone().into()]);
        if is_multiline {
            ted::insert(Pos::after(field.syntax()), ast::make::token(T![,]));
        }
    }
}
impl ast::RecordExprField {
    pub fn replace_expr(&self, expr: ast::Expr) {
        if self.name_ref().is_some() {
            match self.expr() {
                Some(prev) => ted::replace(prev.syntax(), expr.syntax()),
                None => ted::append_child(self.syntax(), expr.syntax()),
            }
            return;
        }
        if let Some(ast::Expr::PathExpr(path_expr)) = self.expr() {
            if let Some(path) = path_expr.path() {
                if let Some(name_ref) = path.as_single_name_ref() {
                    path_expr.syntax().detach();
                    let children = vec![
                        name_ref.syntax().clone().into(),
                        ast::make::token(T![:]).into(),
                        ast::make::tokens::single_space().into(),
                        expr.syntax().clone().into(),
                    ];
                    ted::insert_all_raw(Pos::last_child_of(self.syntax()), children);
                }
            }
        }
    }
}
impl ast::RecordPatFieldList {
    pub fn add_field(&self, field: ast::RecordPatField) {
        let is_multiline = self.syntax().text().contains_char('\n');
        let whitespace = if is_multiline {
            let indent = IndentLevel::from_node(self.syntax()) + 1;
            make::tokens::whitespace(&format!("\n{indent}"))
        } else {
            make::tokens::single_space()
        };
        if is_multiline {
            normalize_ws_between_braces(self.syntax());
        }
        let position = match self.fields().last() {
            Some(last_field) => {
                let syntax = last_field.syntax();
                let comma = get_or_insert_comma_after(syntax);
                Pos::after(comma)
            },
            None => match self.l_curly_token() {
                Some(it) => Pos::after(it),
                None => Pos::last_child_of(self.syntax()),
            },
        };
        ted::insert_all(position, vec![whitespace.into(), field.syntax().clone().into()]);
        if is_multiline {
            ted::insert(Pos::after(field.syntax()), ast::make::token(T![,]));
        }
    }
}
fn get_or_insert_comma_after(syntax: &syntax::Node) -> syntax::Token {
    match syntax
        .siblings_with_tokens(Direction::Next)
        .filter_map(|x| x.into_token())
        .find(|x| x.kind() == T![,])
    {
        Some(x) => x,
        None => {
            let comma = ast::make::token(T![,]);
            ted::insert(Pos::after(syntax), &comma);
            comma
        },
    }
}
impl ast::VariantList {
    pub fn add_variant(&self, variant: ast::Variant) {
        let (indent, position) = match self.variants().last() {
            Some(last_item) => (
                IndentLevel::from_node(last_item.syntax()),
                Pos::after(get_or_insert_comma_after(last_item.syntax())),
            ),
            None => match self.l_curly_token() {
                Some(l_curly) => {
                    normalize_ws_between_braces(self.syntax());
                    (IndentLevel::from_token(&l_curly) + 1, Pos::after(&l_curly))
                },
                None => (IndentLevel::single(), Pos::last_child_of(self.syntax())),
            },
        };
        let elements: Vec<api::Elem<_>> = vec![
            make::tokens::whitespace(&format!("{}{indent}", "\n")).into(),
            variant.syntax().clone().into(),
            ast::make::token(T![,]).into(),
        ];
        ted::insert_all(position, elements);
    }
}
fn normalize_ws_between_braces(node: &syntax::Node) -> Option<()> {
    let l = node
        .children_with_tokens()
        .filter_map(|x| x.into_token())
        .find(|x| x.kind() == T!['{'])?;
    let r = node
        .children_with_tokens()
        .filter_map(|x| x.into_token())
        .find(|x| x.kind() == T!['}'])?;
    let indent = IndentLevel::from_node(node);
    match l.next_sibling_or_token() {
        Some(x) if x.kind() == SyntaxKind::WHITESPACE => {
            if x.next_sibling_or_token()?.into_token()? == r {
                ted::replace(x, make::tokens::whitespace(&format!("\n{indent}")));
            }
        },
        Some(x) if x.kind() == T!['}'] => {
            ted::insert(Pos::after(l), make::tokens::whitespace(&format!("\n{indent}")));
        },
        _ => (),
    }
    Some(())
}
pub trait Indent: ast::Node + Clone + Sized {
    fn indent_level(&self) -> IndentLevel {
        IndentLevel::from_node(self.syntax())
    }
    fn indent(&self, by: IndentLevel) {
        by.increase_indent(self.syntax());
    }
    fn dedent(&self, by: IndentLevel) {
        by.decrease_indent(self.syntax());
    }
    fn reindent_to(&self, target_level: IndentLevel) {
        let current_level = IndentLevel::from_node(self.syntax());
        self.dedent(current_level);
        self.indent(target_level);
    }
}
impl<N: ast::Node + Clone> Indent for N {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_eq_text, syntax::SourceFile};
    use std::fmt;
    fn trim_indent(mut text: &str) -> String {
        if text.starts_with('\n') {
            text = &text[1..];
        }
        let indent = text
            .lines()
            .filter(|x| !x.trim().is_empty())
            .map(|x| x.len() - x.trim_start().len())
            .min()
            .unwrap_or(0);
        text.split_inclusive('\n')
            .map(|line| {
                if line.len() <= indent {
                    line.trim_start_matches(' ')
                } else {
                    &line[indent..]
                }
            })
            .collect()
    }
    fn ast_mut_from_text<N: ast::Node>(text: &str) -> N {
        let parse = SourceFile::parse(text);
        parse
            .tree()
            .syntax()
            .descendants()
            .find_map(N::cast)
            .unwrap()
            .clone_for_update()
    }
    #[test]
    fn test_create_generic_param_list() {
        fn check_create_gpl<N: GenericParamsOwnerEdit + fmt::Display>(before: &str, after: &str) {
            let gpl_owner = ast_mut_from_text::<N>(before);
            gpl_owner.get_or_create_generic_param_list();
            assert_eq!(gpl_owner.to_string(), after);
        }
        check_create_gpl::<ast::Fn>("fn foo", "fn foo<>");
        check_create_gpl::<ast::Fn>("fn foo() {}", "fn foo<>() {}");
        check_create_gpl::<ast::Impl>("impl", "impl<>");
        check_create_gpl::<ast::Impl>("impl Struct {}", "impl<> Struct {}");
        check_create_gpl::<ast::Impl>("impl Trait for Struct {}", "impl<> Trait for Struct {}");
        check_create_gpl::<ast::Trait>("trait Trait<>", "trait Trait<>");
        check_create_gpl::<ast::Trait>("trait Trait<> {}", "trait Trait<> {}");
        check_create_gpl::<ast::Struct>("struct A", "struct A<>");
        check_create_gpl::<ast::Struct>("struct A;", "struct A<>;");
        check_create_gpl::<ast::Struct>("struct A();", "struct A<>();");
        check_create_gpl::<ast::Struct>("struct A {}", "struct A<> {}");
        check_create_gpl::<ast::Enum>("enum E", "enum E<>");
        check_create_gpl::<ast::Enum>("enum E {", "enum E<> {");
    }
    #[test]
    fn test_increase_indent() {
        let arm_list = ast_mut_from_text::<ast::Fn>(
            "fn foo() {
    ;
    ;
}",
        );
        arm_list.indent(IndentLevel(2));
        assert_eq!(
            arm_list.to_string(),
            "fn foo() {
            ;
            ;
        }",
        );
    }
    #[test]
    fn add_variant_to_empty_enum() {
        let variant = make::variant(make::name("Bar"), None).clone_for_update();
        check_add_variant(
            r#"
enum Foo {}
"#,
            r#"
enum Foo {
    Bar,
}
"#,
            variant,
        );
    }
    #[test]
    fn add_variant_to_non_empty_enum() {
        let variant = make::variant(make::name("Baz"), None).clone_for_update();
        check_add_variant(
            r#"
enum Foo {
    Bar,
}
"#,
            r#"
enum Foo {
    Bar,
    Baz,
}
"#,
            variant,
        );
    }
    #[test]
    fn add_variant_with_tuple_field_list() {
        let variant = make::variant(
            make::name("Baz"),
            Some(ast::FieldList::TupleFieldList(make::tuple_field_list(std::iter::once(
                make::tuple_field(None, make::ty("bool")),
            )))),
        )
        .clone_for_update();
        check_add_variant(
            r#"
enum Foo {
    Bar,
}
"#,
            r#"
enum Foo {
    Bar,
    Baz(bool),
}
"#,
            variant,
        );
    }
    #[test]
    fn add_variant_with_record_field_list() {
        let variant = make::variant(
            make::name("Baz"),
            Some(ast::FieldList::RecordFieldList(make::record_field_list(
                std::iter::once(make::record_field(None, make::name("x"), make::ty("bool"))),
            ))),
        )
        .clone_for_update();
        check_add_variant(
            r#"
enum Foo {
    Bar,
}
"#,
            r#"
enum Foo {
    Bar,
    Baz { x: bool },
}
"#,
            variant,
        );
    }
    fn check_add_variant(before: &str, expected: &str, variant: ast::Variant) {
        let enum_ = ast_mut_from_text::<ast::Enum>(before);
        enum_.variant_list().map(|x| x.add_variant(variant));
        let after = enum_.to_string();
        assert_eq_text!(&trim_indent(expected.trim()), &trim_indent(after.trim()));
    }
}