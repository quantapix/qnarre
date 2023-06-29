use crate::{
    syntax::{
        self,
        ast::{self, support},
        core::{green, NodeOrToken},
        SmolStr, TokenText,
    },
    SyntaxKind, T,
};
use itertools::Itertools;
use std::{borrow::Cow, fmt, iter::successors};

impl ast::Lifetime {
    pub fn text(&self) -> TokenText<'_> {
        text_of_first_token(self.syntax())
    }
}
impl ast::Name {
    pub fn text(&self) -> TokenText<'_> {
        text_of_first_token(self.syntax())
    }
}
impl ast::NameRef {
    pub fn text(&self) -> TokenText<'_> {
        text_of_first_token(self.syntax())
    }
    pub fn as_tuple_field(&self) -> Option<usize> {
        self.text().parse().ok()
    }
    pub fn token_kind(&self) -> SyntaxKind {
        self.syntax().first_token().map_or(SyntaxKind::ERROR, |x| x.kind())
    }
}
fn text_of_first_token(x: &syntax::Node) -> TokenText<'_> {
    fn first_token(x: &green::NodeData) -> &green::TokData {
        x.children().next().and_then(NodeOrToken::into_token).unwrap()
    }
    match x.green() {
        Cow::Borrowed(x) => TokenText::borrowed(first_token(x).text()),
        Cow::Owned(x) => TokenText::owned(first_token(&x).to_owned()),
    }
}
impl ast::HasModuleItem for ast::StmtList {}
impl ast::BlockExpr {
    pub fn statements(&self) -> impl Iterator<Item = ast::Stmt> {
        self.stmt_list().into_iter().flat_map(|x| x.statements())
    }
    pub fn tail_expr(&self) -> Option<ast::Expr> {
        self.stmt_list()?.tail_expr()
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Macro {
    MacroRules(ast::MacroRules),
    MacroDef(ast::MacroDef),
}
impl From<ast::MacroRules> for Macro {
    fn from(x: ast::MacroRules) -> Self {
        Macro::MacroRules(x)
    }
}
impl From<ast::MacroDef> for Macro {
    fn from(x: ast::MacroDef) -> Self {
        Macro::MacroDef(x)
    }
}
impl ast::Node for Macro {
    fn can_cast(x: SyntaxKind) -> bool {
        matches!(x, SyntaxKind::MACRO_RULES | SyntaxKind::MACRO_DEF)
    }
    fn cast(syntax: syntax::Node) -> Option<Self> {
        let y = match syntax.kind() {
            SyntaxKind::MACRO_RULES => Macro::MacroRules(ast::MacroRules { syntax }),
            SyntaxKind::MACRO_DEF => Macro::MacroDef(ast::MacroDef { syntax }),
            _ => return None,
        };
        Some(y)
    }
    fn syntax(&self) -> &syntax::Node {
        match self {
            Macro::MacroRules(x) => x.syntax(),
            Macro::MacroDef(x) => x.syntax(),
        }
    }
}
impl ast::HasName for Macro {
    fn name(&self) -> Option<ast::Name> {
        match self {
            Macro::MacroRules(x) => x.name(),
            Macro::MacroDef(x) => x.name(),
        }
    }
}
impl ast::HasAttrs for Macro {}
impl From<ast::AssocItem> for ast::Item {
    fn from(x: ast::AssocItem) -> Self {
        match x {
            ast::AssocItem::Const(x) => ast::Item::Const(x),
            ast::AssocItem::Fn(x) => ast::Item::Fn(x),
            ast::AssocItem::MacroCall(x) => ast::Item::MacroCall(x),
            ast::AssocItem::TypeAlias(x) => ast::Item::TypeAlias(x),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AttrKind {
    Inner,
    Outer,
}
impl AttrKind {
    pub fn is_inner(&self) -> bool {
        matches!(self, Self::Inner)
    }
    pub fn is_outer(&self) -> bool {
        matches!(self, Self::Outer)
    }
}
impl ast::Attr {
    pub fn as_simple_atom(&self) -> Option<SmolStr> {
        let x = self.meta()?;
        if x.eq_token().is_some() || x.token_tree().is_some() {
            return None;
        }
        self.simple_name()
    }
    pub fn as_simple_call(&self) -> Option<(SmolStr, ast::TokenTree)> {
        let x = self.meta()?.token_tree()?;
        Some((self.simple_name()?, x))
    }
    pub fn simple_name(&self) -> Option<SmolStr> {
        let x = self.meta()?.path()?;
        match (x.segment(), x.qualifier()) {
            (Some(x), None) => Some(x.syntax().first_token()?.text().into()),
            _ => None,
        }
    }
    pub fn kind(&self) -> AttrKind {
        match self.excl_token() {
            Some(_) => AttrKind::Inner,
            None => AttrKind::Outer,
        }
    }
    pub fn path(&self) -> Option<ast::Path> {
        self.meta()?.path()
    }
    pub fn expr(&self) -> Option<ast::Expr> {
        self.meta()?.expr()
    }
    pub fn token_tree(&self) -> Option<ast::TokenTree> {
        self.meta()?.token_tree()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegmentKind {
    Name(ast::NameRef),
    Type {
        type_ref: Option<ast::Type>,
        trait_ref: Option<ast::PathType>,
    },
    SelfTypeKw,
    SelfKw,
    SuperKw,
    CrateKw,
}
impl ast::PathSegment {
    pub fn parent_path(&self) -> ast::Path {
        self.syntax()
            .parent()
            .and_then(ast::Path::cast)
            .expect("segments are always nested in paths")
    }
    pub fn crate_token(&self) -> Option<syntax::Token> {
        self.name_ref().and_then(|x| x.crate_token())
    }
    pub fn self_token(&self) -> Option<syntax::Token> {
        self.name_ref().and_then(|x| x.self_token())
    }
    pub fn self_type_token(&self) -> Option<syntax::Token> {
        self.name_ref().and_then(|x| x.Self_token())
    }
    pub fn super_token(&self) -> Option<syntax::Token> {
        self.name_ref().and_then(|x| x.super_token())
    }
    pub fn kind(&self) -> Option<PathSegmentKind> {
        let y = if let Some(x) = self.name_ref() {
            match x.token_kind() {
                T![Self] => PathSegmentKind::SelfTypeKw,
                T![self] => PathSegmentKind::SelfKw,
                T![super] => PathSegmentKind::SuperKw,
                T![crate] => PathSegmentKind::CrateKw,
                _ => PathSegmentKind::Name(x),
            }
        } else {
            match self.syntax().first_child_or_token()?.kind() {
                T![<] => {
                    let mut xs = self.syntax().children().filter(|x| ast::Type::can_cast(x.kind()));
                    let type_ref = xs.next().and_then(ast::Type::cast);
                    let trait_ref = xs.next().and_then(ast::PathType::cast);
                    PathSegmentKind::Type { type_ref, trait_ref }
                },
                _ => return None,
            }
        };
        Some(y)
    }
}
impl ast::Path {
    pub fn parent_path(&self) -> Option<ast::Path> {
        self.syntax().parent().and_then(ast::Path::cast)
    }
    pub fn as_single_segment(&self) -> Option<ast::PathSegment> {
        match self.qualifier() {
            Some(_) => None,
            None => self.segment(),
        }
    }
    pub fn as_single_name_ref(&self) -> Option<ast::NameRef> {
        match self.qualifier() {
            Some(_) => None,
            None => self.segment()?.name_ref(),
        }
    }
    pub fn first_qualifier_or_self(&self) -> ast::Path {
        successors(Some(self.clone()), ast::Path::qualifier).last().unwrap()
    }
    pub fn first_segment(&self) -> Option<ast::PathSegment> {
        self.first_qualifier_or_self().segment()
    }
    pub fn segments(&self) -> impl Iterator<Item = ast::PathSegment> + Clone {
        successors(self.first_segment(), |p| {
            p.parent_path().parent_path().and_then(|p| p.segment())
        })
    }
    pub fn qualifiers(&self) -> impl Iterator<Item = ast::Path> + Clone {
        successors(self.qualifier(), |p| p.qualifier())
    }
    pub fn top_path(&self) -> ast::Path {
        let mut y = self.clone();
        while let Some(x) = y.parent_path() {
            y = x;
        }
        y
    }
}
impl ast::Use {
    pub fn is_simple_glob(&self) -> bool {
        self.use_tree()
            .map_or(false, |x| x.use_tree_list().is_none() && x.star_token().is_some())
    }
}
impl ast::UseTree {
    pub fn is_simple_path(&self) -> bool {
        self.use_tree_list().is_none() && self.star_token().is_none()
    }
}
impl ast::UseTreeList {
    pub fn parent_use_tree(&self) -> ast::UseTree {
        self.syntax()
            .parent()
            .and_then(ast::UseTree::cast)
            .expect("UseTreeLists are always nested in UseTrees")
    }
    pub fn has_inner_comment(&self) -> bool {
        self.syntax()
            .children_with_tokens()
            .filter_map(|x| x.into_token())
            .find_map(ast::Comment::cast)
            .is_some()
    }
}
impl ast::Impl {
    pub fn self_ty(&self) -> Option<ast::Type> {
        match self.target() {
            (Some(t), None) | (_, Some(t)) => Some(t),
            _ => None,
        }
    }
    pub fn trait_(&self) -> Option<ast::Type> {
        match self.target() {
            (Some(t), Some(_)) => Some(t),
            _ => None,
        }
    }
    fn target(&self) -> (Option<ast::Type>, Option<ast::Type>) {
        let mut xs = support::children(self.syntax());
        let first = xs.next();
        let second = xs.next();
        (first, second)
    }
    pub fn for_trait_name_ref(x: &ast::NameRef) -> Option<ast::Impl> {
        let y = x.syntax().ancestors().find_map(ast::Impl::cast)?;
        if y.trait_()?.syntax().text_range().start() == x.syntax().text_range().start() {
            Some(y)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructKind {
    Record(ast::RecordFieldList),
    Tuple(ast::TupleFieldList),
    Unit,
}
impl StructKind {
    fn from_node<N: ast::Node>(x: &N) -> StructKind {
        if let Some(x) = support::child::<ast::RecordFieldList>(x.syntax()) {
            StructKind::Record(x)
        } else if let Some(x) = support::child::<ast::TupleFieldList>(x.syntax()) {
            StructKind::Tuple(x)
        } else {
            StructKind::Unit
        }
    }
}
impl ast::Struct {
    pub fn kind(&self) -> StructKind {
        StructKind::from_node(self)
    }
}
impl ast::RecordExprField {
    pub fn for_field_name(x: &ast::NameRef) -> Option<ast::RecordExprField> {
        let y = Self::for_name_ref(x)?;
        if y.field_name().as_ref() == Some(x) {
            Some(y)
        } else {
            None
        }
    }
    pub fn for_name_ref(x: &ast::NameRef) -> Option<ast::RecordExprField> {
        let y = x.syntax();
        y.parent()
            .and_then(ast::RecordExprField::cast)
            .or_else(|| y.ancestors().nth(4).and_then(ast::RecordExprField::cast))
    }
    pub fn field_name(&self) -> Option<ast::NameRef> {
        if let Some(x) = self.name_ref() {
            return Some(x);
        }
        if let ast::Expr::PathExpr(x) = self.expr()? {
            let path = x.path()?;
            let seg = path.segment()?;
            let y = seg.name_ref()?;
            if path.qualifier().is_none() {
                return Some(y);
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub enum NameLike {
    NameRef(ast::NameRef),
    Name(ast::Name),
    Lifetime(ast::Lifetime),
}
impl NameLike {
    pub fn as_name_ref(&self) -> Option<&ast::NameRef> {
        match self {
            NameLike::NameRef(x) => Some(x),
            _ => None,
        }
    }
    pub fn as_lifetime(&self) -> Option<&ast::Lifetime> {
        match self {
            NameLike::Lifetime(x) => Some(x),
            _ => None,
        }
    }
    pub fn text(&self) -> TokenText<'_> {
        match self {
            NameLike::NameRef(x) => x.text(),
            NameLike::Name(x) => x.text(),
            NameLike::Lifetime(x) => x.text(),
        }
    }
}
impl ast::Node for NameLike {
    fn can_cast(x: SyntaxKind) -> bool {
        matches!(x, SyntaxKind::NAME | SyntaxKind::NAME_REF | SyntaxKind::LIFETIME)
    }
    fn cast(syntax: syntax::Node) -> Option<Self> {
        let y = match syntax.kind() {
            SyntaxKind::NAME => NameLike::Name(ast::Name { syntax }),
            SyntaxKind::NAME_REF => NameLike::NameRef(ast::NameRef { syntax }),
            SyntaxKind::LIFETIME => NameLike::Lifetime(ast::Lifetime { syntax }),
            _ => return None,
        };
        Some(y)
    }
    fn syntax(&self) -> &syntax::Node {
        match self {
            NameLike::NameRef(x) => x.syntax(),
            NameLike::Name(x) => x.syntax(),
            NameLike::Lifetime(x) => x.syntax(),
        }
    }
}
const _: () = {
    use ast::{Lifetime, Name, NameRef};
    crate::impl_from!(NameRef, Name, Lifetime for NameLike);
};

#[derive(Debug, Clone, PartialEq)]
pub enum NameOrNameRef {
    Name(ast::Name),
    NameRef(ast::NameRef),
}
impl fmt::Display for NameOrNameRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NameOrNameRef::Name(x) => fmt::Display::fmt(x, f),
            NameOrNameRef::NameRef(x) => fmt::Display::fmt(x, f),
        }
    }
}
impl NameOrNameRef {
    pub fn text(&self) -> TokenText<'_> {
        match self {
            NameOrNameRef::Name(x) => x.text(),
            NameOrNameRef::NameRef(x) => x.text(),
        }
    }
}
impl ast::RecordPatField {
    pub fn for_field_name_ref(x: &ast::NameRef) -> Option<ast::RecordPatField> {
        let y = x.syntax().parent().and_then(ast::RecordPatField::cast)?;
        match y.field_name()? {
            NameOrNameRef::NameRef(r) if r == *x => Some(y),
            _ => None,
        }
    }
    pub fn for_field_name(x: &ast::Name) -> Option<ast::RecordPatField> {
        let y = x.syntax().ancestors().nth(2).and_then(ast::RecordPatField::cast)?;
        match y.field_name()? {
            NameOrNameRef::Name(n) if n == *x => Some(y),
            _ => None,
        }
    }
    pub fn parent_record_pat(&self) -> ast::RecordPat {
        self.syntax().ancestors().find_map(ast::RecordPat::cast).unwrap()
    }
    pub fn field_name(&self) -> Option<NameOrNameRef> {
        if let Some(x) = self.name_ref() {
            return Some(NameOrNameRef::NameRef(x));
        }
        match self.pat() {
            Some(ast::Pat::IdentPat(x)) => {
                let y = x.name()?;
                Some(NameOrNameRef::Name(y))
            },
            Some(ast::Pat::BoxPat(x)) => match x.pat() {
                Some(ast::Pat::IdentPat(x)) => {
                    let y = x.name()?;
                    Some(NameOrNameRef::Name(y))
                },
                _ => None,
            },
            _ => None,
        }
    }
}
impl ast::Variant {
    pub fn parent_enum(&self) -> ast::Enum {
        self.syntax()
            .parent()
            .and_then(|x| x.parent())
            .and_then(ast::Enum::cast)
            .expect("EnumVariants are always nested in Enums")
    }
    pub fn kind(&self) -> StructKind {
        StructKind::from_node(self)
    }
}
impl ast::Item {
    pub fn generic_param_list(&self) -> Option<ast::GenericParamList> {
        ast::AnyHasGenericParams::cast(self.syntax().clone())?.generic_param_list()
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldKind {
    Name(ast::NameRef),
    Index(syntax::Token),
}
impl ast::FieldExpr {
    pub fn index_token(&self) -> Option<syntax::Token> {
        self.syntax
            .children_with_tokens()
            .find(|x| x.kind() == SyntaxKind::INT_NUMBER || x.kind() == SyntaxKind::FLOAT_NUMBER)
            .as_ref()
            .and_then(syntax::Elem::as_token)
            .cloned()
    }
    pub fn field_access(&self) -> Option<FieldKind> {
        match self.name_ref() {
            Some(x) => Some(FieldKind::Name(x)),
            None => self.index_token().map(FieldKind::Index),
        }
    }
}
pub struct SlicePatComponents {
    pub prefix: Vec<ast::Pat>,
    pub slice: Option<ast::Pat>,
    pub suffix: Vec<ast::Pat>,
}
impl ast::SlicePat {
    pub fn components(&self) -> SlicePatComponents {
        let mut xs = self.pats().peekable();
        let prefix = xs
            .peeking_take_while(|x| match x {
                ast::Pat::RestPat(_) => false,
                ast::Pat::IdentPat(x) => !matches!(x.pat(), Some(ast::Pat::RestPat(_))),
                ast::Pat::RefPat(x) => match x.pat() {
                    Some(ast::Pat::RestPat(_)) => false,
                    Some(ast::Pat::IdentPat(x)) => !matches!(x.pat(), Some(ast::Pat::RestPat(_))),
                    _ => true,
                },
                _ => true,
            })
            .collect();
        let slice = xs.next();
        let suffix = xs.collect();
        SlicePatComponents { prefix, slice, suffix }
    }
}
impl ast::IdentPat {
    pub fn is_simple_ident(&self) -> bool {
        self.at_token().is_none() && self.mut_token().is_none() && self.ref_token().is_none() && self.pat().is_none()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SelfParamKind {
    Owned,
    Ref,
    MutRef,
}
impl ast::SelfParam {
    pub fn kind(&self) -> SelfParamKind {
        if self.amp_token().is_some() {
            if self.mut_token().is_some() {
                SelfParamKind::MutRef
            } else {
                SelfParamKind::Ref
            }
        } else {
            SelfParamKind::Owned
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeBoundKind {
    PathType(ast::PathType),
    ForType(ast::ForType),
    Lifetime(ast::Lifetime),
}
impl ast::TypeBound {
    pub fn kind(&self) -> TypeBoundKind {
        if let Some(x) = support::children(self.syntax()).next() {
            TypeBoundKind::PathType(x)
        } else if let Some(x) = support::children(self.syntax()).next() {
            TypeBoundKind::ForType(x)
        } else if let Some(x) = self.lifetime() {
            TypeBoundKind::Lifetime(x)
        } else {
            unreachable!()
        }
    }
}
#[derive(Debug, Clone)]
pub enum TypeOrConstParam {
    Type(ast::TypeParam),
    Const(ast::ConstParam),
}
impl TypeOrConstParam {
    pub fn name(&self) -> Option<ast::Name> {
        match self {
            TypeOrConstParam::Type(x) => x.name(),
            TypeOrConstParam::Const(x) => x.name(),
        }
    }
}
impl ast::Node for TypeOrConstParam {
    fn can_cast(x: SyntaxKind) -> bool
    where
        Self: Sized,
    {
        matches!(x, SyntaxKind::TYPE_PARAM | SyntaxKind::CONST_PARAM)
    }
    fn cast(syntax: syntax::Node) -> Option<Self>
    where
        Self: Sized,
    {
        let y = match syntax.kind() {
            SyntaxKind::TYPE_PARAM => TypeOrConstParam::Type(ast::TypeParam { syntax }),
            SyntaxKind::CONST_PARAM => TypeOrConstParam::Const(ast::ConstParam { syntax }),
            _ => return None,
        };
        Some(y)
    }
    fn syntax(&self) -> &syntax::Node {
        match self {
            TypeOrConstParam::Type(x) => x.syntax(),
            TypeOrConstParam::Const(x) => x.syntax(),
        }
    }
}
impl ast::HasAttrs for TypeOrConstParam {}
#[derive(Debug, Clone)]
pub enum TraitOrAlias {
    Trait(ast::Trait),
    TraitAlias(ast::TraitAlias),
}
impl TraitOrAlias {
    pub fn name(&self) -> Option<ast::Name> {
        match self {
            TraitOrAlias::Trait(x) => x.name(),
            TraitOrAlias::TraitAlias(x) => x.name(),
        }
    }
}
impl ast::Node for TraitOrAlias {
    fn can_cast(x: SyntaxKind) -> bool
    where
        Self: Sized,
    {
        matches!(x, SyntaxKind::TRAIT | SyntaxKind::TRAIT_ALIAS)
    }
    fn cast(syntax: syntax::Node) -> Option<Self>
    where
        Self: Sized,
    {
        let y = match syntax.kind() {
            SyntaxKind::TRAIT => TraitOrAlias::Trait(ast::Trait { syntax }),
            SyntaxKind::TRAIT_ALIAS => TraitOrAlias::TraitAlias(ast::TraitAlias { syntax }),
            _ => return None,
        };
        Some(y)
    }
    fn syntax(&self) -> &syntax::Node {
        match self {
            TraitOrAlias::Trait(x) => x.syntax(),
            TraitOrAlias::TraitAlias(x) => x.syntax(),
        }
    }
}
impl ast::HasAttrs for TraitOrAlias {}
pub enum VisibilityKind {
    In(ast::Path),
    PubCrate,
    PubSuper,
    PubSelf,
    Pub,
}
impl ast::Visibility {
    pub fn kind(&self) -> VisibilityKind {
        match self.path() {
            Some(x) => {
                if let Some(x) = x.as_single_segment().filter(|x| x.coloncolon_token().is_none()) {
                    if x.crate_token().is_some() {
                        return VisibilityKind::PubCrate;
                    } else if x.super_token().is_some() {
                        return VisibilityKind::PubSuper;
                    } else if x.self_token().is_some() {
                        return VisibilityKind::PubSelf;
                    }
                }
                VisibilityKind::In(x)
            },
            None => VisibilityKind::Pub,
        }
    }
}
impl ast::LifetimeParam {
    pub fn lifetime_bounds(&self) -> impl Iterator<Item = syntax::Token> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|x| x.into_token())
            .skip_while(|x| x.kind() != T![:])
            .filter(|x| x.kind() == T![lifetime_ident])
    }
}
impl ast::Module {
    pub fn parent(&self) -> Option<ast::Module> {
        self.syntax().ancestors().nth(2).and_then(ast::Module::cast)
    }
}
impl ast::RangePat {
    pub fn start(&self) -> Option<ast::Pat> {
        self.syntax()
            .children_with_tokens()
            .take_while(|x| !(x.kind() == T![..] || x.kind() == T![..=]))
            .filter_map(|x| x.into_node())
            .find_map(ast::Pat::cast)
    }
    pub fn end(&self) -> Option<ast::Pat> {
        self.syntax()
            .children_with_tokens()
            .skip_while(|x| !(x.kind() == T![..] || x.kind() == T![..=]))
            .filter_map(|x| x.into_node())
            .find_map(ast::Pat::cast)
    }
}
impl ast::TokenTree {
    pub fn token_trees_and_tokens(&self) -> impl Iterator<Item = NodeOrToken<ast::TokenTree, syntax::Token>> {
        self.syntax().children_with_tokens().filter_map(|x| match x {
            NodeOrToken::Node(x) => ast::TokenTree::cast(x).map(NodeOrToken::Node),
            NodeOrToken::Token(x) => Some(NodeOrToken::Token(x)),
        })
    }
    pub fn left_delimiter_token(&self) -> Option<syntax::Token> {
        self.syntax()
            .first_child_or_token()?
            .into_token()
            .filter(|x| matches!(x.kind(), T!['{'] | T!['('] | T!['[']))
    }
    pub fn right_delimiter_token(&self) -> Option<syntax::Token> {
        self.syntax()
            .last_child_or_token()?
            .into_token()
            .filter(|x| matches!(x.kind(), T!['}'] | T![')'] | T![']']))
    }
    pub fn parent_meta(&self) -> Option<ast::Meta> {
        self.syntax().parent().and_then(ast::Meta::cast)
    }
}
impl ast::Meta {
    pub fn parent_attr(&self) -> Option<ast::Attr> {
        self.syntax().parent().and_then(ast::Attr::cast)
    }
}
impl ast::GenericArgList {
    pub fn lifetime_args(&self) -> impl Iterator<Item = ast::LifetimeArg> {
        self.generic_args().filter_map(|x| match x {
            ast::GenericArg::LifetimeArg(x) => Some(x),
            _ => None,
        })
    }
}
impl ast::GenericParamList {
    pub fn lifetime_params(&self) -> impl Iterator<Item = ast::LifetimeParam> {
        self.generic_params().filter_map(|x| match x {
            ast::GenericParam::LifetimeParam(x) => Some(x),
            ast::GenericParam::TypeParam(_) | ast::GenericParam::ConstParam(_) => None,
        })
    }
    pub fn type_or_const_params(&self) -> impl Iterator<Item = ast::TypeOrConstParam> {
        self.generic_params().filter_map(|x| match x {
            ast::GenericParam::TypeParam(x) => Some(ast::TypeOrConstParam::Type(x)),
            ast::GenericParam::LifetimeParam(_) => None,
            ast::GenericParam::ConstParam(x) => Some(ast::TypeOrConstParam::Const(x)),
        })
    }
}
impl ast::ForExpr {
    pub fn iterable(&self) -> Option<ast::Expr> {
        let mut xs = support::children(self.syntax());
        let first = xs.next();
        match first {
            Some(ast::Expr::BlockExpr(_)) => xs.next().and(first),
            first => first,
        }
    }
}
impl ast::HasLoopBody for ast::ForExpr {
    fn loop_body(&self) -> Option<ast::BlockExpr> {
        let mut xs = support::children(self.syntax());
        let first = xs.next();
        let second = xs.next();
        second.or(first)
    }
}
impl ast::WhileExpr {
    pub fn condition(&self) -> Option<ast::Expr> {
        let mut xs = support::children(self.syntax());
        let first = xs.next();
        match first {
            Some(ast::Expr::BlockExpr(_)) => xs.next().and(first),
            first => first,
        }
    }
}
impl ast::HasLoopBody for ast::WhileExpr {
    fn loop_body(&self) -> Option<ast::BlockExpr> {
        let mut xs = support::children(self.syntax());
        let first = xs.next();
        let second = xs.next();
        second.or(first)
    }
}
impl ast::HasAttrs for ast::AnyHasDocComments {}
impl From<ast::Adt> for ast::Item {
    fn from(x: ast::Adt) -> Self {
        match x {
            ast::Adt::Enum(x) => ast::Item::Enum(x),
            ast::Adt::Struct(x) => ast::Item::Struct(x),
            ast::Adt::Union(x) => ast::Item::Union(x),
        }
    }
}
impl ast::MatchGuard {
    pub fn condition(&self) -> Option<ast::Expr> {
        support::child(&self.syntax)
    }
}
impl From<ast::Item> for ast::AnyHasAttrs {
    fn from(x: ast::Item) -> Self {
        Self::new(x)
    }
}
impl From<ast::AssocItem> for ast::AnyHasAttrs {
    fn from(x: ast::AssocItem) -> Self {
        Self::new(x)
    }
}
impl From<ast::Variant> for ast::AnyHasAttrs {
    fn from(x: ast::Variant) -> Self {
        Self::new(x)
    }
}
impl From<ast::RecordField> for ast::AnyHasAttrs {
    fn from(x: ast::RecordField) -> Self {
        Self::new(x)
    }
}
impl From<ast::TupleField> for ast::AnyHasAttrs {
    fn from(x: ast::TupleField) -> Self {
        Self::new(x)
    }
}
