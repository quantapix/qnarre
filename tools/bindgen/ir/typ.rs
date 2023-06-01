use super::comp::CompInfo;
use super::dot::DotAttrs;
use super::enum_ty::Enum;
use super::func::FnSig;
use super::int::IntKind;
use super::item::{IsOpaque, Item};
use super::layout::{Layout, Opaque};
use super::template::{AsTemplParam, TemplInstantiation, TemplParams};
use super::{Context, EdgeKind, ItemId, Trace, Tracer, TypeId};
use crate::clang::{self, Cursor};
use crate::parse;
use std::borrow::Cow;
use std::io;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FloatKind {
    Float,
    Double,
    LongDouble,
    Float128,
}

#[derive(Debug)]
pub enum TypeKind {
    Void,
    NullPtr,
    Comp(CompInfo),
    Opaque,
    Int(IntKind),
    Float(FloatKind),
    Complex(FloatKind),
    Alias(TypeId),
    TemplAlias(TypeId, Vec<TypeId>),
    Vector(TypeId, usize),
    Array(TypeId, usize),
    Func(FnSig),
    Enum(Enum),
    Pointer(TypeId),
    BlockPtr(TypeId),
    Reference(TypeId),
    TemplInstantiation(TemplInstantiation),
    UnresolvedTypeRef(clang::Type, clang::Cursor, /* parent_id */ Option<ItemId>),
    ResolvedTypeRef(TypeId),
    TypeParam,
}
impl TypeKind {
    fn kind_name(&self) -> &'static str {
        match *self {
            TypeKind::Void => "Void",
            TypeKind::NullPtr => "NullPtr",
            TypeKind::Comp(..) => "Comp",
            TypeKind::Opaque => "Opaque",
            TypeKind::Int(..) => "Int",
            TypeKind::Float(..) => "Float",
            TypeKind::Complex(..) => "Complex",
            TypeKind::Alias(..) => "Alias",
            TypeKind::TemplAlias(..) => "TemplateAlias",
            TypeKind::Array(..) => "Array",
            TypeKind::Vector(..) => "Vector",
            TypeKind::Func(..) => "Func",
            TypeKind::Enum(..) => "Enum",
            TypeKind::Pointer(..) => "Pointer",
            TypeKind::BlockPtr(..) => "BlockPtr",
            TypeKind::Reference(..) => "Reference",
            TypeKind::TemplInstantiation(..) => "TemplateInstantiation",
            TypeKind::UnresolvedTypeRef(..) => "UnresolvedTypeRef",
            TypeKind::ResolvedTypeRef(..) => "ResolvedTypeRef",
            TypeKind::TypeParam => "TypeParam",
        }
    }
}
impl DotAttrs for TypeKind {
    fn dot_attrs<W>(&self, ctx: &Context, y: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(y, "<tr><td>type kind</td><td>{}</td></tr>", self.kind_name())?;
        if let TypeKind::Comp(ref x) = *self {
            x.dot_attrs(ctx, y)?;
        }
        Ok(())
    }
}
impl TemplParams for TypeKind {
    fn self_templ_params(&self, ctx: &Context) -> Vec<TypeId> {
        match *self {
            TypeKind::ResolvedTypeRef(x) => ctx.resolve_type(x).self_templ_params(ctx),
            TypeKind::Comp(ref x) => x.self_templ_params(ctx),
            TypeKind::TemplAlias(_, ref x) => x.clone(),
            TypeKind::Opaque
            | TypeKind::TemplInstantiation(..)
            | TypeKind::Void
            | TypeKind::NullPtr
            | TypeKind::Int(_)
            | TypeKind::Float(_)
            | TypeKind::Complex(_)
            | TypeKind::Array(..)
            | TypeKind::Vector(..)
            | TypeKind::Func(_)
            | TypeKind::Enum(_)
            | TypeKind::Pointer(_)
            | TypeKind::BlockPtr(_)
            | TypeKind::Reference(_)
            | TypeKind::UnresolvedTypeRef(..)
            | TypeKind::TypeParam
            | TypeKind::Alias(_) => vec![],
        }
    }
}
impl AsTemplParam for TypeKind {
    type Extra = Item;
    fn as_templ_param(&self, ctx: &Context, it: &Item) -> Option<TypeId> {
        match *self {
            TypeKind::TypeParam => Some(it.id().expect_type_id(ctx)),
            TypeKind::ResolvedTypeRef(x) => x.as_templ_param(ctx, &()),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Type {
    name: Option<String>,
    layout: Option<Layout>,
    kind: TypeKind,
    is_const: bool,
}
impl Type {
    pub fn as_comp_mut(&mut self) -> Option<&mut CompInfo> {
        match self.kind {
            TypeKind::Comp(ref mut x) => Some(x),
            _ => None,
        }
    }
    pub fn new(name: Option<String>, layout: Option<Layout>, kind: TypeKind, is_const: bool) -> Self {
        Type {
            name,
            layout,
            kind,
            is_const,
        }
    }
    pub fn kind(&self) -> &TypeKind {
        &self.kind
    }
    pub fn kind_mut(&mut self) -> &mut TypeKind {
        &mut self.kind
    }
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    pub fn is_block_ptr(&self) -> bool {
        matches!(self.kind, TypeKind::BlockPtr(..))
    }
    pub fn is_int(&self) -> bool {
        matches!(self.kind, TypeKind::Int(_))
    }
    pub fn is_comp(&self) -> bool {
        matches!(self.kind, TypeKind::Comp(..))
    }
    pub fn is_union(&self) -> bool {
        match self.kind {
            TypeKind::Comp(ref x) => x.is_union(),
            _ => false,
        }
    }
    pub fn is_type_param(&self) -> bool {
        matches!(self.kind, TypeKind::TypeParam)
    }
    pub fn is_templ_instantiation(&self) -> bool {
        matches!(self.kind, TypeKind::TemplInstantiation(..))
    }
    pub fn is_fn(&self) -> bool {
        matches!(self.kind, TypeKind::Func(..))
    }
    pub fn is_enum(&self) -> bool {
        matches!(self.kind, TypeKind::Enum(..))
    }
    pub fn is_void(&self) -> bool {
        matches!(self.kind, TypeKind::Void)
    }
    pub fn is_builtin_or_type_param(&self) -> bool {
        matches!(
            self.kind,
            TypeKind::Void
                | TypeKind::NullPtr
                | TypeKind::Func(..)
                | TypeKind::Array(..)
                | TypeKind::Reference(..)
                | TypeKind::Pointer(..)
                | TypeKind::Int(..)
                | TypeKind::Float(..)
                | TypeKind::TypeParam
        )
    }
    pub fn named(name: String) -> Self {
        let name = if name.is_empty() { None } else { Some(name) };
        Self::new(name, None, TypeKind::TypeParam, false)
    }
    pub fn is_float(&self) -> bool {
        matches!(self.kind, TypeKind::Float(..))
    }
    pub fn is_bool(&self) -> bool {
        matches!(self.kind, TypeKind::Int(IntKind::Bool))
    }
    pub fn is_integer(&self) -> bool {
        matches!(self.kind, TypeKind::Int(..))
    }
    pub fn as_integer(&self) -> Option<IntKind> {
        match self.kind {
            TypeKind::Int(x) => Some(x),
            _ => None,
        }
    }
    pub fn is_const(&self) -> bool {
        self.is_const
    }
    pub fn is_unresolved_ref(&self) -> bool {
        matches!(self.kind, TypeKind::UnresolvedTypeRef(_, _, _))
    }
    pub fn is_incomplete_array(&self, ctx: &Context) -> Option<ItemId> {
        match self.kind {
            TypeKind::Array(x, len) => {
                if len == 0 {
                    Some(x.into())
                } else {
                    None
                }
            },
            TypeKind::ResolvedTypeRef(x) => ctx.resolve_type(x).is_incomplete_array(ctx),
            _ => None,
        }
    }
    pub fn layout(&self, ctx: &Context) -> Option<Layout> {
        self.layout.or_else(|| match self.kind {
            TypeKind::Comp(ref x) => x.layout(ctx),
            TypeKind::Array(x, len) if len == 0 => Some(Layout::new(0, ctx.resolve_type(x).layout(ctx)?.align)),
            TypeKind::Pointer(..) => Some(Layout::new(ctx.target_pointer_size(), ctx.target_pointer_size())),
            TypeKind::ResolvedTypeRef(x) => ctx.resolve_type(x).layout(ctx),
            _ => None,
        })
    }
    pub fn is_invalid_type_param(&self) -> bool {
        match self.kind {
            TypeKind::TypeParam => {
                let name = self.name().expect("Unnamed named type?");
                !clang::is_valid_identifier(name)
            },
            _ => false,
        }
    }
    fn sanitize_name(name: &str) -> Cow<str> {
        if clang::is_valid_identifier(name) {
            return Cow::Borrowed(name);
        }
        let name = name.replace(|x| x == ' ' || x == ':' || x == '.', "_");
        Cow::Owned(name)
    }
    pub fn sanitized_name<'a>(&'a self, ctx: &Context) -> Option<Cow<'a, str>> {
        let y = match *self.kind() {
            TypeKind::Pointer(x) => Some((x, Cow::Borrowed("ptr"))),
            TypeKind::Reference(x) => Some((x, Cow::Borrowed("ref"))),
            TypeKind::Array(x, len) => Some((x, format!("array{}", len).into())),
            _ => None,
        };
        if let Some((x, pre)) = y {
            ctx.resolve_item(x)
                .expect_type()
                .sanitized_name(ctx)
                .map(|x| format!("{}_{}", pre, x).into())
        } else {
            self.name().map(Self::sanitize_name)
        }
    }
    pub fn canon_type<'tr>(&'tr self, ctx: &'tr Context) -> &'tr Type {
        self.safe_canon_type(ctx)
            .expect("Should have been resolved after parsing!")
    }
    pub fn safe_canon_type<'tr>(&'tr self, ctx: &'tr Context) -> Option<&'tr Type> {
        match self.kind {
            TypeKind::TypeParam
            | TypeKind::Array(..)
            | TypeKind::Vector(..)
            | TypeKind::Comp(..)
            | TypeKind::Opaque
            | TypeKind::Int(..)
            | TypeKind::Float(..)
            | TypeKind::Complex(..)
            | TypeKind::Func(..)
            | TypeKind::Enum(..)
            | TypeKind::Reference(..)
            | TypeKind::Void
            | TypeKind::NullPtr
            | TypeKind::Pointer(..)
            | TypeKind::BlockPtr(..) => Some(self),
            TypeKind::ResolvedTypeRef(x) | TypeKind::Alias(x) | TypeKind::TemplAlias(x, _) => {
                ctx.resolve_type(x).safe_canon_type(ctx)
            },
            TypeKind::TemplInstantiation(ref x) => ctx.resolve_type(x.templ_def()).safe_canon_type(ctx),
            TypeKind::UnresolvedTypeRef(..) => None,
        }
    }
    pub fn should_be_traced(&self) -> bool {
        matches!(
            self.kind,
            TypeKind::Comp(..)
                | TypeKind::Func(..)
                | TypeKind::Pointer(..)
                | TypeKind::Array(..)
                | TypeKind::Reference(..)
                | TypeKind::TemplInstantiation(..)
                | TypeKind::ResolvedTypeRef(..)
        )
    }
    pub fn from_clang_ty(
        id: ItemId,
        ty: &clang::Type,
        cur: Cursor,
        parent: Option<ItemId>,
        ctx: &mut Context,
    ) -> Result<parse::Resolved<Self>, parse::Error> {
        use clang_lib::*;
        {
            let y = ctx.builtin_or_resolved_ty(id, parent, ty, Some(cur));
            if let Some(x) = y {
                return Ok(parse::Resolved::AlreadyDone(x.into()));
            }
        }
        let layout = ty.fallible_layout(ctx).ok();
        let cur = ty.declaration();
        let is_anon = cur.is_anonymous();
        let mut name = if is_anon {
            None
        } else {
            Some(cur.spelling()).filter(|x| !x.is_empty())
        };
        debug!("from_clang_ty: {:?}, ty: {:?}, loc: {:?}", id, ty, cur);
        debug!("currently_parsed_types: {:?}", ctx.currently_parsed_types());
        let canon_ty = ty.canonical_type();
        let mut ty_kind = ty.kind();
        if ty_kind == CXType_Typedef {
            let is_templ = ty.declaration().kind() == CXCursor_TemplateTypeParameter;
        }
        if cur.kind() == CXCursor_ClassTemplatePartialSpecialization {
            return Ok(parse::Resolved::New(Opaque::from_clang_ty(&canon_ty, ctx), None));
        }
        let kind = if cur.kind() == CXCursor_TemplateRef || (ty.templ_args().is_some() && ty_kind != CXType_Typedef) {
            match TemplInstantiation::from_ty(ty, ctx) {
                Some(x) => TypeKind::TemplInstantiation(x),
                None => TypeKind::Opaque,
            }
        } else {
            match ty_kind {
                CXType_Unexposed
                    if *ty != canon_ty
                        && canon_ty.kind() != CXType_Invalid
                        && ty.ret_type().is_none()
                        && !canon_ty.spelling().contains("type-parameter") =>
                {
                    return Self::from_clang_ty(id, &canon_ty, cur, parent, ctx);
                },
                CXType_Unexposed | CXType_Invalid => {
                    if ty.ret_type().is_some() {
                        let x = FnSig::from_ty(ty, &cur, ctx)?;
                        TypeKind::Func(x)
                    } else if ty.is_fully_instantiated_templ() {
                        let x = CompInfo::from_ty(id, ty, Some(cur), ctx).expect("C'mon");
                        TypeKind::Comp(x)
                    } else {
                        match cur.kind() {
                            CXCursor_CXXBaseSpecifier | CXCursor_ClassTemplate => {
                                if cur.kind() == CXCursor_CXXBaseSpecifier {
                                    if cur.spelling().chars().all(|x| x.is_alphanumeric() || x == '_') {
                                        return Err(parse::Error::Recurse);
                                    }
                                } else {
                                    name = Some(cur.spelling());
                                }
                                let x = CompInfo::from_ty(id, ty, Some(cur), ctx);
                                match x {
                                    Ok(x) => TypeKind::Comp(x),
                                    Err(_) => {
                                        let x = Opaque::from_clang_ty(ty, ctx);
                                        return Ok(parse::Resolved::New(x, None));
                                    },
                                }
                            },
                            CXCursor_TypeAliasTemplateDecl => {
                                let mut y = Err(parse::Error::Continue);
                                let mut args = vec![];
                                cur.visit(|x| {
                                    match x.kind() {
                                        CXCursor_TypeAliasDecl => {
                                            debug_assert_eq!(x.cur_type().kind(), CXType_Typedef);
                                            name = Some(x.spelling());
                                            let x2 = x.typedef_type().expect("Not valid Type?");
                                            y = Ok(Item::from_ty_or_ref(x2, x, Some(id), ctx));
                                        },
                                        CXCursor_TemplateTypeParameter => {
                                            let x = Item::type_param(None, x, ctx).expect("A TemplateTypeParameter");
                                            args.push(x);
                                        },
                                        _ => {},
                                    }
                                    CXChildVisit_Continue
                                });
                                let y = match y {
                                    Ok(x) => x,
                                    Err(..) => {
                                        warn!("Failed to parse template alias {:?}", cur);
                                        return Err(parse::Error::Continue);
                                    },
                                };
                                TypeKind::TemplAlias(y, args)
                            },
                            CXCursor_TemplateRef => {
                                let x = cur.referenced().unwrap();
                                let ty = x.cur_type();
                                return Self::from_clang_ty(id, &ty, x, parent, ctx);
                            },
                            CXCursor_TypeRef => {
                                let x = cur.referenced().unwrap();
                                let ty = x.cur_type();
                                let decl = ty.declaration();
                                let y = Item::from_ty_or_ref_with_id(id, ty, decl, parent, ctx);
                                return Ok(parse::Resolved::AlreadyDone(y.into()));
                            },
                            CXCursor_NamespaceRef => {
                                return Err(parse::Error::Continue);
                            },
                            _ => {
                                if ty.kind() == CXType_Unexposed {
                                    return Err(parse::Error::Recurse);
                                }
                                warn!("invalid type {:?}", ty);
                                return Err(parse::Error::Continue);
                            },
                        }
                    }
                },
                CXType_Auto => {
                    if canon_ty == *ty {
                        debug!("Couldn't find deduced type: {:?}", ty);
                        return Err(parse::Error::Continue);
                    }
                    return Self::from_clang_ty(id, &canon_ty, cur, parent, ctx);
                },
                CXType_MemberPointer | CXType_Pointer => {
                    let mut y = ty.pointee_type().unwrap();
                    if *ty != canon_ty {
                        let x = canon_ty.pointee_type().unwrap();
                        if x.is_const() != y.is_const() {
                            y = x;
                        }
                    }
                    let y = Item::from_ty_or_ref(y, cur, None, ctx);
                    TypeKind::Pointer(y)
                },
                CXType_BlockPointer => {
                    let x = ty.pointee_type().expect("Not valid Type?");
                    let y = Item::from_ty_or_ref(x, cur, None, ctx);
                    TypeKind::BlockPtr(y)
                },
                CXType_RValueReference | CXType_LValueReference => {
                    let y = Item::from_ty_or_ref(ty.pointee_type().unwrap(), cur, None, ctx);
                    TypeKind::Reference(y)
                },
                CXType_VariableArray | CXType_DependentSizedArray => {
                    let y = Item::from_ty(ty.elem_type().as_ref().unwrap(), cur, None, ctx)
                        .expect("Not able to resolve array element?");
                    TypeKind::Pointer(y)
                },
                CXType_IncompleteArray => {
                    let y = Item::from_ty(ty.elem_type().as_ref().unwrap(), cur, None, ctx)
                        .expect("Not able to resolve array element?");
                    TypeKind::Array(y, 0)
                },
                CXType_FunctionNoProto | CXType_FunctionProto => {
                    let y = FnSig::from_ty(ty, &cur, ctx)?;
                    TypeKind::Func(y)
                },
                CXType_Typedef => {
                    let x = cur.typedef_type().expect("Not valid Type?");
                    let y = Item::from_ty_or_ref(x, cur, None, ctx);
                    if y == id {
                        TypeKind::Opaque
                    } else {
                        if let Some(ref mut name) = name {
                            if x.kind() == CXType_Pointer && !ctx.opts().c_naming {
                                let x = x.pointee_type().unwrap();
                                if x.kind() == CXType_Elaborated && x.declaration().spelling() == *name {
                                    *name += "_ptr";
                                }
                            }
                        }
                        TypeKind::Alias(y)
                    }
                },
                CXType_Enum => {
                    let y = Enum::from_ty(ty, ctx).expect("Not an enum?");
                    if !is_anon {
                        let x = ty.spelling();
                        if clang::is_valid_identifier(&x) {
                            name = Some(x);
                        }
                    }
                    TypeKind::Enum(y)
                },
                CXType_Record => {
                    let y = CompInfo::from_ty(id, ty, Some(cur), ctx).expect("Not a complex type?");
                    if !is_anon {
                        let x = ty.spelling();
                        if clang::is_valid_identifier(&x) {
                            name = Some(x);
                        }
                    }
                    TypeKind::Comp(y)
                },
                CXType_Vector => {
                    let y = Item::from_ty(ty.elem_type().as_ref().unwrap(), cur, None, ctx)?;
                    TypeKind::Vector(y, ty.num_elements().unwrap())
                },
                CXType_ConstantArray => {
                    let y = Item::from_ty(ty.elem_type().as_ref().unwrap(), cur, None, ctx)
                        .expect("Not able to resolve array element?");
                    TypeKind::Array(y, ty.num_elements().unwrap())
                },
                CXType_Elaborated => {
                    return Self::from_clang_ty(id, &ty.named(), cur, parent, ctx);
                },
                CXType_Dependent => {
                    return Err(parse::Error::Continue);
                },
                _ => {
                    warn!("unsupported type: kind = {:?}; ty = {:?}; at {:?}", ty.kind(), ty, cur);
                    return Err(parse::Error::Continue);
                },
            }
        };
        name = name.filter(|x| !x.is_empty());
        let is_const =
            ty.is_const() || (ty.kind() == CXType_ConstantArray && ty.elem_type().map_or(false, |x| x.is_const()));
        let y = Type::new(name, layout, kind, is_const);
        Ok(parse::Resolved::New(y, Some(cur.canonical())))
    }
}
impl IsOpaque for Type {
    type Extra = Item;
    fn is_opaque(&self, ctx: &Context, it: &Item) -> bool {
        match self.kind {
            TypeKind::Opaque => true,
            TypeKind::TemplInstantiation(ref x) => x.is_opaque(ctx, it),
            TypeKind::Comp(ref x) => x.is_opaque(ctx, &self.layout),
            TypeKind::ResolvedTypeRef(x) => x.is_opaque(ctx, &()),
            _ => false,
        }
    }
}
impl DotAttrs for Type {
    fn dot_attrs<W>(&self, ctx: &Context, y: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        if let Some(ref x) = self.layout {
            writeln!(
                y,
                "<tr><td>size</td><td>{}</td></tr>
                           <tr><td>align</td><td>{}</td></tr>",
                x.size, x.align
            )?;
            if x.packed {
                writeln!(y, "<tr><td>packed</td><td>true</td></tr>")?;
            }
        }
        if self.is_const {
            writeln!(y, "<tr><td>const</td><td>true</td></tr>")?;
        }
        self.kind.dot_attrs(ctx, y)
    }
}
impl TemplParams for Type {
    fn self_templ_params(&self, ctx: &Context) -> Vec<TypeId> {
        self.kind.self_templ_params(ctx)
    }
}
impl AsTemplParam for Type {
    type Extra = Item;
    fn as_templ_param(&self, ctx: &Context, it: &Item) -> Option<TypeId> {
        self.kind.as_templ_param(ctx, it)
    }
}
impl Trace for Type {
    type Extra = Item;
    fn trace<T>(&self, ctx: &Context, tracer: &mut T, it: &Item)
    where
        T: Tracer,
    {
        if self.name().map_or(false, |x| ctx.is_stdint_type(x)) {
            return;
        }
        match *self.kind() {
            TypeKind::Pointer(x)
            | TypeKind::Reference(x)
            | TypeKind::Array(x, _)
            | TypeKind::Vector(x, _)
            | TypeKind::BlockPtr(x)
            | TypeKind::Alias(x)
            | TypeKind::ResolvedTypeRef(x) => {
                tracer.visit_kind(x.into(), EdgeKind::TypeRef);
            },
            TypeKind::TemplAlias(x, ref ps) => {
                tracer.visit_kind(x.into(), EdgeKind::TypeRef);
                for p in ps {
                    tracer.visit_kind(p.into(), EdgeKind::TemplParamDef);
                }
            },
            TypeKind::TemplInstantiation(ref x) => {
                x.trace(ctx, tracer, &());
            },
            TypeKind::Comp(ref x) => x.trace(ctx, tracer, it),
            TypeKind::Func(ref x) => x.trace(ctx, tracer, &()),
            TypeKind::Enum(ref x) => {
                if let Some(x) = x.repr() {
                    tracer.visit(x.into());
                }
            },
            TypeKind::UnresolvedTypeRef(_, _, Some(x)) => {
                tracer.visit(x);
            },
            TypeKind::Opaque
            | TypeKind::UnresolvedTypeRef(_, _, None)
            | TypeKind::TypeParam
            | TypeKind::Void
            | TypeKind::NullPtr
            | TypeKind::Int(_)
            | TypeKind::Float(_)
            | TypeKind::Complex(_) => {},
        }
    }
}

#[test]
fn is_invalid_type_param_valid() {
    let y = Type::new(Some("foo".into()), None, TypeKind::TypeParam, false);
    assert!(!y.is_invalid_type_param())
}
#[test]
fn is_invalid_type_param_valid_underscore() {
    let y = Type::new(Some("_foo123456789_".into()), None, TypeKind::TypeParam, false);
    assert!(!y.is_invalid_type_param())
}
#[test]
fn is_invalid_type_param_valid_unnamed() {
    let y = Type::new(Some("foo".into()), None, TypeKind::Void, false);
    assert!(!y.is_invalid_type_param())
}
#[test]
fn is_invalid_type_param_invalid_start() {
    let y = Type::new(Some("1foo".into()), None, TypeKind::TypeParam, false);
    assert!(y.is_invalid_type_param())
}
#[test]
fn is_invalid_type_param_invalid_remaing() {
    let y = Type::new(Some("foo-".into()), None, TypeKind::TypeParam, false);
    assert!(y.is_invalid_type_param())
}
#[test]
#[should_panic]
fn is_invalid_type_param_unnamed() {
    let y = Type::new(None, None, TypeKind::TypeParam, false);
    assert!(y.is_invalid_type_param())
}
#[test]
fn is_invalid_type_param_empty_name() {
    let y = Type::new(Some("".into()), None, TypeKind::TypeParam, false);
    assert!(y.is_invalid_type_param())
}
