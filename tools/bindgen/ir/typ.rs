use super::comp::CompInfo;
use super::dot::DotAttrs;
use super::enum_ty::Enum;
use super::function::FnSig;
use super::int::IntKind;
use super::item::{IsOpaque, Item};
use super::layout::{Layout, Opaque};
use super::template::{AsTemplParam, TemplInstantiation, TemplParams};
use super::traversal::{EdgeKind, Trace, Tracer};
use super::{Context, ItemId, TypeId};
use crate::clang::{self, Cursor};
use crate::parse;
use std::borrow::Cow;
use std::io;

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
            TypeKind::Comp(ref mut ci) => Some(ci),
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
    pub fn is_block_pointer(&self) -> bool {
        matches!(self.kind, TypeKind::BlockPointer(..))
    }
    pub fn is_int(&self) -> bool {
        matches!(self.kind, TypeKind::Int(_))
    }
    pub fn is_comp(&self) -> bool {
        matches!(self.kind, TypeKind::Comp(..))
    }
    pub fn is_union(&self) -> bool {
        match self.kind {
            TypeKind::Comp(ref comp) => comp.is_union(),
            _ => false,
        }
    }
    pub fn is_type_param(&self) -> bool {
        matches!(self.kind, TypeKind::TypeParam)
    }
    pub fn is_template_instantiation(&self) -> bool {
        matches!(self.kind, TypeKind::TemplateInstantiation(..))
    }
    pub fn is_function(&self) -> bool {
        matches!(self.kind, TypeKind::Function(..))
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
                | TypeKind::Function(..)
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
            TypeKind::Int(int_kind) => Some(int_kind),
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
            TypeKind::Array(item, len) => {
                if len == 0 {
                    Some(item.into())
                } else {
                    None
                }
            },
            TypeKind::ResolvedTypeRef(inner) => ctx.resolve_type(inner).is_incomplete_array(ctx),
            _ => None,
        }
    }
    pub fn layout(&self, ctx: &Context) -> Option<Layout> {
        self.layout.or_else(|| match self.kind {
            TypeKind::Comp(ref ci) => ci.layout(ctx),
            TypeKind::Array(inner, length) if length == 0 => {
                Some(Layout::new(0, ctx.resolve_type(inner).layout(ctx)?.align))
            },
            TypeKind::Pointer(..) => Some(Layout::new(ctx.target_pointer_size(), ctx.target_pointer_size())),
            TypeKind::ResolvedTypeRef(inner) => ctx.resolve_type(inner).layout(ctx),
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
        let name = name.replace(|c| c == ' ' || c == ':' || c == '.', "_");
        Cow::Owned(name)
    }
    pub fn sanitized_name<'a>(&'a self, ctx: &Context) -> Option<Cow<'a, str>> {
        let name_info = match *self.kind() {
            TypeKind::Pointer(inner) => Some((inner, Cow::Borrowed("ptr"))),
            TypeKind::Reference(inner) => Some((inner, Cow::Borrowed("ref"))),
            TypeKind::Array(inner, length) => Some((inner, format!("array{}", length).into())),
            _ => None,
        };
        if let Some((inner, prefix)) = name_info {
            ctx.resolve_item(inner)
                .expect_type()
                .sanitized_name(ctx)
                .map(|name| format!("{}_{}", prefix, name).into())
        } else {
            self.name().map(Self::sanitize_name)
        }
    }
    pub fn canonical_type<'tr>(&'tr self, ctx: &'tr Context) -> &'tr Type {
        self.safe_canonical_type(ctx)
            .expect("Should have been resolved after parsing!")
    }
    pub fn safe_canonical_type<'tr>(&'tr self, ctx: &'tr Context) -> Option<&'tr Type> {
        match self.kind {
            TypeKind::TypeParam
            | TypeKind::Array(..)
            | TypeKind::Vector(..)
            | TypeKind::Comp(..)
            | TypeKind::Opaque
            | TypeKind::Int(..)
            | TypeKind::Float(..)
            | TypeKind::Complex(..)
            | TypeKind::Function(..)
            | TypeKind::Enum(..)
            | TypeKind::Reference(..)
            | TypeKind::Void
            | TypeKind::NullPtr
            | TypeKind::Pointer(..)
            | TypeKind::BlockPointer(..) => Some(self),
            TypeKind::ResolvedTypeRef(inner) | TypeKind::Alias(inner) | TypeKind::TemplateAlias(inner, _) => {
                ctx.resolve_type(inner).safe_canonical_type(ctx)
            },
            TypeKind::TemplateInstantiation(ref x) => {
                ctx.resolve_type(x.template_definition()).safe_canonical_type(ctx)
            },
            TypeKind::UnresolvedTypeRef(..) => None,
        }
    }
    pub fn should_be_traced_unconditionally(&self) -> bool {
        matches!(
            self.kind,
            TypeKind::Comp(..)
                | TypeKind::Function(..)
                | TypeKind::Pointer(..)
                | TypeKind::Array(..)
                | TypeKind::Reference(..)
                | TypeKind::TemplateInstantiation(..)
                | TypeKind::ResolvedTypeRef(..)
        )
    }
    pub fn from_clang_ty(
        potential_id: ItemId,
        ty: &clang::Type,
        location: Cursor,
        parent_id: Option<ItemId>,
        ctx: &mut Context,
    ) -> Result<parse::Resolved<Self>, parse::Error> {
        use clang_lib::*;
        {
            let already_resolved = ctx.builtin_or_resolved_ty(potential_id, parent_id, ty, Some(location));
            if let Some(ty) = already_resolved {
                debug!("{:?} already resolved: {:?}", ty, location);
                return Ok(parse::Resolved::AlreadyResolved(ty.into()));
            }
        }
        let layout = ty.fallible_layout(ctx).ok();
        let cur = ty.declaration();
        let is_anonymous = cur.is_anonymous();
        let mut name = if is_anonymous {
            None
        } else {
            Some(cur.spelling()).filter(|n| !n.is_empty())
        };
        debug!("from_clang_ty: {:?}, ty: {:?}, loc: {:?}", potential_id, ty, location);
        debug!("currently_parsed_types: {:?}", ctx.currently_parsed_types());
        let canonical_ty = ty.canonical_type();
        let mut ty_kind = ty.kind();
        if ty_kind == CXType_Typedef {
            let is_template_type_param = ty.declaration().kind() == CXCursor_TemplateTypeParameter;
        }
        if location.kind() == CXCursor_ClassTemplatePartialSpecialization {
            warn!(
                "Found a partial template specialization; bindgen does not \
                 support partial template specialization! Constructing \
                 opaque type instead."
            );
            return Ok(parse::Resolved::New(Opaque::from_clang_ty(&canonical_ty, ctx), None));
        }
        let kind = if location.kind() == CXCursor_TemplateRef
            || (ty.template_args().is_some() && ty_kind != CXType_Typedef)
        {
            match TemplInstantiation::from_ty(ty, ctx) {
                Some(inst) => TypeKind::TemplateInstantiation(inst),
                None => TypeKind::Opaque,
            }
        } else {
            match ty_kind {
                CXType_Unexposed
                    if *ty != canonical_ty
                        && canonical_ty.kind() != CXType_Invalid
                        && ty.ret_type().is_none()
                        && !canonical_ty.spelling().contains("type-parameter") =>
                {
                    debug!("Looking for canonical type: {:?}", canonical_ty);
                    return Self::from_clang_ty(potential_id, &canonical_ty, location, parent_id, ctx);
                },
                CXType_Unexposed | CXType_Invalid => {
                    if ty.ret_type().is_some() {
                        let signature = FnSig::from_ty(ty, &location, ctx)?;
                        TypeKind::Function(signature)
                    } else if ty.is_fully_instantiated_template() {
                        debug!("Template specialization: {:?}, {:?} {:?}", ty, location, canonical_ty);
                        let complex = CompInfo::from_ty(potential_id, ty, Some(location), ctx).expect("C'mon");
                        TypeKind::Comp(complex)
                    } else {
                        match location.kind() {
                            CXCursor_CXXBaseSpecifier | CXCursor_ClassTemplate => {
                                if location.kind() == CXCursor_CXXBaseSpecifier {
                                    if location.spelling().chars().all(|c| c.is_alphanumeric() || c == '_') {
                                        return Err(parse::Error::Recurse);
                                    }
                                } else {
                                    name = Some(location.spelling());
                                }
                                let complex = CompInfo::from_ty(potential_id, ty, Some(location), ctx);
                                match complex {
                                    Ok(complex) => TypeKind::Comp(complex),
                                    Err(_) => {
                                        warn!(
                                            "Could not create complex type \
                                             from class template or base \
                                             specifier, using opaque blob"
                                        );
                                        let opaque = Opaque::from_clang_ty(ty, ctx);
                                        return Ok(parse::Resolved::New(opaque, None));
                                    },
                                }
                            },
                            CXCursor_TypeAliasTemplateDecl => {
                                debug!("TypeAliasTemplateDecl");
                                let mut inner = Err(parse::Error::Continue);
                                let mut args = vec![];
                                location.visit(|cur| {
                                    match cur.kind() {
                                        CXCursor_TypeAliasDecl => {
                                            let current = cur.cur_type();
                                            debug_assert_eq!(current.kind(), CXType_Typedef);
                                            name = Some(location.spelling());
                                            let inner_ty = cur.typedef_type().expect("Not valid Type?");
                                            inner = Ok(Item::from_ty_or_ref(inner_ty, cur, Some(potential_id), ctx));
                                        },
                                        CXCursor_TemplateTypeParameter => {
                                            let param = Item::type_param(None, cur, ctx).expect(
                                                "Item::type_param shouldn't \
                                                 ever fail if we are looking \
                                                 at a TemplateTypeParameter",
                                            );
                                            args.push(param);
                                        },
                                        _ => {},
                                    }
                                    CXChildVisit_Continue
                                });
                                let inner_type = match inner {
                                    Ok(inner) => inner,
                                    Err(..) => {
                                        warn!(
                                            "Failed to parse template alias \
                                             {:?}",
                                            location
                                        );
                                        return Err(parse::Error::Continue);
                                    },
                                };
                                TypeKind::TemplateAlias(inner_type, args)
                            },
                            CXCursor_TemplateRef => {
                                let referenced = location.referenced().unwrap();
                                let referenced_ty = referenced.cur_type();
                                debug!(
                                    "TemplateRef: location = {:?}; referenced = \
                                        {:?}; referenced_ty = {:?}",
                                    location, referenced, referenced_ty
                                );
                                return Self::from_clang_ty(potential_id, &referenced_ty, referenced, parent_id, ctx);
                            },
                            CXCursor_TypeRef => {
                                let referenced = location.referenced().unwrap();
                                let referenced_ty = referenced.cur_type();
                                let declaration = referenced_ty.declaration();
                                debug!(
                                    "TypeRef: location = {:?}; referenced = \
                                     {:?}; referenced_ty = {:?}",
                                    location, referenced, referenced_ty
                                );
                                let id = Item::from_ty_or_ref_with_id(
                                    potential_id,
                                    referenced_ty,
                                    declaration,
                                    parent_id,
                                    ctx,
                                );
                                return Ok(parse::Resolved::AlreadyResolved(id.into()));
                            },
                            CXCursor_NamespaceRef => {
                                return Err(parse::Error::Continue);
                            },
                            _ => {
                                if ty.kind() == CXType_Unexposed {
                                    warn!(
                                        "Unexposed type {:?}, recursing inside, \
                                          loc: {:?}",
                                        ty, location
                                    );
                                    return Err(parse::Error::Recurse);
                                }
                                warn!("invalid type {:?}", ty);
                                return Err(parse::Error::Continue);
                            },
                        }
                    }
                },
                CXType_Auto => {
                    if canonical_ty == *ty {
                        debug!("Couldn't find deduced type: {:?}", ty);
                        return Err(parse::Error::Continue);
                    }
                    return Self::from_clang_ty(potential_id, &canonical_ty, location, parent_id, ctx);
                },
                CXType_MemberPointer | CXType_Pointer => {
                    let mut pointee = ty.pointee_type().unwrap();
                    if *ty != canonical_ty {
                        let canonical_pointee = canonical_ty.pointee_type().unwrap();
                        if canonical_pointee.is_const() != pointee.is_const() {
                            pointee = canonical_pointee;
                        }
                    }
                    let inner = Item::from_ty_or_ref(pointee, location, None, ctx);
                    TypeKind::Pointer(inner)
                },
                CXType_BlockPointer => {
                    let pointee = ty.pointee_type().expect("Not valid Type?");
                    let inner = Item::from_ty_or_ref(pointee, location, None, ctx);
                    TypeKind::BlockPointer(inner)
                },
                CXType_RValueReference | CXType_LValueReference => {
                    let inner = Item::from_ty_or_ref(ty.pointee_type().unwrap(), location, None, ctx);
                    TypeKind::Reference(inner)
                },
                CXType_VariableArray | CXType_DependentSizedArray => {
                    let inner = Item::from_ty(ty.elem_type().as_ref().unwrap(), location, None, ctx)
                        .expect("Not able to resolve array element?");
                    TypeKind::Pointer(inner)
                },
                CXType_IncompleteArray => {
                    let inner = Item::from_ty(ty.elem_type().as_ref().unwrap(), location, None, ctx)
                        .expect("Not able to resolve array element?");
                    TypeKind::Array(inner, 0)
                },
                CXType_FunctionNoProto | CXType_FunctionProto => {
                    let signature = FnSig::from_ty(ty, &location, ctx)?;
                    TypeKind::Function(signature)
                },
                CXType_Typedef => {
                    let inner = cur.typedef_type().expect("Not valid Type?");
                    let inner_id = Item::from_ty_or_ref(inner, location, None, ctx);
                    if inner_id == potential_id {
                        warn!(
                            "Generating oqaque type instead of self-referential \
                            typedef"
                        );
                        TypeKind::Opaque
                    } else {
                        if let Some(ref mut name) = name {
                            if inner.kind() == CXType_Pointer && !ctx.opts().c_naming {
                                let pointee = inner.pointee_type().unwrap();
                                if pointee.kind() == CXType_Elaborated && pointee.declaration().spelling() == *name {
                                    *name += "_ptr";
                                }
                            }
                        }
                        TypeKind::Alias(inner_id)
                    }
                },
                CXType_Enum => {
                    let enum_ = Enum::from_ty(ty, ctx).expect("Not an enum?");
                    if !is_anonymous {
                        let pretty_name = ty.spelling();
                        if clang::is_valid_identifier(&pretty_name) {
                            name = Some(pretty_name);
                        }
                    }
                    TypeKind::Enum(enum_)
                },
                CXType_Record => {
                    let complex =
                        CompInfo::from_ty(potential_id, ty, Some(location), ctx).expect("Not a complex type?");
                    if !is_anonymous {
                        let pretty_name = ty.spelling();
                        if clang::is_valid_identifier(&pretty_name) {
                            name = Some(pretty_name);
                        }
                    }
                    TypeKind::Comp(complex)
                },
                CXType_Vector => {
                    let inner = Item::from_ty(ty.elem_type().as_ref().unwrap(), location, None, ctx)?;
                    TypeKind::Vector(inner, ty.num_elements().unwrap())
                },
                CXType_ConstantArray => {
                    let inner = Item::from_ty(ty.elem_type().as_ref().unwrap(), location, None, ctx)
                        .expect("Not able to resolve array element?");
                    TypeKind::Array(inner, ty.num_elements().unwrap())
                },
                CXType_Elaborated => {
                    return Self::from_clang_ty(potential_id, &ty.named(), location, parent_id, ctx);
                },
                CXType_Dependent => {
                    return Err(parse::Error::Continue);
                },
                _ => {
                    warn!(
                        "unsupported type: kind = {:?}; ty = {:?}; at {:?}",
                        ty.kind(),
                        ty,
                        location
                    );
                    return Err(parse::Error::Continue);
                },
            }
        };
        name = name.filter(|n| !n.is_empty());
        let is_const = ty.is_const()
            || (ty.kind() == CXType_ConstantArray && ty.elem_type().map_or(false, |element| element.is_const()));
        let ty = Type::new(name, layout, kind, is_const);
        Ok(parse::Resolved::New(ty, Some(cur.canonical())))
    }
}
impl IsOpaque for Type {
    type Extra = Item;
    fn is_opaque(&self, ctx: &Context, it: &Item) -> bool {
        match self.kind {
            TypeKind::Opaque => true,
            TypeKind::TemplateInstantiation(ref x) => x.is_opaque(ctx, it),
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
        if let Some(ref layout) = self.layout {
            writeln!(
                y,
                "<tr><td>size</td><td>{}</td></tr>
                           <tr><td>align</td><td>{}</td></tr>",
                layout.size, layout.align
            )?;
            if layout.packed {
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
            TypeKind::Pointer(inner)
            | TypeKind::Reference(inner)
            | TypeKind::Array(inner, _)
            | TypeKind::Vector(inner, _)
            | TypeKind::BlockPointer(inner)
            | TypeKind::Alias(inner)
            | TypeKind::ResolvedTypeRef(inner) => {
                tracer.visit_kind(inner.into(), EdgeKind::TypeReference);
            },
            TypeKind::TemplateAlias(inner, ref template_params) => {
                tracer.visit_kind(inner.into(), EdgeKind::TypeReference);
                for param in template_params {
                    tracer.visit_kind(param.into(), EdgeKind::TemplateParameterDefinition);
                }
            },
            TypeKind::TemplateInstantiation(ref x) => {
                x.trace(ctx, tracer, &());
            },
            TypeKind::Comp(ref x) => x.trace(ctx, tracer, it),
            TypeKind::Function(ref x) => x.trace(ctx, tracer, &()),
            TypeKind::Enum(ref x) => {
                if let Some(x) = x.repr() {
                    tracer.visit(x.into());
                }
            },
            TypeKind::UnresolvedTypeRef(_, _, Some(id)) => {
                tracer.visit(id);
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
    TemplateAlias(TypeId, Vec<TypeId>),
    Vector(TypeId, usize),
    Array(TypeId, usize),
    Function(FnSig),
    Enum(Enum),
    Pointer(TypeId),
    BlockPointer(TypeId),
    Reference(TypeId),
    TemplateInstantiation(TemplInstantiation),
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
            TypeKind::TemplateAlias(..) => "TemplateAlias",
            TypeKind::Array(..) => "Array",
            TypeKind::Vector(..) => "Vector",
            TypeKind::Function(..) => "Function",
            TypeKind::Enum(..) => "Enum",
            TypeKind::Pointer(..) => "Pointer",
            TypeKind::BlockPointer(..) => "BlockPointer",
            TypeKind::Reference(..) => "Reference",
            TypeKind::TemplateInstantiation(..) => "TemplateInstantiation",
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
            TypeKind::ResolvedTypeRef(id) => ctx.resolve_type(id).self_templ_params(ctx),
            TypeKind::Comp(ref comp) => comp.self_templ_params(ctx),
            TypeKind::TemplateAlias(_, ref args) => args.clone(),
            TypeKind::Opaque
            | TypeKind::TemplateInstantiation(..)
            | TypeKind::Void
            | TypeKind::NullPtr
            | TypeKind::Int(_)
            | TypeKind::Float(_)
            | TypeKind::Complex(_)
            | TypeKind::Array(..)
            | TypeKind::Vector(..)
            | TypeKind::Function(_)
            | TypeKind::Enum(_)
            | TypeKind::Pointer(_)
            | TypeKind::BlockPointer(_)
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
            TypeKind::ResolvedTypeRef(id) => id.as_templ_param(ctx, &()),
            _ => None,
        }
    }
}

#[test]
fn is_invalid_type_param_valid() {
    let ty = Type::new(Some("foo".into()), None, TypeKind::TypeParam, false);
    assert!(!ty.is_invalid_type_param())
}
#[test]
fn is_invalid_type_param_valid_underscore_and_numbers() {
    let ty = Type::new(Some("_foo123456789_".into()), None, TypeKind::TypeParam, false);
    assert!(!ty.is_invalid_type_param())
}
#[test]
fn is_invalid_type_param_valid_unnamed_kind() {
    let ty = Type::new(Some("foo".into()), None, TypeKind::Void, false);
    assert!(!ty.is_invalid_type_param())
}
#[test]
fn is_invalid_type_param_invalid_start() {
    let ty = Type::new(Some("1foo".into()), None, TypeKind::TypeParam, false);
    assert!(ty.is_invalid_type_param())
}
#[test]
fn is_invalid_type_param_invalid_remaing() {
    let ty = Type::new(Some("foo-".into()), None, TypeKind::TypeParam, false);
    assert!(ty.is_invalid_type_param())
}
#[test]
#[should_panic]
fn is_invalid_type_param_unnamed() {
    let ty = Type::new(None, None, TypeKind::TypeParam, false);
    assert!(ty.is_invalid_type_param())
}
#[test]
fn is_invalid_type_param_empty_name() {
    let ty = Type::new(Some("".into()), None, TypeKind::TypeParam, false);
    assert!(ty.is_invalid_type_param())
}
