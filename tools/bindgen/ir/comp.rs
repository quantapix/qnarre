use super::analysis::Sizedness;
use super::annos::Annotations;
use super::dot::DotAttrs;
use super::item::{IsOpaque, Item};
use super::templ::Params;
use super::Layout;
use super::{Context, EdgeKind, FnId, ItemId, Trace, Tracer, TypeId, VarId};
use crate::clang;
use crate::codegen::structure::{align_to, bytes_from_bits_pow2};
use crate::codegen::utils::variation;
use crate::ir::derive::CanDeriveCopy;
use crate::parse;
use crate::HashMap;
use peeking_take_while::PeekableExt;
use std::cmp;
use std::io;
use std::mem;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CompKind {
    Struct,
    Union,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MethKind {
    Constr,
    Destr,
    VirtDestr { pure: bool },
    Static,
    Normal,
    Virt { pure: bool },
}
impl MethKind {
    pub fn is_destr(&self) -> bool {
        matches!(*self, MethKind::Destr | MethKind::VirtDestr { .. })
    }
    pub fn is_pure_virt(&self) -> bool {
        match *self {
            MethKind::Virt { pure } | MethKind::VirtDestr { pure } => pure,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct Method {
    kind: MethKind,
    sig: FnId,
    is_const: bool,
}
impl Method {
    pub fn new(kind: MethKind, sig: FnId, is_const: bool) -> Self {
        Method { kind, sig, is_const }
    }
    pub fn kind(&self) -> MethKind {
        self.kind
    }
    pub fn is_constr(&self) -> bool {
        self.kind == MethKind::Constr
    }
    pub fn is_virt(&self) -> bool {
        matches!(self.kind, MethKind::Virt { .. } | MethKind::VirtDestr { .. })
    }
    pub fn is_static(&self) -> bool {
        self.kind == MethKind::Static
    }
    pub fn sig(&self) -> FnId {
        self.sig
    }
    pub fn is_const(&self) -> bool {
        self.is_const
    }
}

pub trait FieldMeths {
    fn name(&self) -> Option<&str>;
    fn ty(&self) -> TypeId;
    fn comment(&self) -> Option<&str>;
    fn bitfield_width(&self) -> Option<u32>;
    fn is_public(&self) -> bool;
    fn annos(&self) -> &Annotations;
    fn offset(&self) -> Option<usize>;
}

#[derive(Debug)]
pub enum Field {
    Data(Data),
}
impl Field {
    pub fn layout(&self, ctx: &Context) -> Option<Layout> {
        match *self {
            Field::Data(ref x) => ctx.resolve_type(x.ty).layout(ctx),
        }
    }
}
impl Trace for Field {
    type Extra = ();
    fn trace<T>(&self, _: &Context, tracer: &mut T, _: &())
    where
        T: Tracer,
    {
        match *self {
            Field::Data(ref x) => {
                tracer.visit_kind(x.ty.into(), EdgeKind::Field);
            },
        }
    }
}
impl DotAttrs for Field {
    fn dot_attrs<W>(&self, ctx: &Context, y: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        match *self {
            Field::Data(ref x) => x.dot_attrs(ctx, y),
        }
    }
}

#[derive(Debug)]
enum CompFields {
    Before(Vec<RawField>),
    After {
        fields: Vec<Field>,
        has_bitfield_units: bool,
    },
    Error,
}
impl CompFields {
    fn append_raw_field(&mut self, raw: RawField) {
        match *self {
            CompFields::Before(ref mut xs) => {
                xs.push(raw);
            },
            _ => {
                panic!("Must not append new fields after computing bitfield allocation units");
            },
        }
    }
    fn compute_bitfield_units(&mut self, ctx: &Context, packed: bool) {
        let raws = match *self {
            CompFields::Before(ref mut xs) => mem::take(xs),
            _ => {
                panic!("Already computed bitfield units");
            },
        };
        let y = raw_fields_to_fields_and_bitfield_units(ctx, raws, packed);
        match y {
            Ok((fields, has_bitfield_units)) => {
                *self = CompFields::After {
                    fields,
                    has_bitfield_units,
                };
            },
            Err(()) => {
                *self = CompFields::Error;
            },
        }
    }
    fn deanonymize_fields(&mut self, ctx: &Context, methods: &[Method]) {
        let fields = match *self {
            CompFields::After { ref mut fields, .. } => fields,
            CompFields::Error => return,
            CompFields::Before(_) => {
                panic!("Not yet computed bitfield units.");
            },
        };
        fn has_method(methods: &[Method], ctx: &Context, name: &str) -> bool {
            methods.iter().any(|x| {
                let x = ctx.resolve_fn(x.sig()).name();
                x == name || ctx.rust_mangle(x) == name
            })
        }
        struct AccessorNamesPair {
            getter: String,
            setter: String,
        }
        let mut accessor_names: HashMap<String, AccessorNamesPair> = fields
            .iter()
            .flat_map(|x| match *x {
                Field::Data(_) => &[],
            })
            .filter_map(|x| x.name())
            .map(|x| {
                let x = x.to_string();
                let getter = {
                    let mut getter = ctx.rust_mangle(&x).to_string();
                    if has_method(methods, ctx, &getter) {
                        getter.push_str("_bindgen_bitfield");
                    }
                    getter
                };
                let setter = {
                    let setter = format!("set_{}", x);
                    let mut setter = ctx.rust_mangle(&setter).to_string();
                    if has_method(methods, ctx, &setter) {
                        setter.push_str("_bindgen_bitfield");
                    }
                    setter
                };
                (x, AccessorNamesPair { getter, setter })
            })
            .collect();
        let mut anon_field_counter = 0;
        for field in fields.iter_mut() {
            match *field {
                Field::Data(Data { ref mut name, .. }) => {
                    if name.is_some() {
                        continue;
                    }
                    anon_field_counter += 1;
                    *name = Some(format!("{}{}", ctx.opts().anon_fields_prefix, anon_field_counter));
                },
            }
        }
    }
}
impl Default for CompFields {
    fn default() -> CompFields {
        CompFields::Before(vec![])
    }
}
impl Trace for CompFields {
    type Extra = ();
    fn trace<T>(&self, ctx: &Context, tracer: &mut T, _: &())
    where
        T: Tracer,
    {
        match *self {
            CompFields::Error => {},
            CompFields::Before(ref xs) => {
                for x in xs {
                    tracer.visit_kind(x.ty().into(), EdgeKind::Field);
                }
            },
            CompFields::After { ref fields, .. } => {
                for x in fields {
                    x.trace(ctx, tracer, &());
                }
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct Data {
    name: Option<String>,
    ty: TypeId,
    comment: Option<String>,
    annos: Annotations,
    bitfield_width: Option<u32>,
    public: bool,
    offset: Option<usize>,
}
impl FieldMeths for Data {
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    fn ty(&self) -> TypeId {
        self.ty
    }
    fn comment(&self) -> Option<&str> {
        self.comment.as_deref()
    }
    fn bitfield_width(&self) -> Option<u32> {
        self.bitfield_width
    }
    fn is_public(&self) -> bool {
        self.public
    }
    fn annos(&self) -> &Annotations {
        &self.annos
    }
    fn offset(&self) -> Option<usize> {
        self.offset
    }
}
impl DotAttrs for Data {
    fn dot_attrs<W>(&self, _: &Context, y: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(
            y,
            "<tr><td>{}</td><td>{:?}</td></tr>",
            self.name().unwrap_or("(anonymous)"),
            self.ty()
        )
    }
}

#[derive(Debug)]
struct RawField(Data);
impl RawField {
    fn new(
        name: Option<String>,
        ty: TypeId,
        comment: Option<String>,
        annos: Option<Annotations>,
        bitfield_width: Option<u32>,
        public: bool,
        offset: Option<usize>,
    ) -> RawField {
        RawField(Data {
            name,
            ty,
            comment,
            annos: annos.unwrap_or_default(),
            bitfield_width,
            public,
            offset,
        })
    }
}
impl FieldMeths for RawField {
    fn name(&self) -> Option<&str> {
        self.0.name()
    }
    fn ty(&self) -> TypeId {
        self.0.ty()
    }
    fn comment(&self) -> Option<&str> {
        self.0.comment()
    }
    fn bitfield_width(&self) -> Option<u32> {
        self.0.bitfield_width()
    }
    fn is_public(&self) -> bool {
        self.0.is_public()
    }
    fn annos(&self) -> &Annotations {
        self.0.annos()
    }
    fn offset(&self) -> Option<usize> {
        self.0.offset()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BaseKind {
    Normal,
    Virt,
}

#[derive(Clone, Debug)]
pub struct Base {
    pub ty: TypeId,
    pub kind: BaseKind,
    pub field_name: String,
    pub is_pub: bool,
}
impl Base {
    pub fn is_virt(&self) -> bool {
        self.kind == BaseKind::Virt
    }
    pub fn requires_storage(&self, ctx: &Context) -> bool {
        if self.is_virt() {
            return false;
        }
        if self.ty.is_zero_sized(ctx) {
            return false;
        }
        true
    }
    pub fn is_public(&self) -> bool {
        self.is_pub
    }
}

#[derive(Debug)]
pub struct Comp {
    bases: Vec<Base>,
    constrs: Vec<FnId>,
    destr: Option<(MethKind, FnId)>,
    fields: CompFields,
    found_unknown_attr: bool,
    has_destr: bool,
    has_non_type_params: bool,
    has_nonempty_base: bool,
    has_own_virt_method: bool,
    has_unevaluable_width: bool,
    inner_types: Vec<TypeId>,
    inner_vars: Vec<VarId>,
    is_fwd_decl: bool,
    kind: CompKind,
    methods: Vec<Method>,
    packed_attr: bool,
    templ_params: Vec<TypeId>,
}
impl Comp {
    pub fn new(kind: CompKind) -> Self {
        Comp {
            bases: vec![],
            constrs: vec![],
            destr: None,
            fields: CompFields::default(),
            found_unknown_attr: false,
            has_destr: false,
            has_non_type_params: false,
            has_nonempty_base: false,
            has_own_virt_method: false,
            has_unevaluable_width: false,
            inner_types: vec![],
            inner_vars: vec![],
            is_fwd_decl: false,
            kind,
            methods: vec![],
            packed_attr: false,
            templ_params: vec![],
        }
    }
    pub fn layout(&self, ctx: &Context) -> Option<Layout> {
        if self.kind == CompKind::Struct {
            return None;
        }
        if self.is_fwd_decl() {
            return None;
        }
        if !self.has_fields() {
            return None;
        }
        let mut max_size = 0;
        let mut max_align = 1;
        self.each_known_layout(ctx, |x| {
            max_size = cmp::max(max_size, x.size);
            max_align = cmp::max(max_align, x.align);
        });
        Some(Layout::new(max_size, max_align))
    }
    pub fn fields(&self) -> &[Field] {
        match self.fields {
            CompFields::Error => &[],
            CompFields::After { ref fields, .. } => fields,
            CompFields::Before(..) => {
                panic!("Should always have computed bitfield units first");
            },
        }
    }
    fn has_fields(&self) -> bool {
        match self.fields {
            CompFields::Error => false,
            CompFields::After { ref fields, .. } => !fields.is_empty(),
            CompFields::Before(ref raw_fields) => !raw_fields.is_empty(),
        }
    }
    fn each_known_layout(&self, ctx: &Context, mut cb: impl FnMut(Layout)) {
        match self.fields {
            CompFields::Error => {},
            CompFields::After { ref fields, .. } => {
                for field in fields.iter() {
                    if let Some(layout) = field.layout(ctx) {
                        cb(layout);
                    }
                }
            },
            CompFields::Before(ref raw_fields) => {
                for field in raw_fields.iter() {
                    let field_ty = ctx.resolve_type(field.0.ty);
                    if let Some(layout) = field_ty.layout(ctx) {
                        cb(layout);
                    }
                }
            },
        }
    }
    fn has_bitfields(&self) -> bool {
        match self.fields {
            CompFields::Error => false,
            CompFields::After { has_bitfield_units, .. } => has_bitfield_units,
            CompFields::Before(_) => {
                panic!("Should always have computed bitfield units first");
            },
        }
    }
    pub fn has_too_large_bitfield_unit(&self) -> bool {
        if !self.has_bitfields() {
            return false;
        }
        self.fields().iter().any(|x| match *x {
            Field::Data(..) => false,
        })
    }
    pub fn has_non_type_params(&self) -> bool {
        self.has_non_type_params
    }
    pub fn has_own_virt_method(&self) -> bool {
        self.has_own_virt_method
    }
    pub fn has_own_destr(&self) -> bool {
        self.has_destr
    }
    pub fn methods(&self) -> &[Method] {
        &self.methods
    }
    pub fn constrs(&self) -> &[FnId] {
        &self.constrs
    }
    pub fn destr(&self) -> Option<(MethKind, FnId)> {
        self.destr
    }
    pub fn kind(&self) -> CompKind {
        self.kind
    }
    pub fn is_union(&self) -> bool {
        self.kind() == CompKind::Union
    }
    pub fn bases(&self) -> &[Base] {
        &self.bases
    }
    pub fn from_ty(
        potential_id: ItemId,
        ty: &clang::Type,
        cur: Option<clang::Cursor>,
        ctx: &mut Context,
    ) -> Result<Self, parse::Error> {
        use clang_lib::*;
        assert!(ty.templ_args().is_none(), "We handle template instantiations elsewhere");
        let mut cursor = ty.decl();
        let mut kind = Self::kind_from_cursor(&cursor);
        if kind.is_err() {
            if let Some(x) = cur {
                kind = Self::kind_from_cursor(&x);
                cursor = x;
            }
        }
        let kind = kind?;
        debug!("CompInfo::from_ty({:?}, {:?})", kind, cursor);
        let mut ci = Comp::new(kind);
        ci.is_fwd_decl = cur.map_or(true, |x| match x.kind() {
            CXCursor_ParmDecl => true,
            CXCursor_StructDecl | CXCursor_UnionDecl | CXCursor_ClassDecl => !x.is_definition(),
            _ => false,
        });
        let mut maybe_anonymous_struct_field = None;
        cursor.visit(|cur2| {
            if cur2.kind() != CXCursor_FieldDecl {
                if let Some((ty, clang_ty, public, offset)) = maybe_anonymous_struct_field.take() {
                    if cur2.kind() == CXCursor_TypedefDecl && cur2.typedef_type().unwrap().canon_type() == clang_ty {
                    } else {
                        let field = RawField::new(None, ty, None, None, None, public, offset);
                        ci.fields.append_raw_field(field);
                    }
                }
            }
            match cur2.kind() {
                CXCursor_FieldDecl => {
                    if let Some((ty, clang_ty, public, offset)) = maybe_anonymous_struct_field.take() {
                        let mut used = false;
                        cur2.visit(|x| {
                            if x.cur_type() == clang_ty {
                                used = true;
                            }
                            CXChildVisit_Continue
                        });
                        if !used {
                            let field = RawField::new(None, ty, None, None, None, public, offset);
                            ci.fields.append_raw_field(field);
                        }
                    }
                    let bit_width = if cur2.is_bit_field() {
                        let width = cur2.bit_width();
                        if width.is_none() {
                            ci.has_unevaluable_width = true;
                            return CXChildVisit_Break;
                        }
                        width
                    } else {
                        None
                    };
                    let field_type = Item::from_ty_or_ref(cur2.cur_type(), cur2, Some(potential_id), ctx);
                    let comment = cur2.raw_comment();
                    let annos = Annotations::new(&cur2);
                    let name = cur2.spelling();
                    let is_public = cur2.public_accessible();
                    let offset = cur2.offset_of_field().ok();
                    assert!(!name.is_empty() || bit_width.is_some(), "Empty field name?");
                    let name = if name.is_empty() { None } else { Some(name) };
                    let field = RawField::new(name, field_type, comment, annos, bit_width, is_public, offset);
                    ci.fields.append_raw_field(field);
                    cur2.visit(|x| {
                        if x.kind() == CXCursor_UnexposedAttr {
                            ci.found_unknown_attr = true;
                        }
                        CXChildVisit_Continue
                    });
                },
                CXCursor_UnexposedAttr => {
                    ci.found_unknown_attr = true;
                },
                CXCursor_EnumDecl
                | CXCursor_TypeAliasDecl
                | CXCursor_TypeAliasTemplateDecl
                | CXCursor_TypedefDecl
                | CXCursor_StructDecl
                | CXCursor_UnionDecl
                | CXCursor_ClassTemplate
                | CXCursor_ClassDecl => {
                    let is_inner_struct = cur2.semantic_parent() == cursor || cur2.is_definition();
                    if !is_inner_struct {
                        return CXChildVisit_Continue;
                    }
                    let inner = Item::parse(cur2, Some(potential_id), ctx).expect("Inner ClassDecl");
                    if ctx.resolve_item_fallible(inner).is_some() {
                        let inner = inner.expect_type_id(ctx);
                        ci.inner_types.push(inner);
                        if cur2.is_anonymous() && cur2.kind() != CXCursor_EnumDecl {
                            let ty = cur2.cur_type();
                            let public = cur2.public_accessible();
                            let offset = cur2.offset_of_field().ok();
                            maybe_anonymous_struct_field = Some((inner, ty, public, offset));
                        }
                    }
                },
                CXCursor_PackedAttr => {
                    ci.packed_attr = true;
                },
                CXCursor_TemplateTypeParameter => {
                    let param = Item::type_param(None, cur2, ctx).expect(
                        "Item::type_param should't fail when pointing \
                         at a TemplateTypeParameter",
                    );
                    ci.templ_params.push(param);
                },
                CXCursor_CXXBaseSpecifier => {
                    let is_virt_base = cur2.is_virt_base();
                    ci.has_own_virt_method |= is_virt_base;
                    let kind = if is_virt_base { BaseKind::Virt } else { BaseKind::Normal };
                    let field_name = match ci.bases.len() {
                        0 => "_base".into(),
                        n => format!("_base_{}", n),
                    };
                    let type_id = Item::from_ty_or_ref(cur2.cur_type(), cur2, None, ctx);
                    ci.bases.push(Base {
                        ty: type_id,
                        kind,
                        field_name,
                        is_pub: cur2.access_specifier() == clang_lib::CX_CXXPublic,
                    });
                },
                CXCursor_Constructor | CXCursor_Destructor | CXCursor_CXXMethod => {
                    let is_virt = cur2.method_is_virt();
                    let is_static = cur2.method_is_static();
                    debug_assert!(!(is_static && is_virt), "How?");
                    ci.has_destr |= cur2.kind() == CXCursor_Destructor;
                    ci.has_own_virt_method |= is_virt;
                    if !ci.templ_params.is_empty() {
                        return CXChildVisit_Continue;
                    }
                    let signature = match Item::parse(cur2, Some(potential_id), ctx) {
                        Ok(item) if ctx.resolve_item(item).kind().is_fn() => item,
                        _ => return CXChildVisit_Continue,
                    };
                    let signature = signature.expect_fn_id(ctx);
                    match cur2.kind() {
                        CXCursor_Constructor => {
                            ci.constrs.push(signature);
                        },
                        CXCursor_Destructor => {
                            let kind = if is_virt {
                                MethKind::VirtDestr {
                                    pure: cur2.method_is_pure_virt(),
                                }
                            } else {
                                MethKind::Destr
                            };
                            ci.destr = Some((kind, signature));
                        },
                        CXCursor_CXXMethod => {
                            let is_const = cur2.method_is_const();
                            let method_kind = if is_static {
                                MethKind::Static
                            } else if is_virt {
                                MethKind::Virt {
                                    pure: cur2.method_is_pure_virt(),
                                }
                            } else {
                                MethKind::Normal
                            };
                            let method = Method::new(method_kind, signature, is_const);
                            ci.methods.push(method);
                        },
                        _ => unreachable!("How can we see this here?"),
                    }
                },
                CXCursor_NonTypeTemplateParameter => {
                    ci.has_non_type_params = true;
                },
                CXCursor_VarDecl => {
                    let linkage = cur2.linkage();
                    if linkage != CXLinkage_External && linkage != CXLinkage_UniqueExternal {
                        return CXChildVisit_Continue;
                    }
                    let visibility = cur2.visibility();
                    if visibility != CXVisibility_Default {
                        return CXChildVisit_Continue;
                    }
                    if let Ok(item) = Item::parse(cur2, Some(potential_id), ctx) {
                        ci.inner_vars.push(item.as_var_id_unchecked());
                    }
                },
                CXCursor_CXXAccessSpecifier
                | CXCursor_CXXFinalAttr
                | CXCursor_FunctionTemplate
                | CXCursor_ConversionFunction => {},
                _ => {
                    warn!(
                        "unhandled comp member `{}` (kind {:?}) in `{}` ({})",
                        cur2.spelling(),
                        clang::kind_to_str(cur2.kind()),
                        cursor.spelling(),
                        cur2.location()
                    );
                },
            }
            CXChildVisit_Continue
        });
        if let Some((ty, _, public, offset)) = maybe_anonymous_struct_field {
            let field = RawField::new(None, ty, None, None, None, public, offset);
            ci.fields.append_raw_field(field);
        }
        Ok(ci)
    }
    fn kind_from_cursor(cur: &clang::Cursor) -> Result<CompKind, parse::Error> {
        use clang_lib::*;
        Ok(match cur.kind() {
            CXCursor_UnionDecl => CompKind::Union,
            CXCursor_ClassDecl | CXCursor_StructDecl => CompKind::Struct,
            CXCursor_CXXBaseSpecifier | CXCursor_ClassTemplatePartialSpecialization | CXCursor_ClassTemplate => {
                match cur.templ_kind() {
                    CXCursor_UnionDecl => CompKind::Union,
                    _ => CompKind::Struct,
                }
            },
            _ => {
                warn!("Unknown kind for comp type: {:?}", cur);
                return Err(parse::Error::Continue);
            },
        })
    }
    pub fn inner_types(&self) -> &[TypeId] {
        &self.inner_types
    }
    pub fn inner_vars(&self) -> &[VarId] {
        &self.inner_vars
    }
    pub fn found_unknown_attr(&self) -> bool {
        self.found_unknown_attr
    }
    pub fn is_packed(&self, ctx: &Context, layout: Option<&Layout>) -> bool {
        if self.packed_attr {
            return true;
        }
        if let Some(parent_layout) = layout {
            let mut packed = false;
            self.each_known_layout(ctx, |x| {
                packed = packed || x.align > parent_layout.align;
            });
            if packed {
                info!("Found a struct that was defined within `#pragma packed(...)`");
                return true;
            }
            if self.has_own_virt_method && parent_layout.align == 1 {
                return true;
            }
        }
        false
    }
    pub fn is_fwd_decl(&self) -> bool {
        self.is_fwd_decl
    }
    pub fn compute_bitfield_units(&mut self, ctx: &Context, layout: Option<&Layout>) {
        let packed = self.is_packed(ctx, layout);
        self.fields.compute_bitfield_units(ctx, packed)
    }
    pub fn deanonymize_fields(&mut self, ctx: &Context) {
        self.fields.deanonymize_fields(ctx, &self.methods);
    }
    pub fn is_rust_union(&self, ctx: &Context, layout: Option<&Layout>, name: &str) -> (bool, bool) {
        if !self.is_union() {
            return (false, false);
        }
        if !ctx.opts().untagged_union {
            return (false, false);
        }
        if self.is_fwd_decl() {
            return (false, false);
        }
        let union_style = if ctx.opts().bindgen_wrapper_union.matches(name) {
            variation::NonCopyUnion::BindgenWrapper
        } else if ctx.opts().manually_drop_union.matches(name) {
            variation::NonCopyUnion::ManuallyDrop
        } else {
            ctx.opts().default_non_copy_union_style
        };
        let all_can_copy = self.fields().iter().all(|f| match *f {
            Field::Data(ref field_data) => field_data.ty().can_derive_copy(ctx),
        });
        if !all_can_copy && union_style == variation::NonCopyUnion::BindgenWrapper {
            return (false, false);
        }
        if layout.map_or(false, |l| l.size == 0) {
            return (false, false);
        }
        (true, all_can_copy)
    }
}
impl DotAttrs for Comp {
    fn dot_attrs<W>(&self, ctx: &Context, y: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(y, "<tr><td>CompKind</td><td>{:?}</td></tr>", self.kind)?;
        if self.has_own_virt_method {
            writeln!(y, "<tr><td>has_vtable</td><td>true</td></tr>")?;
        }
        if self.has_destr {
            writeln!(y, "<tr><td>has_destructor</td><td>true</td></tr>")?;
        }
        if self.has_nonempty_base {
            writeln!(y, "<tr><td>has_nonempty_base</td><td>true</td></tr>")?;
        }
        if self.has_non_type_params {
            writeln!(y, "<tr><td>has_non_type_params</td><td>true</td></tr>")?;
        }
        if self.packed_attr {
            writeln!(y, "<tr><td>packed_attr</td><td>true</td></tr>")?;
        }
        if self.is_fwd_decl {
            writeln!(y, "<tr><td>is_forward_declaration</td><td>true</td></tr>")?;
        }
        if !self.fields().is_empty() {
            writeln!(y, r#"<tr><td>fields</td><td><table border="0">"#)?;
            for field in self.fields() {
                field.dot_attrs(ctx, y)?;
            }
            writeln!(y, "</table></td></tr>")?;
        }
        Ok(())
    }
}
impl IsOpaque for Comp {
    type Extra = Option<Layout>;
    fn is_opaque(&self, ctx: &Context, layout: &Option<Layout>) -> bool {
        if self.has_non_type_params || self.has_unevaluable_width {
            return true;
        }
        if let CompFields::Error = self.fields {
            return true;
        }
        if self.fields().iter().any(|f| match *f {
            Field::Data(_) => false,
        }) {
            return true;
        }
        false
    }
}
impl Params for Comp {
    fn self_templ_params(&self, _ctx: &Context) -> Vec<TypeId> {
        self.templ_params.clone()
    }
}
impl Trace for Comp {
    type Extra = Item;
    fn trace<T>(&self, ctx: &Context, tracer: &mut T, it: &Item)
    where
        T: Tracer,
    {
        for p in it.all_templ_params(ctx) {
            tracer.visit_kind(p.into(), EdgeKind::TemplParamDef);
        }
        for ty in self.inner_types() {
            tracer.visit_kind(ty.into(), EdgeKind::InnerType);
        }
        for &var in self.inner_vars() {
            tracer.visit_kind(var.into(), EdgeKind::InnerVar);
        }
        for method in self.methods() {
            tracer.visit_kind(method.sig.into(), EdgeKind::Method);
        }
        if let Some((_kind, signature)) = self.destr() {
            tracer.visit_kind(signature.into(), EdgeKind::Destructor);
        }
        for ctor in self.constrs() {
            tracer.visit_kind(ctor.into(), EdgeKind::Constructor);
        }
        if it.is_opaque(ctx, &()) {
            return;
        }
        for base in self.bases() {
            tracer.visit_kind(base.ty.into(), EdgeKind::BaseMember);
        }
        self.fields.trace(ctx, tracer, &());
    }
}

fn raw_fields_to_fields_and_bitfield_units<I>(
    ctx: &Context,
    raw_fields: I,
    packed: bool,
) -> Result<(Vec<Field>, bool), ()>
where
    I: IntoIterator<Item = RawField>,
{
    let mut raws = raw_fields.into_iter().fuse().peekable();
    let mut fields = vec![];
    let mut bitfield_unit_count = 0;
    loop {
        {
            let non_bfs = raws
                .by_ref()
                .peeking_take_while(|x| x.bitfield_width().is_none())
                .map(|x| Field::Data(x.0));
            fields.extend(non_bfs);
        }
        let mut bfs = raws
            .by_ref()
            .peeking_take_while(|x| x.bitfield_width().is_some())
            .peekable();
        if bfs.peek().is_none() {
            break;
        }
        bitfields_to_allocation_units(ctx, &mut bitfield_unit_count, &mut fields, bfs, packed)?;
    }
    assert!(raws.next().is_none());
    Ok((fields, bitfield_unit_count != 0))
}

fn bitfields_to_allocation_units<E, I>(
    ctx: &Context,
    bitfield_unit_count: &mut usize,
    fields: &mut E,
    raw_bfs: I,
    packed: bool,
) -> Result<(), ()>
where
    E: Extend<Field>,
    I: IntoIterator<Item = RawField>,
{
    assert!(ctx.collected_typerefs());
    fn flush_allocation_unit<E>(
        fields: &mut E,
        bitfield_unit_count: &mut usize,
        unit_size_in_bits: usize,
        unit_align_in_bits: usize,
        bitfields: Vec<Bitfield>,
        packed: bool,
    ) where
        E: Extend<Field>,
    {
        *bitfield_unit_count += 1;
        let align = if packed {
            1
        } else {
            bytes_from_bits_pow2(unit_align_in_bits)
        };
        let size = align_to(unit_size_in_bits, 8) / 8;
        let layout = Layout::new(size, align);
        fields.extend(Some(Field::Bitfields(BitfieldUnit {
            nth: *bitfield_unit_count,
            layout,
            bitfields,
        })));
    }
    let mut max_align = 0;
    let mut unfilled_bits_in_unit = 0;
    let mut unit_size_in_bits = 0;
    let mut unit_align = 0;
    let mut bitfields_in_unit = vec![];
    const is_ms_struct: bool = false;
    for bitfield in raw_bfs {
        let bitfield_width = bitfield.bitfield_width().unwrap() as usize;
        let bitfield_layout = ctx.resolve_type(bitfield.ty()).layout(ctx).ok_or(())?;
        let bitfield_size = bitfield_layout.size;
        let bitfield_align = bitfield_layout.align;
        let mut offset = unit_size_in_bits;
        if !packed {
            if is_ms_struct {
                if unit_size_in_bits != 0 && (bitfield_width == 0 || bitfield_width > unfilled_bits_in_unit) {
                    unit_size_in_bits = align_to(unit_size_in_bits, unit_align * 8);
                    flush_allocation_unit(
                        fields,
                        bitfield_unit_count,
                        unit_size_in_bits,
                        unit_align,
                        mem::take(&mut bitfields_in_unit),
                        packed,
                    );
                    offset = 0;
                    unit_align = 0;
                }
            } else if offset != 0
                && (bitfield_width == 0 || (offset & (bitfield_align * 8 - 1)) + bitfield_width > bitfield_size * 8)
            {
                offset = align_to(offset, bitfield_align * 8);
            }
        }
        if bitfield.name().is_some() {
            max_align = cmp::max(max_align, bitfield_align);
            unit_align = cmp::max(unit_align, bitfield_width);
        }
        bitfields_in_unit.push(Bitfield::new(offset, bitfield));
        unit_size_in_bits = offset + bitfield_width;
        let data_size = align_to(unit_size_in_bits, bitfield_align * 8);
        unfilled_bits_in_unit = data_size - unit_size_in_bits;
    }
    if unit_size_in_bits != 0 {
        flush_allocation_unit(
            fields,
            bitfield_unit_count,
            unit_size_in_bits,
            unit_align,
            bitfields_in_unit,
            packed,
        );
    }
    Ok(())
}
