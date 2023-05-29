use super::analysis::Sizedness;
use super::annotations::Annotations;
use super::dot::DotAttrs;
use super::item::{IsOpaque, Item};
use super::layout::Layout;
use super::template::TemplParams;
use super::traversal::{EdgeKind, Trace, Tracer};
use super::RUST_DERIVE_IN_ARRAY_LIMIT;
use super::{Context, FnId, ItemId, TypeId, VarId};
use crate::clang;
use crate::codegen::struct_layout::{align_to, bytes_from_bits_pow2};
use crate::ir::derive::CanDeriveCopy;
use crate::parse;
use crate::HashMap;
use crate::NonCopyUnionStyle;
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
pub enum MethodKind {
    Constructor,
    Destructor,
    VirtualDestructor { pure_virtual: bool },
    Static,
    Normal,
    Virtual { pure_virtual: bool },
}
impl MethodKind {
    pub fn is_destructor(&self) -> bool {
        matches!(*self, MethodKind::Destructor | MethodKind::VirtualDestructor { .. })
    }
    pub fn is_pure_virtual(&self) -> bool {
        match *self {
            MethodKind::Virtual { pure_virtual } | MethodKind::VirtualDestructor { pure_virtual } => pure_virtual,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct Method {
    kind: MethodKind,
    sig: FnId,
    is_const: bool,
}
impl Method {
    pub fn new(kind: MethodKind, sig: FnId, is_const: bool) -> Self {
        Method { kind, sig, is_const }
    }
    pub fn kind(&self) -> MethodKind {
        self.kind
    }
    pub fn is_constructor(&self) -> bool {
        self.kind == MethodKind::Constructor
    }
    pub fn is_virtual(&self) -> bool {
        matches!(
            self.kind,
            MethodKind::Virtual { .. } | MethodKind::VirtualDestructor { .. }
        )
    }
    pub fn is_static(&self) -> bool {
        self.kind == MethodKind::Static
    }
    pub fn sig(&self) -> FnId {
        self.sig
    }
    pub fn is_const(&self) -> bool {
        self.is_const
    }
}

pub trait FieldMethods {
    fn name(&self) -> Option<&str>;
    fn ty(&self) -> TypeId;
    fn comment(&self) -> Option<&str>;
    fn bitfield_width(&self) -> Option<u32>;
    fn is_public(&self) -> bool;
    fn annotations(&self) -> &Annotations;
    fn offset(&self) -> Option<usize>;
}

#[derive(Debug)]
pub struct BitfieldUnit {
    nth: usize,
    layout: Layout,
    bitfields: Vec<Bitfield>,
}
impl BitfieldUnit {
    pub fn nth(&self) -> usize {
        self.nth
    }
    pub fn layout(&self) -> Layout {
        self.layout
    }
    pub fn bitfields(&self) -> &[Bitfield] {
        &self.bitfields
    }
}

#[derive(Debug)]
pub enum Field {
    DataMember(FieldData),
    Bitfields(BitfieldUnit),
}
impl Field {
    pub fn layout(&self, ctx: &Context) -> Option<Layout> {
        match *self {
            Field::Bitfields(BitfieldUnit { layout, .. }) => Some(layout),
            Field::DataMember(ref x) => ctx.resolve_type(x.ty).layout(ctx),
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
            Field::DataMember(ref x) => {
                tracer.visit_kind(x.ty.into(), EdgeKind::Field);
            },
            Field::Bitfields(BitfieldUnit { ref bitfields, .. }) => {
                for x in bitfields {
                    tracer.visit_kind(x.ty().into(), EdgeKind::Field);
                }
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
            Field::DataMember(ref x) => x.dot_attrs(ctx, y),
            Field::Bitfields(BitfieldUnit {
                layout, ref bitfields, ..
            }) => {
                writeln!(
                    y,
                    r#"<tr>
                              <td>bitfield unit</td>
                              <td>
                                <table border="0">
                                  <tr>
                                    <td>unit.size</td><td>{}</td>
                                  </tr>
                                  <tr>
                                    <td>unit.align</td><td>{}</td>
                                  </tr>
                         "#,
                    layout.size, layout.align
                )?;
                for x in bitfields {
                    x.dot_attrs(ctx, y)?;
                }
                writeln!(y, "</table></td></tr>")
            },
        }
    }
}

#[derive(Debug)]
pub struct Bitfield {
    offset_into_unit: usize,
    data: FieldData,
    getter: Option<String>,
    setter: Option<String>,
}
impl Bitfield {
    fn new(offset_into_unit: usize, raw: RawField) -> Bitfield {
        assert!(raw.bitfield_width().is_some());
        Bitfield {
            offset_into_unit,
            data: raw.0,
            getter: None,
            setter: None,
        }
    }
    pub fn offset_into_unit(&self) -> usize {
        self.offset_into_unit
    }
    pub fn width(&self) -> u32 {
        self.data.bitfield_width().unwrap()
    }
    pub fn getter(&self) -> &str {
        assert!(self.name().is_some());
        self.getter.as_ref().expect(
            "`Bitfield::getter_name` should only be called after\
             assigning bitfield accessor names",
        )
    }
    pub fn setter(&self) -> &str {
        assert!(self.name().is_some());
        self.setter.as_ref().expect(
            "`Bitfield::setter_name` should only be called\
             after assigning bitfield accessor names",
        )
    }
}
impl FieldMethods for Bitfield {
    fn name(&self) -> Option<&str> {
        self.data.name()
    }
    fn ty(&self) -> TypeId {
        self.data.ty()
    }
    fn comment(&self) -> Option<&str> {
        self.data.comment()
    }
    fn bitfield_width(&self) -> Option<u32> {
        self.data.bitfield_width()
    }
    fn is_public(&self) -> bool {
        self.data.is_public()
    }
    fn annotations(&self) -> &Annotations {
        self.data.annotations()
    }
    fn offset(&self) -> Option<usize> {
        self.data.offset()
    }
}
impl DotAttrs for Bitfield {
    fn dot_attrs<W>(&self, _: &Context, y: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(
            y,
            "<tr><td>{} : {}</td><td>{:?}</td></tr>",
            self.name().unwrap_or("(anonymous)"),
            self.width(),
            self.ty()
        )
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
            methods.iter().any(|method| {
                let method_name = ctx.resolve_func(method.sig()).name();
                method_name == name || ctx.rust_mangle(method_name) == name
            })
        }
        struct AccessorNamesPair {
            getter: String,
            setter: String,
        }
        let mut accessor_names: HashMap<String, AccessorNamesPair> = fields
            .iter()
            .flat_map(|field| match *field {
                Field::Bitfields(ref bu) => &*bu.bitfields,
                Field::DataMember(_) => &[],
            })
            .filter_map(|x| x.name())
            .map(|bitfield_name| {
                let bitfield_name = bitfield_name.to_string();
                let getter = {
                    let mut getter = ctx.rust_mangle(&bitfield_name).to_string();
                    if has_method(methods, ctx, &getter) {
                        getter.push_str("_bindgen_bitfield");
                    }
                    getter
                };
                let setter = {
                    let setter = format!("set_{}", bitfield_name);
                    let mut setter = ctx.rust_mangle(&setter).to_string();
                    if has_method(methods, ctx, &setter) {
                        setter.push_str("_bindgen_bitfield");
                    }
                    setter
                };
                (bitfield_name, AccessorNamesPair { getter, setter })
            })
            .collect();
        let mut anon_field_counter = 0;
        for field in fields.iter_mut() {
            match *field {
                Field::DataMember(FieldData { ref mut name, .. }) => {
                    if name.is_some() {
                        continue;
                    }
                    anon_field_counter += 1;
                    *name = Some(format!("{}{}", ctx.opts().anon_fields_prefix, anon_field_counter));
                },
                Field::Bitfields(ref mut bu) => {
                    for bitfield in &mut bu.bitfields {
                        if bitfield.name().is_none() {
                            continue;
                        }
                        if let Some(AccessorNamesPair { getter, setter }) =
                            accessor_names.remove(bitfield.name().unwrap())
                        {
                            bitfield.getter = Some(getter);
                            bitfield.setter = Some(setter);
                        }
                    }
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
            CompFields::Before(ref fields) => {
                for f in fields {
                    tracer.visit_kind(f.ty().into(), EdgeKind::Field);
                }
            },
            CompFields::After { ref fields, .. } => {
                for f in fields {
                    f.trace(ctx, tracer, &());
                }
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct FieldData {
    name: Option<String>,
    ty: TypeId,
    comment: Option<String>,
    annotations: Annotations,
    bitfield_width: Option<u32>,
    public: bool,
    offset: Option<usize>,
}
impl FieldMethods for FieldData {
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
    fn annotations(&self) -> &Annotations {
        &self.annotations
    }
    fn offset(&self) -> Option<usize> {
        self.offset
    }
}
impl DotAttrs for FieldData {
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
struct RawField(FieldData);
impl RawField {
    fn new(
        name: Option<String>,
        ty: TypeId,
        comment: Option<String>,
        annotations: Option<Annotations>,
        bitfield_width: Option<u32>,
        public: bool,
        offset: Option<usize>,
    ) -> RawField {
        RawField(FieldData {
            name,
            ty,
            comment,
            annotations: annotations.unwrap_or_default(),
            bitfield_width,
            public,
            offset,
        })
    }
}
impl FieldMethods for RawField {
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
    fn annotations(&self) -> &Annotations {
        self.0.annotations()
    }
    fn offset(&self) -> Option<usize> {
        self.0.offset()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BaseKind {
    Normal,
    Virtual,
}

#[derive(Clone, Debug)]
pub struct Base {
    pub ty: TypeId,
    pub kind: BaseKind,
    pub field_name: String,
    pub is_pub: bool,
}
impl Base {
    pub fn is_virtual(&self) -> bool {
        self.kind == BaseKind::Virtual
    }
    pub fn requires_storage(&self, ctx: &Context) -> bool {
        if self.is_virtual() {
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
pub struct CompInfo {
    kind: CompKind,
    fields: CompFields,
    template_params: Vec<TypeId>,
    methods: Vec<Method>,
    constructors: Vec<FnId>,
    destructor: Option<(MethodKind, FnId)>,
    base_members: Vec<Base>,
    inner_types: Vec<TypeId>,
    inner_vars: Vec<VarId>,
    has_own_virtual_method: bool,
    has_destructor: bool,
    has_nonempty_base: bool,
    has_non_type_template_params: bool,
    has_unevaluable_bit_field_width: bool,
    packed_attr: bool,
    found_unknown_attr: bool,
    is_forward_declaration: bool,
}
impl CompInfo {
    pub fn new(kind: CompKind) -> Self {
        CompInfo {
            kind,
            fields: CompFields::default(),
            template_params: vec![],
            methods: vec![],
            constructors: vec![],
            destructor: None,
            base_members: vec![],
            inner_types: vec![],
            inner_vars: vec![],
            has_own_virtual_method: false,
            has_destructor: false,
            has_nonempty_base: false,
            has_non_type_template_params: false,
            has_unevaluable_bit_field_width: false,
            packed_attr: false,
            found_unknown_attr: false,
            is_forward_declaration: false,
        }
    }
    pub fn layout(&self, ctx: &Context) -> Option<Layout> {
        if self.kind == CompKind::Struct {
            return None;
        }
        if self.is_forward_declaration() {
            return None;
        }
        if !self.has_fields() {
            return None;
        }
        let mut max_size = 0;
        let mut max_align = 1;
        self.each_known_field_layout(ctx, |layout| {
            max_size = cmp::max(max_size, layout.size);
            max_align = cmp::max(max_align, layout.align);
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
    fn each_known_field_layout(&self, ctx: &Context, mut callback: impl FnMut(Layout)) {
        match self.fields {
            CompFields::Error => {},
            CompFields::After { ref fields, .. } => {
                for field in fields.iter() {
                    if let Some(layout) = field.layout(ctx) {
                        callback(layout);
                    }
                }
            },
            CompFields::Before(ref raw_fields) => {
                for field in raw_fields.iter() {
                    let field_ty = ctx.resolve_type(field.0.ty);
                    if let Some(layout) = field_ty.layout(ctx) {
                        callback(layout);
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
        self.fields().iter().any(|field| match *field {
            Field::DataMember(..) => false,
            Field::Bitfields(ref unit) => unit.layout.size > RUST_DERIVE_IN_ARRAY_LIMIT,
        })
    }
    pub fn has_non_type_template_params(&self) -> bool {
        self.has_non_type_template_params
    }
    pub fn has_own_virtual_method(&self) -> bool {
        self.has_own_virtual_method
    }
    pub fn has_own_destructor(&self) -> bool {
        self.has_destructor
    }
    pub fn methods(&self) -> &[Method] {
        &self.methods
    }
    pub fn constructors(&self) -> &[FnId] {
        &self.constructors
    }
    pub fn destructor(&self) -> Option<(MethodKind, FnId)> {
        self.destructor
    }
    pub fn kind(&self) -> CompKind {
        self.kind
    }
    pub fn is_union(&self) -> bool {
        self.kind() == CompKind::Union
    }
    pub fn base_members(&self) -> &[Base] {
        &self.base_members
    }
    pub fn from_ty(
        potential_id: ItemId,
        ty: &clang::Type,
        location: Option<clang::Cursor>,
        ctx: &mut Context,
    ) -> Result<Self, parse::Error> {
        use clang_lib::*;
        assert!(
            ty.template_args().is_none(),
            "We handle template instantiations elsewhere"
        );
        let mut cursor = ty.declaration();
        let mut kind = Self::kind_from_cursor(&cursor);
        if kind.is_err() {
            if let Some(location) = location {
                kind = Self::kind_from_cursor(&location);
                cursor = location;
            }
        }
        let kind = kind?;
        debug!("CompInfo::from_ty({:?}, {:?})", kind, cursor);
        let mut ci = CompInfo::new(kind);
        ci.is_forward_declaration = location.map_or(true, |cur| match cur.kind() {
            CXCursor_ParmDecl => true,
            CXCursor_StructDecl | CXCursor_UnionDecl | CXCursor_ClassDecl => !cur.is_definition(),
            _ => false,
        });
        let mut maybe_anonymous_struct_field = None;
        cursor.visit(|cur2| {
            if cur2.kind() != CXCursor_FieldDecl {
                if let Some((ty, clang_ty, public, offset)) = maybe_anonymous_struct_field.take() {
                    if cur2.kind() == CXCursor_TypedefDecl && cur2.typedef_type().unwrap().canonical_type() == clang_ty
                    {
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
                        cur2.visit(|child| {
                            if child.cur_type() == clang_ty {
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
                            ci.has_unevaluable_bit_field_width = true;
                            return CXChildVisit_Break;
                        }
                        width
                    } else {
                        None
                    };
                    let field_type = Item::from_ty_or_ref(cur2.cur_type(), cur2, Some(potential_id), ctx);
                    let comment = cur2.raw_comment();
                    let annotations = Annotations::new(&cur2);
                    let name = cur2.spelling();
                    let is_public = cur2.public_accessible();
                    let offset = cur2.offset_of_field().ok();
                    assert!(!name.is_empty() || bit_width.is_some(), "Empty field name?");
                    let name = if name.is_empty() { None } else { Some(name) };
                    let field = RawField::new(name, field_type, comment, annotations, bit_width, is_public, offset);
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
                    ci.template_params.push(param);
                },
                CXCursor_CXXBaseSpecifier => {
                    let is_virtual_base = cur2.is_virtual_base();
                    ci.has_own_virtual_method |= is_virtual_base;
                    let kind = if is_virtual_base {
                        BaseKind::Virtual
                    } else {
                        BaseKind::Normal
                    };
                    let field_name = match ci.base_members.len() {
                        0 => "_base".into(),
                        n => format!("_base_{}", n),
                    };
                    let type_id = Item::from_ty_or_ref(cur2.cur_type(), cur2, None, ctx);
                    ci.base_members.push(Base {
                        ty: type_id,
                        kind,
                        field_name,
                        is_pub: cur2.access_specifier() == clang_lib::CX_CXXPublic,
                    });
                },
                CXCursor_Constructor | CXCursor_Destructor | CXCursor_CXXMethod => {
                    let is_virtual = cur2.method_is_virtual();
                    let is_static = cur2.method_is_static();
                    debug_assert!(!(is_static && is_virtual), "How?");
                    ci.has_destructor |= cur2.kind() == CXCursor_Destructor;
                    ci.has_own_virtual_method |= is_virtual;
                    if !ci.template_params.is_empty() {
                        return CXChildVisit_Continue;
                    }
                    let signature = match Item::parse(cur2, Some(potential_id), ctx) {
                        Ok(item) if ctx.resolve_item(item).kind().is_function() => item,
                        _ => return CXChildVisit_Continue,
                    };
                    let signature = signature.expect_function_id(ctx);
                    match cur2.kind() {
                        CXCursor_Constructor => {
                            ci.constructors.push(signature);
                        },
                        CXCursor_Destructor => {
                            let kind = if is_virtual {
                                MethodKind::VirtualDestructor {
                                    pure_virtual: cur2.method_is_pure_virtual(),
                                }
                            } else {
                                MethodKind::Destructor
                            };
                            ci.destructor = Some((kind, signature));
                        },
                        CXCursor_CXXMethod => {
                            let is_const = cur2.method_is_const();
                            let method_kind = if is_static {
                                MethodKind::Static
                            } else if is_virtual {
                                MethodKind::Virtual {
                                    pure_virtual: cur2.method_is_pure_virtual(),
                                }
                            } else {
                                MethodKind::Normal
                            };
                            let method = Method::new(method_kind, signature, is_const);
                            ci.methods.push(method);
                        },
                        _ => unreachable!("How can we see this here?"),
                    }
                },
                CXCursor_NonTypeTemplateParameter => {
                    ci.has_non_type_template_params = true;
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
                match cur.template_kind() {
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
            self.each_known_field_layout(ctx, |layout| {
                packed = packed || layout.align > parent_layout.align;
            });
            if packed {
                info!("Found a struct that was defined within `#pragma packed(...)`");
                return true;
            }
            if self.has_own_virtual_method && parent_layout.align == 1 {
                return true;
            }
        }
        false
    }
    pub fn is_forward_declaration(&self) -> bool {
        self.is_forward_declaration
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
        if self.is_forward_declaration() {
            return (false, false);
        }
        let union_style = if ctx.opts().bindgen_wrapper_union.matches(name) {
            NonCopyUnionStyle::BindgenWrapper
        } else if ctx.opts().manually_drop_union.matches(name) {
            NonCopyUnionStyle::ManuallyDrop
        } else {
            ctx.opts().default_non_copy_union_style
        };
        let all_can_copy = self.fields().iter().all(|f| match *f {
            Field::DataMember(ref field_data) => field_data.ty().can_derive_copy(ctx),
            Field::Bitfields(_) => true,
        });
        if !all_can_copy && union_style == NonCopyUnionStyle::BindgenWrapper {
            return (false, false);
        }
        if layout.map_or(false, |l| l.size == 0) {
            return (false, false);
        }
        (true, all_can_copy)
    }
}
impl DotAttrs for CompInfo {
    fn dot_attrs<W>(&self, ctx: &Context, y: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(y, "<tr><td>CompKind</td><td>{:?}</td></tr>", self.kind)?;
        if self.has_own_virtual_method {
            writeln!(y, "<tr><td>has_vtable</td><td>true</td></tr>")?;
        }
        if self.has_destructor {
            writeln!(y, "<tr><td>has_destructor</td><td>true</td></tr>")?;
        }
        if self.has_nonempty_base {
            writeln!(y, "<tr><td>has_nonempty_base</td><td>true</td></tr>")?;
        }
        if self.has_non_type_template_params {
            writeln!(y, "<tr><td>has_non_type_template_params</td><td>true</td></tr>")?;
        }
        if self.packed_attr {
            writeln!(y, "<tr><td>packed_attr</td><td>true</td></tr>")?;
        }
        if self.is_forward_declaration {
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
impl IsOpaque for CompInfo {
    type Extra = Option<Layout>;
    fn is_opaque(&self, ctx: &Context, layout: &Option<Layout>) -> bool {
        if self.has_non_type_template_params || self.has_unevaluable_bit_field_width {
            return true;
        }
        if let CompFields::Error = self.fields {
            return true;
        }
        if self.fields().iter().any(|f| match *f {
            Field::DataMember(_) => false,
            Field::Bitfields(ref unit) => unit.bitfields().iter().any(|bf| {
                let bitfield_layout = ctx
                    .resolve_type(bf.ty())
                    .layout(ctx)
                    .expect("Bitfield without layout? Gah!");
                bf.width() / 8 > bitfield_layout.size as u32
            }),
        }) {
            return true;
        }
        false
    }
}
impl TemplParams for CompInfo {
    fn self_template_params(&self, _ctx: &Context) -> Vec<TypeId> {
        self.template_params.clone()
    }
}
impl Trace for CompInfo {
    type Extra = Item;
    fn trace<T>(&self, ctx: &Context, tracer: &mut T, i: &Item)
    where
        T: Tracer,
    {
        for p in i.all_template_params(ctx) {
            tracer.visit_kind(p.into(), EdgeKind::TemplateParameterDefinition);
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
        if let Some((_kind, signature)) = self.destructor() {
            tracer.visit_kind(signature.into(), EdgeKind::Destructor);
        }
        for ctor in self.constructors() {
            tracer.visit_kind(ctor.into(), EdgeKind::Constructor);
        }
        if i.is_opaque(ctx, &()) {
            return;
        }
        for base in self.base_members() {
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
    let mut raw_fields = raw_fields.into_iter().fuse().peekable();
    let mut fields = vec![];
    let mut bitfield_unit_count = 0;
    loop {
        {
            let non_bitfields = raw_fields
                .by_ref()
                .peeking_take_while(|f| f.bitfield_width().is_none())
                .map(|f| Field::DataMember(f.0));
            fields.extend(non_bitfields);
        }
        let mut bitfields = raw_fields
            .by_ref()
            .peeking_take_while(|f| f.bitfield_width().is_some())
            .peekable();
        if bitfields.peek().is_none() {
            break;
        }
        bitfields_to_allocation_units(ctx, &mut bitfield_unit_count, &mut fields, bitfields, packed)?;
    }
    assert!(
        raw_fields.next().is_none(),
        "The above loop should consume all items in `raw_fields`"
    );
    Ok((fields, bitfield_unit_count != 0))
}

fn bitfields_to_allocation_units<E, I>(
    ctx: &Context,
    bitfield_unit_count: &mut usize,
    fields: &mut E,
    raw_bitfields: I,
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
    for bitfield in raw_bitfields {
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
