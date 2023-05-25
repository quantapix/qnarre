use super::analysis::Sizedness;
use super::annotations::Annotations;
use super::context::{BindgenContext, FunctionId, ItemId, TypeId, VarId};
use super::dot::DotAttributes;
use super::item::{IsOpaque, Item};
use super::layout::Layout;
use super::template::TemplateParameters;
use super::traversal::{EdgeKind, Trace, Tracer};
use super::ty::RUST_DERIVE_IN_ARRAY_LIMIT;
use crate::clang;
use crate::codegen::struct_layout::{align_to, bytes_from_bits_pow2};
use crate::ir::derive::CanDeriveCopy;
use crate::parse::ParseError;
use crate::HashMap;
use crate::NonCopyUnionStyle;
use peeking_take_while::PeekableExt;
use std::cmp;
use std::io;
use std::mem;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum CompKind {
    Struct,
    Union,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum MethodKind {
    Constructor,
    Destructor,
    VirtualDestructor { pure_virtual: bool },
    Static,
    Normal,
    Virtual { pure_virtual: bool },
}

impl MethodKind {
    pub(crate) fn is_destructor(&self) -> bool {
        matches!(*self, MethodKind::Destructor | MethodKind::VirtualDestructor { .. })
    }

    pub(crate) fn is_pure_virtual(&self) -> bool {
        match *self {
            MethodKind::Virtual { pure_virtual } | MethodKind::VirtualDestructor { pure_virtual } => pure_virtual,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Method {
    kind: MethodKind,
    signature: FunctionId,
    is_const: bool,
}

impl Method {
    pub(crate) fn new(kind: MethodKind, signature: FunctionId, is_const: bool) -> Self {
        Method {
            kind,
            signature,
            is_const,
        }
    }

    pub(crate) fn kind(&self) -> MethodKind {
        self.kind
    }

    pub(crate) fn is_constructor(&self) -> bool {
        self.kind == MethodKind::Constructor
    }

    pub(crate) fn is_virtual(&self) -> bool {
        matches!(
            self.kind,
            MethodKind::Virtual { .. } | MethodKind::VirtualDestructor { .. }
        )
    }

    pub(crate) fn is_static(&self) -> bool {
        self.kind == MethodKind::Static
    }

    pub(crate) fn signature(&self) -> FunctionId {
        self.signature
    }

    pub(crate) fn is_const(&self) -> bool {
        self.is_const
    }
}

pub(crate) trait FieldMethods {
    fn name(&self) -> Option<&str>;

    fn ty(&self) -> TypeId;

    fn comment(&self) -> Option<&str>;

    fn bitfield_width(&self) -> Option<u32>;

    fn is_public(&self) -> bool;

    fn annotations(&self) -> &Annotations;

    fn offset(&self) -> Option<usize>;
}

#[derive(Debug)]
pub(crate) struct BitfieldUnit {
    nth: usize,
    layout: Layout,
    bitfields: Vec<Bitfield>,
}

impl BitfieldUnit {
    pub(crate) fn nth(&self) -> usize {
        self.nth
    }

    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }

    pub(crate) fn bitfields(&self) -> &[Bitfield] {
        &self.bitfields
    }
}

#[derive(Debug)]
pub(crate) enum Field {
    DataMember(FieldData),

    Bitfields(BitfieldUnit),
}

impl Field {
    pub(crate) fn layout(&self, ctx: &BindgenContext) -> Option<Layout> {
        match *self {
            Field::Bitfields(BitfieldUnit { layout, .. }) => Some(layout),
            Field::DataMember(ref data) => ctx.resolve_type(data.ty).layout(ctx),
        }
    }
}

impl Trace for Field {
    type Extra = ();

    fn trace<T>(&self, _: &BindgenContext, tracer: &mut T, _: &())
    where
        T: Tracer,
    {
        match *self {
            Field::DataMember(ref data) => {
                tracer.visit_kind(data.ty.into(), EdgeKind::Field);
            },
            Field::Bitfields(BitfieldUnit { ref bitfields, .. }) => {
                for bf in bitfields {
                    tracer.visit_kind(bf.ty().into(), EdgeKind::Field);
                }
            },
        }
    }
}

impl DotAttributes for Field {
    fn dot_attributes<W>(&self, ctx: &BindgenContext, out: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        match *self {
            Field::DataMember(ref data) => data.dot_attributes(ctx, out),
            Field::Bitfields(BitfieldUnit {
                layout, ref bitfields, ..
            }) => {
                writeln!(
                    out,
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
                for bf in bitfields {
                    bf.dot_attributes(ctx, out)?;
                }
                writeln!(out, "</table></td></tr>")
            },
        }
    }
}

impl DotAttributes for FieldData {
    fn dot_attributes<W>(&self, _ctx: &BindgenContext, out: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(
            out,
            "<tr><td>{}</td><td>{:?}</td></tr>",
            self.name().unwrap_or("(anonymous)"),
            self.ty()
        )
    }
}

impl DotAttributes for Bitfield {
    fn dot_attributes<W>(&self, _ctx: &BindgenContext, out: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(
            out,
            "<tr><td>{} : {}</td><td>{:?}</td></tr>",
            self.name().unwrap_or("(anonymous)"),
            self.width(),
            self.ty()
        )
    }
}

#[derive(Debug)]
pub(crate) struct Bitfield {
    offset_into_unit: usize,

    data: FieldData,

    getter_name: Option<String>,

    setter_name: Option<String>,
}

impl Bitfield {
    fn new(offset_into_unit: usize, raw: RawField) -> Bitfield {
        assert!(raw.bitfield_width().is_some());

        Bitfield {
            offset_into_unit,
            data: raw.0,
            getter_name: None,
            setter_name: None,
        }
    }

    pub(crate) fn offset_into_unit(&self) -> usize {
        self.offset_into_unit
    }

    pub(crate) fn width(&self) -> u32 {
        self.data.bitfield_width().unwrap()
    }

    pub(crate) fn getter_name(&self) -> &str {
        assert!(
            self.name().is_some(),
            "`Bitfield::getter_name` called on anonymous field"
        );
        self.getter_name.as_ref().expect(
            "`Bitfield::getter_name` should only be called after\
             assigning bitfield accessor names",
        )
    }

    pub(crate) fn setter_name(&self) -> &str {
        assert!(
            self.name().is_some(),
            "`Bitfield::setter_name` called on anonymous field"
        );
        self.setter_name.as_ref().expect(
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

fn raw_fields_to_fields_and_bitfield_units<I>(
    ctx: &BindgenContext,
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
    ctx: &BindgenContext,
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
                    // We've reached the end of this allocation unit, so flush it
                    // and its bitfields.
                    unit_size_in_bits = align_to(unit_size_in_bits, unit_align * 8);
                    flush_allocation_unit(
                        fields,
                        bitfield_unit_count,
                        unit_size_in_bits,
                        unit_align,
                        mem::take(&mut bitfields_in_unit),
                        packed,
                    );

                    // Now we're working on a fresh bitfield allocation unit, so reset
                    // the current unit size and alignment.
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

#[derive(Debug)]
enum CompFields {
    Before(Vec<RawField>),
    After {
        fields: Vec<Field>,
        has_bitfield_units: bool,
    },
    Error,
}

impl Default for CompFields {
    fn default() -> CompFields {
        CompFields::Before(vec![])
    }
}

impl CompFields {
    fn append_raw_field(&mut self, raw: RawField) {
        match *self {
            CompFields::Before(ref mut raws) => {
                raws.push(raw);
            },
            _ => {
                panic!("Must not append new fields after computing bitfield allocation units");
            },
        }
    }

    fn compute_bitfield_units(&mut self, ctx: &BindgenContext, packed: bool) {
        let raws = match *self {
            CompFields::Before(ref mut raws) => mem::take(raws),
            _ => {
                panic!("Already computed bitfield units");
            },
        };

        let result = raw_fields_to_fields_and_bitfield_units(ctx, raws, packed);

        match result {
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

    fn deanonymize_fields(&mut self, ctx: &BindgenContext, methods: &[Method]) {
        let fields = match *self {
            CompFields::After { ref mut fields, .. } => fields,
            CompFields::Error => return,
            CompFields::Before(_) => {
                panic!("Not yet computed bitfield units.");
            },
        };

        fn has_method(methods: &[Method], ctx: &BindgenContext, name: &str) -> bool {
            methods.iter().any(|method| {
                let method_name = ctx.resolve_func(method.signature()).name();
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
            .filter_map(|bitfield| bitfield.name())
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
                    *name = Some(format!("{}{}", ctx.options().anon_fields_prefix, anon_field_counter));
                },
                Field::Bitfields(ref mut bu) => {
                    for bitfield in &mut bu.bitfields {
                        if bitfield.name().is_none() {
                            continue;
                        }

                        if let Some(AccessorNamesPair { getter, setter }) =
                            accessor_names.remove(bitfield.name().unwrap())
                        {
                            bitfield.getter_name = Some(getter);
                            bitfield.setter_name = Some(setter);
                        }
                    }
                },
            }
        }
    }
}

impl Trace for CompFields {
    type Extra = ();

    fn trace<T>(&self, context: &BindgenContext, tracer: &mut T, _: &())
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
                    f.trace(context, tracer, &());
                }
            },
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FieldData {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum BaseKind {
    Normal,
    Virtual,
}

#[derive(Clone, Debug)]
pub(crate) struct Base {
    pub(crate) ty: TypeId,
    pub(crate) kind: BaseKind,
    pub(crate) field_name: String,
    pub(crate) is_pub: bool,
}

impl Base {
    pub(crate) fn is_virtual(&self) -> bool {
        self.kind == BaseKind::Virtual
    }

    pub(crate) fn requires_storage(&self, ctx: &BindgenContext) -> bool {
        if self.is_virtual() {
            return false;
        }

        if self.ty.is_zero_sized(ctx) {
            return false;
        }

        true
    }

    pub(crate) fn is_public(&self) -> bool {
        self.is_pub
    }
}

#[derive(Debug)]
pub(crate) struct CompInfo {
    kind: CompKind,

    fields: CompFields,

    template_params: Vec<TypeId>,

    methods: Vec<Method>,

    constructors: Vec<FunctionId>,

    destructor: Option<(MethodKind, FunctionId)>,

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
    pub(crate) fn new(kind: CompKind) -> Self {
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

    pub(crate) fn layout(&self, ctx: &BindgenContext) -> Option<Layout> {
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

    pub(crate) fn fields(&self) -> &[Field] {
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

    fn each_known_field_layout(&self, ctx: &BindgenContext, mut callback: impl FnMut(Layout)) {
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

    pub(crate) fn has_too_large_bitfield_unit(&self) -> bool {
        if !self.has_bitfields() {
            return false;
        }
        self.fields().iter().any(|field| match *field {
            Field::DataMember(..) => false,
            Field::Bitfields(ref unit) => unit.layout.size > RUST_DERIVE_IN_ARRAY_LIMIT,
        })
    }

    pub(crate) fn has_non_type_template_params(&self) -> bool {
        self.has_non_type_template_params
    }

    pub(crate) fn has_own_virtual_method(&self) -> bool {
        self.has_own_virtual_method
    }

    pub(crate) fn has_own_destructor(&self) -> bool {
        self.has_destructor
    }

    pub(crate) fn methods(&self) -> &[Method] {
        &self.methods
    }

    pub(crate) fn constructors(&self) -> &[FunctionId] {
        &self.constructors
    }

    pub(crate) fn destructor(&self) -> Option<(MethodKind, FunctionId)> {
        self.destructor
    }

    pub(crate) fn kind(&self) -> CompKind {
        self.kind
    }

    pub(crate) fn is_union(&self) -> bool {
        self.kind() == CompKind::Union
    }

    pub(crate) fn base_members(&self) -> &[Base] {
        &self.base_members
    }

    pub(crate) fn from_ty(
        potential_id: ItemId,
        ty: &clang::Type,
        location: Option<clang::Cursor>,
        ctx: &mut BindgenContext,
    ) -> Result<Self, ParseError> {
        use clang::*;
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
        cursor.visit(|cur| {
            if cur.kind() != CXCursor_FieldDecl {
                if let Some((ty, clang_ty, public, offset)) = maybe_anonymous_struct_field.take() {
                    if cur.kind() == CXCursor_TypedefDecl && cur.typedef_type().unwrap().canonical_type() == clang_ty {
                        // Typedefs of anonymous structs appear later in the ast
                        // than the struct itself, that would otherwise be an
                        // anonymous field. Detect that case here, and do
                        // nothing.
                    } else {
                        let field = RawField::new(None, ty, None, None, None, public, offset);
                        ci.fields.append_raw_field(field);
                    }
                }
            }

            match cur.kind() {
                CXCursor_FieldDecl => {
                    if let Some((ty, clang_ty, public, offset)) = maybe_anonymous_struct_field.take() {
                        let mut used = false;
                        cur.visit(|child| {
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

                    let bit_width = if cur.is_bit_field() {
                        let width = cur.bit_width();

                        // Make opaque type if the bit width couldn't be
                        // evaluated.
                        if width.is_none() {
                            ci.has_unevaluable_bit_field_width = true;
                            return CXChildVisit_Break;
                        }

                        width
                    } else {
                        None
                    };

                    let field_type = Item::from_ty_or_ref(cur.cur_type(), cur, Some(potential_id), ctx);

                    let comment = cur.raw_comment();
                    let annotations = Annotations::new(&cur);
                    let name = cur.spelling();
                    let is_public = cur.public_accessible();
                    let offset = cur.offset_of_field().ok();

                    // Name can be empty if there are bitfields, for example,
                    // see tests/headers/struct_with_bitfields.h
                    assert!(!name.is_empty() || bit_width.is_some(), "Empty field name?");

                    let name = if name.is_empty() { None } else { Some(name) };

                    let field = RawField::new(name, field_type, comment, annotations, bit_width, is_public, offset);
                    ci.fields.append_raw_field(field);

                    // No we look for things like attributes and stuff.
                    cur.visit(|cur| {
                        if cur.kind() == CXCursor_UnexposedAttr {
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
                    // We can find non-semantic children here, clang uses a
                    // StructDecl to note incomplete structs that haven't been
                    // forward-declared before, see [1].
                    //
                    // Also, clang seems to scope struct definitions inside
                    // unions, and other named struct definitions inside other
                    // structs to the whole translation unit.
                    //
                    // Let's just assume that if the cursor we've found is a
                    // definition, it's a valid inner type.
                    //
                    // [1]: https://github.com/rust-lang/rust-bindgen/issues/482
                    let is_inner_struct = cur.semantic_parent() == cursor || cur.is_definition();
                    if !is_inner_struct {
                        return CXChildVisit_Continue;
                    }

                    // Even if this is a definition, we may not be the semantic
                    // parent, see #1281.
                    let inner = Item::parse(cur, Some(potential_id), ctx).expect("Inner ClassDecl");

                    // If we avoided recursion parsing this type (in
                    // `Item::from_ty_with_id()`), then this might not be a
                    // valid type ID, so check and gracefully handle this.
                    if ctx.resolve_item_fallible(inner).is_some() {
                        let inner = inner.expect_type_id(ctx);

                        ci.inner_types.push(inner);

                        // A declaration of an union or a struct without name
                        // could also be an unnamed field, unfortunately.
                        if cur.is_anonymous() && cur.kind() != CXCursor_EnumDecl {
                            let ty = cur.cur_type();
                            let public = cur.public_accessible();
                            let offset = cur.offset_of_field().ok();

                            maybe_anonymous_struct_field = Some((inner, ty, public, offset));
                        }
                    }
                },
                CXCursor_PackedAttr => {
                    ci.packed_attr = true;
                },
                CXCursor_TemplateTypeParameter => {
                    let param = Item::type_param(None, cur, ctx).expect(
                        "Item::type_param should't fail when pointing \
                         at a TemplateTypeParameter",
                    );
                    ci.template_params.push(param);
                },
                CXCursor_CXXBaseSpecifier => {
                    let is_virtual_base = cur.is_virtual_base();
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
                    let type_id = Item::from_ty_or_ref(cur.cur_type(), cur, None, ctx);
                    ci.base_members.push(Base {
                        ty: type_id,
                        kind,
                        field_name,
                        is_pub: cur.access_specifier() == clang::CX_CXXPublic,
                    });
                },
                CXCursor_Constructor | CXCursor_Destructor | CXCursor_CXXMethod => {
                    let is_virtual = cur.method_is_virtual();
                    let is_static = cur.method_is_static();
                    debug_assert!(!(is_static && is_virtual), "How?");

                    ci.has_destructor |= cur.kind() == CXCursor_Destructor;
                    ci.has_own_virtual_method |= is_virtual;

                    // This used to not be here, but then I tried generating
                    // stylo bindings with this (without path filters), and
                    // cried a lot with a method in gfx/Point.h
                    // (ToUnknownPoint), that somehow was causing the same type
                    // to be inserted in the map two times.
                    //
                    // I couldn't make a reduced test case, but anyway...
                    // Methods of template functions not only used to be inlined,
                    // but also instantiated, and we wouldn't be able to call
                    // them, so just bail out.
                    if !ci.template_params.is_empty() {
                        return CXChildVisit_Continue;
                    }

                    // NB: This gets us an owned `Function`, not a
                    // `FunctionSig`.
                    let signature = match Item::parse(cur, Some(potential_id), ctx) {
                        Ok(item) if ctx.resolve_item(item).kind().is_function() => item,
                        _ => return CXChildVisit_Continue,
                    };

                    let signature = signature.expect_function_id(ctx);

                    match cur.kind() {
                        CXCursor_Constructor => {
                            ci.constructors.push(signature);
                        },
                        CXCursor_Destructor => {
                            let kind = if is_virtual {
                                MethodKind::VirtualDestructor {
                                    pure_virtual: cur.method_is_pure_virtual(),
                                }
                            } else {
                                MethodKind::Destructor
                            };
                            ci.destructor = Some((kind, signature));
                        },
                        CXCursor_CXXMethod => {
                            let is_const = cur.method_is_const();
                            let method_kind = if is_static {
                                MethodKind::Static
                            } else if is_virtual {
                                MethodKind::Virtual {
                                    pure_virtual: cur.method_is_pure_virtual(),
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
                    let linkage = cur.linkage();
                    if linkage != CXLinkage_External && linkage != CXLinkage_UniqueExternal {
                        return CXChildVisit_Continue;
                    }

                    let visibility = cur.visibility();
                    if visibility != CXVisibility_Default {
                        return CXChildVisit_Continue;
                    }

                    if let Ok(item) = Item::parse(cur, Some(potential_id), ctx) {
                        ci.inner_vars.push(item.as_var_id_unchecked());
                    }
                },
                // Intentionally not handled
                CXCursor_CXXAccessSpecifier
                | CXCursor_CXXFinalAttr
                | CXCursor_FunctionTemplate
                | CXCursor_ConversionFunction => {},
                _ => {
                    warn!(
                        "unhandled comp member `{}` (kind {:?}) in `{}` ({})",
                        cur.spelling(),
                        clang::kind_to_str(cur.kind()),
                        cursor.spelling(),
                        cur.location()
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

    fn kind_from_cursor(cursor: &clang::Cursor) -> Result<CompKind, ParseError> {
        use clang::*;
        Ok(match cursor.kind() {
            CXCursor_UnionDecl => CompKind::Union,
            CXCursor_ClassDecl | CXCursor_StructDecl => CompKind::Struct,
            CXCursor_CXXBaseSpecifier | CXCursor_ClassTemplatePartialSpecialization | CXCursor_ClassTemplate => {
                match cursor.template_kind() {
                    CXCursor_UnionDecl => CompKind::Union,
                    _ => CompKind::Struct,
                }
            },
            _ => {
                warn!("Unknown kind for comp type: {:?}", cursor);
                return Err(ParseError::Continue);
            },
        })
    }

    pub(crate) fn inner_types(&self) -> &[TypeId] {
        &self.inner_types
    }

    pub(crate) fn inner_vars(&self) -> &[VarId] {
        &self.inner_vars
    }

    pub(crate) fn found_unknown_attr(&self) -> bool {
        self.found_unknown_attr
    }

    pub(crate) fn is_packed(&self, ctx: &BindgenContext, layout: Option<&Layout>) -> bool {
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

    pub(crate) fn is_forward_declaration(&self) -> bool {
        self.is_forward_declaration
    }

    pub(crate) fn compute_bitfield_units(&mut self, ctx: &BindgenContext, layout: Option<&Layout>) {
        let packed = self.is_packed(ctx, layout);
        self.fields.compute_bitfield_units(ctx, packed)
    }

    pub(crate) fn deanonymize_fields(&mut self, ctx: &BindgenContext) {
        self.fields.deanonymize_fields(ctx, &self.methods);
    }

    pub(crate) fn is_rust_union(&self, ctx: &BindgenContext, layout: Option<&Layout>, name: &str) -> (bool, bool) {
        if !self.is_union() {
            return (false, false);
        }

        if !ctx.options().untagged_union {
            return (false, false);
        }

        if self.is_forward_declaration() {
            return (false, false);
        }

        let union_style = if ctx.options().bindgen_wrapper_union.matches(name) {
            NonCopyUnionStyle::BindgenWrapper
        } else if ctx.options().manually_drop_union.matches(name) {
            NonCopyUnionStyle::ManuallyDrop
        } else {
            ctx.options().default_non_copy_union_style
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

impl DotAttributes for CompInfo {
    fn dot_attributes<W>(&self, ctx: &BindgenContext, out: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(out, "<tr><td>CompKind</td><td>{:?}</td></tr>", self.kind)?;

        if self.has_own_virtual_method {
            writeln!(out, "<tr><td>has_vtable</td><td>true</td></tr>")?;
        }

        if self.has_destructor {
            writeln!(out, "<tr><td>has_destructor</td><td>true</td></tr>")?;
        }

        if self.has_nonempty_base {
            writeln!(out, "<tr><td>has_nonempty_base</td><td>true</td></tr>")?;
        }

        if self.has_non_type_template_params {
            writeln!(out, "<tr><td>has_non_type_template_params</td><td>true</td></tr>")?;
        }

        if self.packed_attr {
            writeln!(out, "<tr><td>packed_attr</td><td>true</td></tr>")?;
        }

        if self.is_forward_declaration {
            writeln!(out, "<tr><td>is_forward_declaration</td><td>true</td></tr>")?;
        }

        if !self.fields().is_empty() {
            writeln!(out, r#"<tr><td>fields</td><td><table border="0">"#)?;
            for field in self.fields() {
                field.dot_attributes(ctx, out)?;
            }
            writeln!(out, "</table></td></tr>")?;
        }

        Ok(())
    }
}

impl IsOpaque for CompInfo {
    type Extra = Option<Layout>;

    fn is_opaque(&self, ctx: &BindgenContext, layout: &Option<Layout>) -> bool {
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

        if !ctx.options().rust_features().repr_packed_n {
            if self.is_packed(ctx, layout.as_ref()) && layout.map_or(false, |l| l.align > 1) {
                warn!(
                    "Found a type that is both packed and aligned to greater than \
                       1; Rust before version 1.33 doesn't have `#[repr(packed(N))]`, so we \
                       are treating it as opaque. You may wish to set bindgen's rust target \
                       version to 1.33 or later to enable `#[repr(packed(N))]` support."
                );
                return true;
            }
        }

        false
    }
}

impl TemplateParameters for CompInfo {
    fn self_template_params(&self, _ctx: &BindgenContext) -> Vec<TypeId> {
        self.template_params.clone()
    }
}

impl Trace for CompInfo {
    type Extra = Item;

    fn trace<T>(&self, context: &BindgenContext, tracer: &mut T, item: &Item)
    where
        T: Tracer,
    {
        for p in item.all_template_params(context) {
            tracer.visit_kind(p.into(), EdgeKind::TemplateParameterDefinition);
        }

        for ty in self.inner_types() {
            tracer.visit_kind(ty.into(), EdgeKind::InnerType);
        }

        for &var in self.inner_vars() {
            tracer.visit_kind(var.into(), EdgeKind::InnerVar);
        }

        for method in self.methods() {
            tracer.visit_kind(method.signature.into(), EdgeKind::Method);
        }

        if let Some((_kind, signature)) = self.destructor() {
            tracer.visit_kind(signature.into(), EdgeKind::Destructor);
        }

        for ctor in self.constructors() {
            tracer.visit_kind(ctor.into(), EdgeKind::Constructor);
        }

        if item.is_opaque(context, &()) {
            return;
        }

        for base in self.base_members() {
            tracer.visit_kind(base.ty.into(), EdgeKind::BaseMember);
        }

        self.fields.trace(context, tracer, &());
    }
}
