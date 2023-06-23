use super::utils;
use crate::ir::{
    comp::Comp,
    typ::{Type, TypeKind},
    Context, Layout,
};
use proc_macro2::{self, Ident, Span};
use std::cmp;

const MAX_GUARANTEED_ALIGN: usize = 8;

#[derive(Debug)]
pub struct Structure<'a> {
    name: &'a str,
    ctx: &'a Context,
    comp: &'a Comp,
    is_packed: bool,
    known_type_layout: Option<Layout>,
    is_union: bool,
    can_copy_union_fields: bool,
    latest_offset: usize,
    padding_count: usize,
    latest_field_layout: Option<Layout>,
    max_field_align: usize,
    last_field_was_bitfield: bool,
}
impl<'a> Structure<'a> {
    pub fn new(ctx: &'a Context, comp: &'a Comp, ty: &'a Type, name: &'a str) -> Self {
        let known_type_layout = ty.layout(ctx);
        let is_packed = comp.is_packed(ctx, known_type_layout.as_ref());
        let (is_union, can_copy_union_fields) = comp.is_rust_union(ctx, known_type_layout.as_ref(), name);
        Structure {
            name,
            ctx,
            comp,
            is_packed,
            known_type_layout,
            is_union,
            can_copy_union_fields,
            latest_offset: 0,
            padding_count: 0,
            latest_field_layout: None,
            max_field_align: 0,
            last_field_was_bitfield: false,
        }
    }
    pub fn can_copy_union_fields(&self) -> bool {
        self.can_copy_union_fields
    }
    pub fn is_union(&self) -> bool {
        self.is_union
    }
    pub fn saw_vtable(&mut self) {
        debug!("saw vtable for {}", self.name);
        let ptr_size = self.ctx.target_ptr_size();
        self.latest_offset += ptr_size;
        self.latest_field_layout = Some(Layout::new(ptr_size, ptr_size));
        self.max_field_align = ptr_size;
    }
    pub fn saw_base(&mut self, base_ty: &Type) {
        debug!("saw base for {}", self.name);
        if let Some(layout) = base_ty.layout(self.ctx) {
            self.align_to_latest_field(layout);
            self.latest_offset += self.padding_bytes(layout) + layout.size;
            self.latest_field_layout = Some(layout);
            self.max_field_align = cmp::max(self.max_field_align, layout.align);
        }
    }
    pub fn saw_bitfield(&mut self, layout: Layout) {
        debug!("saw bitfield unit for {}: {:?}", self.name, layout);
        self.align_to_latest_field(layout);
        self.latest_offset += layout.size;
        debug!(
            "Offset: <bitfield>: {} -> {}",
            self.latest_offset - layout.size,
            self.latest_offset
        );
        self.latest_field_layout = Some(layout);
        self.last_field_was_bitfield = true;
    }
    pub fn saw_field(
        &mut self,
        field_name: &str,
        field_ty: &Type,
        field_offset: Option<usize>,
    ) -> Option<proc_macro2::TokenStream> {
        let mut field_layout = field_ty.layout(self.ctx)?;
        if let TypeKind::Array(inner, len) = *field_ty.canon_type(self.ctx).kind() {
            if let Some(layout) = self.ctx.resolve_type(inner).layout(self.ctx) {
                if layout.align > MAX_GUARANTEED_ALIGN {
                    field_layout.size = align_to(layout.size, layout.align) * len;
                    field_layout.align = MAX_GUARANTEED_ALIGN;
                }
            }
        }
        self.saw_field_with_layout(field_name, field_layout, field_offset)
    }
    pub fn saw_field_with_layout(
        &mut self,
        field_name: &str,
        field_layout: Layout,
        field_offset: Option<usize>,
    ) -> Option<proc_macro2::TokenStream> {
        let will_merge_with_bitfield = self.align_to_latest_field(field_layout);
        let is_union = self.comp.is_union();
        let padding_bytes = match field_offset {
            Some(offset) if offset / 8 > self.latest_offset => offset / 8 - self.latest_offset,
            _ => {
                if will_merge_with_bitfield || field_layout.align == 0 || is_union {
                    0
                } else if !self.is_packed {
                    self.padding_bytes(field_layout)
                } else if let Some(l) = self.known_type_layout {
                    self.padding_bytes(l)
                } else {
                    0
                }
            },
        };
        self.latest_offset += padding_bytes;
        let padding_layout = if self.is_packed || is_union {
            None
        } else {
            let force_padding = self.ctx.opts().force_explicit_padding;
            let need_padding =
                force_padding || padding_bytes >= field_layout.align || field_layout.align > MAX_GUARANTEED_ALIGN;
            debug!(
                "Offset: <padding>: {} -> {}",
                self.latest_offset - padding_bytes,
                self.latest_offset
            );
            debug!(
                "align field {} to {}/{} with {} padding bytes {:?}",
                field_name,
                self.latest_offset,
                field_offset.unwrap_or(0) / 8,
                padding_bytes,
                field_layout
            );
            let padding_align = if force_padding {
                1
            } else {
                cmp::min(field_layout.align, MAX_GUARANTEED_ALIGN)
            };
            if need_padding && padding_bytes != 0 {
                Some(Layout::new(padding_bytes, padding_align))
            } else {
                None
            }
        };
        self.latest_offset += field_layout.size;
        self.latest_field_layout = Some(field_layout);
        self.max_field_align = cmp::max(self.max_field_align, field_layout.align);
        self.last_field_was_bitfield = false;
        debug!(
            "Offset: {}: {} -> {}",
            field_name,
            self.latest_offset - field_layout.size,
            self.latest_offset
        );
        padding_layout.map(|x| self.padding_field(x))
    }
    pub fn add_tail_padding(&mut self, comp_name: &str, comp_layout: Layout) -> Option<proc_macro2::TokenStream> {
        if !self.ctx.opts().force_explicit_padding {
            return None;
        }
        if self.is_union {
            return None;
        }
        if self.latest_offset == comp_layout.size {
            return None;
        }
        trace!(
            "need a tail padding field for {}: offset {} -> size {}",
            comp_name,
            self.latest_offset,
            comp_layout.size
        );
        let size = comp_layout.size - self.latest_offset;
        Some(self.padding_field(Layout::new(size, 0)))
    }
    pub fn pad_struct(&mut self, layout: Layout) -> Option<proc_macro2::TokenStream> {
        debug!("pad_struct:\n\tself = {:#?}\n\tlayout = {:#?}", self, layout);
        if layout.size < self.latest_offset {
            warn!(
                "Calculated wrong layout for {}, too more {} bytes",
                self.name,
                self.latest_offset - layout.size
            );
            return None;
        }
        let padding_bytes = layout.size - self.latest_offset;
        if padding_bytes == 0 {
            return None;
        }
        let repr_align = true;
        if padding_bytes >= layout.align
            || (self.last_field_was_bitfield && padding_bytes >= self.latest_field_layout.unwrap().align)
            || (!repr_align && layout.align > MAX_GUARANTEED_ALIGN)
        {
            let layout = if self.is_packed {
                Layout::new(padding_bytes, 1)
            } else if self.last_field_was_bitfield || layout.align > MAX_GUARANTEED_ALIGN {
                Layout::for_size(self.ctx, padding_bytes)
            } else {
                Layout::new(padding_bytes, layout.align)
            };
            debug!("pad bytes to struct {}, {:?}", self.name, layout);
            Some(self.padding_field(layout))
        } else {
            None
        }
    }
    pub fn requires_explicit_align(&self, layout: Layout) -> bool {
        let repr_align = true;
        if repr_align && self.max_field_align >= 16 {
            return true;
        }
        if self.max_field_align >= layout.align {
            return false;
        }
        repr_align || layout.align <= MAX_GUARANTEED_ALIGN
    }
    fn padding_bytes(&self, layout: Layout) -> usize {
        align_to(self.latest_offset, layout.align) - self.latest_offset
    }
    fn padding_field(&mut self, layout: Layout) -> proc_macro2::TokenStream {
        let ty = utils::blob(self.ctx, layout);
        let padding_count = self.padding_count;
        self.padding_count += 1;
        let padding_field_name = Ident::new(&format!("__bindgen_padding_{}", padding_count), Span::call_site());
        self.max_field_align = cmp::max(self.max_field_align, layout.align);
        quote! {
            pub #padding_field_name : #ty ,
        }
    }
    fn align_to_latest_field(&mut self, new_field_layout: Layout) -> bool {
        if self.is_packed {
            return false;
        }
        let layout = match self.latest_field_layout {
            Some(l) => l,
            None => return false,
        };
        debug!(
            "align_to_bitfield? {}: {:?} {:?}",
            self.last_field_was_bitfield, layout, new_field_layout
        );
        let align = cmp::max(1, layout.align);
        if self.last_field_was_bitfield
            && new_field_layout.align <= layout.size % align
            && new_field_layout.size <= layout.size % align
        {
            debug!("Will merge with bitfield");
            return true;
        }
        self.latest_offset += self.padding_bytes(layout);
        false
    }
}

pub fn align_to(size: usize, align: usize) -> usize {
    if align == 0 {
        return size;
    }
    let rem = size % align;
    if rem == 0 {
        return size;
    }
    size + align - rem
}

pub fn bytes_from_bits_pow2(mut n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n <= 8 {
        return 1;
    }
    if !n.is_power_of_two() {
        n = n.next_power_of_two();
    }
    n / 8
}

#[test]
fn test_align_to() {
    assert_eq!(align_to(1, 1), 1);
    assert_eq!(align_to(1, 2), 2);
    assert_eq!(align_to(1, 4), 4);
    assert_eq!(align_to(5, 1), 5);
    assert_eq!(align_to(17, 4), 20);
}

#[test]
fn test_bytes_from_bits_pow2() {
    assert_eq!(bytes_from_bits_pow2(0), 0);
    for i in 1..9 {
        assert_eq!(bytes_from_bits_pow2(i), 1);
    }
    for i in 9..17 {
        assert_eq!(bytes_from_bits_pow2(i), 2);
    }
    for i in 17..33 {
        assert_eq!(bytes_from_bits_pow2(i), 4);
    }
}