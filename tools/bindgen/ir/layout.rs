use super::derive::YDerive;
use super::ty::{Type, TypeKind, RUST_DERIVE_IN_ARRAY_LIMIT};
use crate::clang;
use crate::ir::context::BindgenContext;
use std::cmp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Layout {
    pub(crate) size: usize,
    pub(crate) align: usize,
    pub(crate) packed: bool,
}

#[test]
fn test_layout_for_size() {
    use std::mem;

    let ptr_size = mem::size_of::<*mut ()>();
    assert_eq!(
        Layout::for_size_internal(ptr_size, ptr_size),
        Layout::new(ptr_size, ptr_size)
    );
    assert_eq!(
        Layout::for_size_internal(ptr_size, 3 * ptr_size),
        Layout::new(3 * ptr_size, ptr_size)
    );
}

impl Layout {
    pub(crate) fn known_type_for_size(ctx: &BindgenContext, size: usize) -> Option<&'static str> {
        Some(match size {
            16 => "u128",
            8 => "u64",
            4 => "u32",
            2 => "u16",
            1 => "u8",
            _ => return None,
        })
    }

    pub(crate) fn new(size: usize, align: usize) -> Self {
        Layout {
            size,
            align,
            packed: false,
        }
    }

    fn for_size_internal(ptr_size: usize, size: usize) -> Self {
        let mut next_align = 2;
        while size % next_align == 0 && next_align <= ptr_size {
            next_align *= 2;
        }
        Layout {
            size,
            align: next_align / 2,
            packed: false,
        }
    }

    pub(crate) fn for_size(ctx: &BindgenContext, size: usize) -> Self {
        Self::for_size_internal(ctx.target_pointer_size(), size)
    }

    pub(crate) fn opaque(&self) -> Opaque {
        Opaque(*self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Opaque(pub(crate) Layout);

impl Opaque {
    pub(crate) fn from_clang_ty(ty: &clang::Type, ctx: &BindgenContext) -> Type {
        let layout = Layout::new(ty.size(ctx), ty.align(ctx));
        let ty_kind = TypeKind::Opaque;
        let is_const = ty.is_const();
        Type::new(None, Some(layout), ty_kind, is_const)
    }

    pub(crate) fn known_rust_type_for_array(&self, ctx: &BindgenContext) -> Option<&'static str> {
        Layout::known_type_for_size(ctx, self.0.align)
    }

    pub(crate) fn array_size(&self, ctx: &BindgenContext) -> Option<usize> {
        if self.known_rust_type_for_array(ctx).is_some() {
            Some(self.0.size / cmp::max(self.0.align, 1))
        } else {
            None
        }
    }

    pub(crate) fn array_size_within_derive_limit(&self, ctx: &BindgenContext) -> YDerive {
        if self
            .array_size(ctx)
            .map_or(false, |size| size <= RUST_DERIVE_IN_ARRAY_LIMIT)
        {
            YDerive::Yes
        } else {
            YDerive::Manually
        }
    }
}
