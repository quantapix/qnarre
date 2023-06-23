use super::MetaItem;
use rustc_span::{def_id::DefId, symbol::Ident};

pub mod allocator {
    use rustc_span::symbol::{sym, Symbol};

    pub const NO_ALLOC_SHIM_IS_UNSTABLE: &str = "__rust_no_alloc_shim_is_unstable";

    #[derive(Clone, Debug, Copy, Eq, PartialEq, HashStable_Generic)]
    pub enum AllocatorKind {
        Global,
        Default,
    }

    pub enum AllocatorTy {
        Layout,
        Ptr,
        ResultPtr,
        Unit,
        Usize,
    }

    pub struct AllocatorMethod {
        pub name: Symbol,
        pub inputs: &'static [AllocatorTy],
        pub output: AllocatorTy,
    }

    pub static ALLOCATOR_METHODS: &[AllocatorMethod] = &[
        AllocatorMethod {
            name: sym::alloc,
            inputs: &[AllocatorTy::Layout],
            output: AllocatorTy::ResultPtr,
        },
        AllocatorMethod {
            name: sym::dealloc,
            inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout],
            output: AllocatorTy::Unit,
        },
        AllocatorMethod {
            name: sym::realloc,
            inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout, AllocatorTy::Usize],
            output: AllocatorTy::ResultPtr,
        },
        AllocatorMethod {
            name: sym::alloc_zeroed,
            inputs: &[AllocatorTy::Layout],
            output: AllocatorTy::ResultPtr,
        },
    ];

    pub fn global_fn_name(base: Symbol) -> String {
        format!("__rust_{base}")
    }
    pub fn default_fn_name(base: Symbol) -> String {
        format!("__rdl_{base}")
    }
    pub fn alloc_error_handler_name(alloc_error_handler_kind: AllocatorKind) -> &'static str {
        match alloc_error_handler_kind {
            AllocatorKind::Global => "__rg_oom",
            AllocatorKind::Default => "__rdl_oom",
        }
    }
}

#[derive(Debug, Clone, Encodable, Decodable, HashStable_Generic)]
pub struct StrippedCfgItem<ModId = DefId> {
    pub parent_module: ModId,
    pub name: Ident,
    pub cfg: MetaItem,
}
impl<ModId> StrippedCfgItem<ModId> {
    pub fn map_mod_id<New>(self, f: impl FnOnce(ModId) -> New) -> StrippedCfgItem<New> {
        StrippedCfgItem {
            parent_module: f(self.parent_module),
            name: self.name,
            cfg: self.cfg,
        }
    }
}
