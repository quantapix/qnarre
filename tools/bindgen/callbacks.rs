pub use crate::ir::analysis::DeriveTrait;
pub use crate::ir::derive::CanDerive as ImplementsTrait;
pub use crate::ir::enum_ty::{EnumVariantCustomBehavior, EnumVariantValue};
pub use crate::ir::int::IntKind;
use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MacroParsing {
    Ignore,
    Default,
}

impl Default for MacroParsing {
    fn default() -> Self {
        MacroParsing::Default
    }
}

pub trait ParseCallbacks: fmt::Debug {
    #[cfg(feature = "__cli")]
    fn cli_args(&self) -> Vec<String> {
        vec![]
    }
    fn will_parse_macro(&self, _name: &str) -> MacroParsing {
        MacroParsing::Default
    }
    fn generated_name_override(&self, _item_info: ItemInfo<'_>) -> Option<String> {
        None
    }
    fn generated_link_name_override(&self, _item_info: ItemInfo<'_>) -> Option<String> {
        None
    }
    fn int_macro(&self, _name: &str, _value: i64) -> Option<IntKind> {
        None
    }
    fn str_macro(&self, _name: &str, _value: &[u8]) {}
    fn func_macro(&self, _name: &str, _value: &[&[u8]]) {}
    fn enum_variant_behavior(
        &self,
        _enum_name: Option<&str>,
        _original_variant_name: &str,
        _variant_value: EnumVariantValue,
    ) -> Option<EnumVariantCustomBehavior> {
        None
    }
    fn enum_variant_name(
        &self,
        _enum_name: Option<&str>,
        _original_variant_name: &str,
        _variant_value: EnumVariantValue,
    ) -> Option<String> {
        None
    }
    fn item_name(&self, _original_item_name: &str) -> Option<String> {
        None
    }
    fn include_file(&self, _filename: &str) {}
    fn read_env_var(&self, _key: &str) {}
    fn blocklisted_type_implements_trait(&self, _name: &str, _derive_trait: DeriveTrait) -> Option<ImplementsTrait> {
        None
    }
    fn add_derives(&self, _info: &DeriveInfo<'_>) -> Vec<String> {
        vec![]
    }
    fn process_comment(&self, _comment: &str) -> Option<String> {
        None
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub struct DeriveInfo<'a> {
    pub name: &'a str,
    pub kind: TypeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeKind {
    Struct,
    Enum,
    Union,
}

#[non_exhaustive]
pub struct ItemInfo<'a> {
    pub name: &'a str,
    pub kind: ItemKind,
}

#[non_exhaustive]
pub enum ItemKind {
    Function,
    Var,
}
