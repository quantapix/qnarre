#![allow(dead_code, non_snake_case, non_camel_case_types, non_upper_case_globals)]

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Struct {
    /// This is field
    pub field: ::std::os::raw::c_int,
}
pub type AliasToStruct = Struct;
pub type AliasToInt = ::std::os::raw::c_int;
pub type AliasToAliasToInt = AliasToInt;
