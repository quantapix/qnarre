#![allow(dead_code, non_snake_case, non_camel_case_types, non_upper_case_globals)]

#[repr(C)]
#[derive(Default, Debug)]
pub struct my_type {
    pub a: ::std::os::raw::c_int,
}
#[repr(C)]
#[derive(Default, Debug, Clone)]
pub struct my_type2 {
    pub a: ::std::os::raw::c_uint,
}
#[repr(C)]
#[derive(Default, Debug, Clone)]
pub struct my_type3 {
    pub a: ::std::os::raw::c_ulong,
}
