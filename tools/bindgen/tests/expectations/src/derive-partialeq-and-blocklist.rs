#![allow(dead_code, non_snake_case, non_camel_case_types, non_upper_case_globals)]

pub struct BlocklistMe(u8);

#[repr(C)]
pub struct ShouldNotDerivePartialEq {
    pub a: BlocklistMe,
}
#[test]
fn bindgen_test_layout_ShouldNotDerivePartialEq() {
    const UNINIT: ::std::mem::MaybeUninit<ShouldNotDerivePartialEq> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ShouldNotDerivePartialEq>(),
        1usize,
        concat!("Size of: ", stringify!(ShouldNotDerivePartialEq))
    );
    assert_eq!(
        ::std::mem::align_of::<ShouldNotDerivePartialEq>(),
        1usize,
        concat!("Alignment of ", stringify!(ShouldNotDerivePartialEq))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).a) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ShouldNotDerivePartialEq),
            "::",
            stringify!(a)
        )
    );
}
impl Default for ShouldNotDerivePartialEq {
    fn default() -> Self {
        let mut s = ::std::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::std::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
