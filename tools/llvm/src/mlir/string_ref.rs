use mlir_sys::{mlirStringRefCreateFromCString, mlirStringRefEqual, MlirStringRef};
use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    ffi::CString,
    marker::PhantomData,
    slice,
    str::{self, Utf8Error},
    sync::RwLock,
};

static STRING_CACHE: Lazy<RwLock<HashMap<String, CString>>> = Lazy::new(Default::default);

//
#[derive(Clone, Copy, Debug)]
pub struct StringRef<'a> {
    raw: MlirStringRef,
    _parent: PhantomData<&'a ()>,
}

impl<'a> StringRef<'a> {
    pub fn as_str(&self) -> Result<&'a str, Utf8Error> {
        unsafe {
            let bytes = slice::from_raw_parts(self.raw.data as *mut u8, self.raw.length);

            str::from_utf8(if bytes[bytes.len() - 1] == 0 {
                &bytes[..bytes.len() - 1]
            } else {
                bytes
            })
        }
    }

    pub fn to_raw(self) -> MlirStringRef {
        self.raw
    }

    pub unsafe fn from_raw(string: MlirStringRef) -> Self {
        Self {
            raw: string,
            _parent: Default::default(),
        }
    }
}

impl<'a> PartialEq for StringRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirStringRefEqual(self.raw, other.raw) }
    }
}

impl<'a> Eq for StringRef<'a> {}

impl From<&str> for StringRef<'static> {
    fn from(string: &str) -> Self {
        if !STRING_CACHE.read().unwrap().contains_key(string) {
            STRING_CACHE
                .write()
                .unwrap()
                .insert(string.to_owned(), CString::new(string).unwrap());
        }

        let lock = STRING_CACHE.read().unwrap();
        let string = lock.get(string).unwrap();

        unsafe { Self::from_raw(mlirStringRefCreateFromCString(string.as_ptr())) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal() {
        assert_eq!(StringRef::from("foo"), StringRef::from("foo"));
    }

    #[test]
    fn equal_str() {
        assert_eq!(StringRef::from("foo").as_str().unwrap(), "foo");
    }

    #[test]
    fn not_equal() {
        assert_ne!(StringRef::from("foo"), StringRef::from("bar"));
    }
}
