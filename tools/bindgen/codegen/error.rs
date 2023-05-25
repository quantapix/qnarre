use std::error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Error {
    NoLayoutForOpaqueBlob,

    InstantiationOfOpaqueType,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match *self {
            Error::NoLayoutForOpaqueBlob => "Tried to generate an opaque blob, but had no layout",
            Error::InstantiationOfOpaqueType => {
                "Instantiation of opaque template type or partial template \
                 specialization"
            },
        })
    }
}

impl error::Error for Error {}

pub(crate) type Result<T> = ::std::result::Result<T, Error>;
