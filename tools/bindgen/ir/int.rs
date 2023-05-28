#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IntKind {
    Bool,
    SChar,
    UChar,
    WChar,
    Char { is_signed: bool },
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    LongLong,
    ULongLong,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    I128,
    U128,
    Custom { name: &'static str, is_signed: bool },
}

impl IntKind {
    pub(crate) fn is_signed(&self) -> bool {
        use self::IntKind::*;
        match *self {
            Bool | UChar | UShort | UInt | ULong | ULongLong | U8 | U16 | WChar | U32 | U64 | U128 => false,
            SChar | Short | Int | Long | LongLong | I8 | I16 | I32 | I64 | I128 => true,
            Char { is_signed } => is_signed,
            Custom { is_signed, .. } => is_signed,
        }
    }

    pub(crate) fn known_size(&self) -> Option<usize> {
        use self::IntKind::*;
        Some(match *self {
            Bool | UChar | SChar | U8 | I8 | Char { .. } => 1,
            U16 | I16 => 2,
            U32 | I32 => 4,
            U64 | I64 => 8,
            I128 | U128 => 16,
            _ => return None,
        })
    }

    pub(crate) fn signedness_matches(&self, val: i64) -> bool {
        val >= 0 || self.is_signed()
    }
}
