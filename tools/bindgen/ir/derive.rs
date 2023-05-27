use super::context::BindgenContext;

use std::cmp;
use std::ops;

pub(crate) trait CanDeriveDebug {
    fn can_derive_debug(&self, ctx: &BindgenContext) -> bool;
}

pub(crate) trait CanDeriveCopy {
    fn can_derive_copy(&self, ctx: &BindgenContext) -> bool;
}

pub(crate) trait CanDeriveDefault {
    fn can_derive_default(&self, ctx: &BindgenContext) -> bool;
}

pub(crate) trait CanDeriveHash {
    fn can_derive_hash(&self, ctx: &BindgenContext) -> bool;
}

pub(crate) trait CanDerivePartialEq {
    fn can_derive_partialeq(&self, ctx: &BindgenContext) -> bool;
}

pub(crate) trait CanDerivePartialOrd {
    fn can_derive_partialord(&self, ctx: &BindgenContext) -> bool;
}

pub(crate) trait CanDeriveEq {
    fn can_derive_eq(&self, ctx: &BindgenContext) -> bool;
}

pub(crate) trait CanDeriveOrd {
    fn can_derive_ord(&self, ctx: &BindgenContext) -> bool;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum YDerive {
    Yes,
    Manually,
    No,
}

impl Default for YDerive {
    fn default() -> YDerive {
        YDerive::Yes
    }
}

impl YDerive {
    pub(crate) fn join(self, rhs: Self) -> Self {
        cmp::max(self, rhs)
    }
}

impl ops::BitOr for YDerive {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.join(rhs)
    }
}

impl ops::BitOrAssign for YDerive {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.join(rhs)
    }
}
