use super::{Block, BlockRef};
use mlir_sys::{
    mlirRegionAppendOwnedBlock, mlirRegionCreate, mlirRegionDestroy, mlirRegionEqual, mlirRegionGetFirstBlock,
    mlirRegionInsertOwnedBlockAfter, mlirRegionInsertOwnedBlockBefore, MlirRegion,
};
use std::{
    marker::PhantomData,
    mem::{forget, transmute},
    ops::Deref,
};

#[derive(Debug)]
pub struct Region {
    raw: MlirRegion,
}

impl Region {
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirRegionCreate() },
        }
    }

    pub fn first_block(&self) -> Option<BlockRef> {
        unsafe {
            let block = mlirRegionGetFirstBlock(self.raw);

            if block.ptr.is_null() {
                None
            } else {
                Some(BlockRef::from_raw(block))
            }
        }
    }

    pub fn insert_block_after(&self, one: BlockRef, other: Block) -> BlockRef {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockAfter(self.raw, one.to_raw(), other.into_raw());

            r#ref
        }
    }

    pub fn insert_block_before(&self, one: BlockRef, other: Block) -> BlockRef {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockBefore(self.raw, one.to_raw(), other.into_raw());

            r#ref
        }
    }

    pub fn append_block(&self, block: Block) -> BlockRef {
        unsafe {
            let r#ref = BlockRef::from_raw(block.to_raw());

            mlirRegionAppendOwnedBlock(self.raw, block.into_raw());

            r#ref
        }
    }

    pub fn into_raw(self) -> mlir_sys::MlirRegion {
        let region = self.raw;

        forget(self);

        region
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        unsafe { mlirRegionDestroy(self.raw) }
    }
}

impl PartialEq for Region {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl Eq for Region {}

#[derive(Clone, Copy, Debug)]
pub struct RegionRef<'a> {
    raw: MlirRegion,
    _region: PhantomData<&'a Region>,
}

impl<'a> RegionRef<'a> {
    pub unsafe fn from_raw(raw: MlirRegion) -> Self {
        Self {
            raw,
            _region: Default::default(),
        }
    }

    pub unsafe fn from_option_raw(raw: MlirRegion) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}

impl<'a> Deref for RegionRef<'a> {
    type Target = Region;

    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}

impl<'a> PartialEq for RegionRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl<'a> Eq for RegionRef<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Region::new();
    }

    #[test]
    fn first_block() {
        assert!(Region::new().first_block().is_none());
    }

    #[test]
    fn append_block() {
        let region = Region::new();
        let block = Block::new(&[]);

        region.append_block(block);

        assert!(region.first_block().is_some());
    }

    #[test]
    fn insert_block_after() {
        let region = Region::new();

        let block = region.append_block(Block::new(&[]));
        region.insert_block_after(block, Block::new(&[]));

        assert_eq!(region.first_block(), Some(block));
    }

    #[test]
    fn insert_block_before() {
        let region = Region::new();

        let block = region.append_block(Block::new(&[]));
        let block = region.insert_block_before(block, Block::new(&[]));

        assert_eq!(region.first_block(), Some(block));
    }

    #[test]
    fn equal() {
        let region = Region::new();

        assert_eq!(region, region);
    }

    #[test]
    fn not_equal() {
        assert_ne!(Region::new(), Region::new());
    }
}
