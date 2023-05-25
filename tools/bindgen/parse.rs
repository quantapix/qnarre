#![deny(clippy::missing_docs_in_private_items)]

use crate::clang;
use crate::ir::context::{BindgenContext, ItemId};

#[derive(Debug)]
pub(crate) enum ParseError {
    Recurse,
    Continue,
}

#[derive(Debug)]
pub(crate) enum ParseResult<T> {
    AlreadyResolved(ItemId),

    New(T, Option<clang::Cursor>),
}

pub(crate) trait ClangSubItemParser: Sized {
    fn parse(cursor: clang::Cursor, context: &mut BindgenContext) -> Result<ParseResult<Self>, ParseError>;
}
