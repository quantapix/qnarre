use super::context::BindgenContext;
use super::dot::DotAttrs;
use super::item::ItemSet;
use crate::clang;
use crate::parse::{ClangSubItemParser, ParseError, ParseResult};
use crate::parse_one;
use std::io;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum ModKind {
    Normal,
    Inline,
}

#[derive(Clone, Debug)]
pub(crate) struct Module {
    name: Option<String>,
    kind: ModKind,
    children: ItemSet,
}

impl Module {
    pub(crate) fn new(name: Option<String>, kind: ModKind) -> Self {
        Module {
            name,
            kind,
            children: ItemSet::new(),
        }
    }

    pub(crate) fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub(crate) fn children_mut(&mut self) -> &mut ItemSet {
        &mut self.children
    }

    pub(crate) fn children(&self) -> &ItemSet {
        &self.children
    }

    pub(crate) fn is_inline(&self) -> bool {
        self.kind == ModKind::Inline
    }
}

impl DotAttrs for Module {
    fn dot_attributes<W>(&self, _ctx: &BindgenContext, out: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(out, "<tr><td>ModuleKind</td><td>{:?}</td></tr>", self.kind)
    }
}

impl ClangSubItemParser for Module {
    fn parse(cursor: clang::Cursor, ctx: &mut BindgenContext) -> Result<ParseResult<Self>, ParseError> {
        use clang_lib::*;
        match cursor.kind() {
            CXCursor_Namespace => {
                let module_id = ctx.module(cursor);
                ctx.with_module(module_id, |ctx| {
                    cursor.visit(|cursor| parse_one(ctx, cursor, Some(module_id.into())))
                });

                Ok(ParseResult::AlreadyResolved(module_id.into()))
            },
            _ => Err(ParseError::Continue),
        }
    }
}
