use super::context::BindgenContext;
use super::dot::DotAttrs;
use super::item::ItemSet;
use crate::clang;
use crate::parse;
use crate::parse_one;
use std::io;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ModKind {
    Normal,
    Inline,
}

#[derive(Clone, Debug)]
pub struct Module {
    name: Option<String>,
    kind: ModKind,
    children: ItemSet,
}

impl Module {
    pub fn new(name: Option<String>, kind: ModKind) -> Self {
        Module {
            name,
            kind,
            children: ItemSet::new(),
        }
    }
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    pub fn children_mut(&mut self) -> &mut ItemSet {
        &mut self.children
    }
    pub fn children(&self) -> &ItemSet {
        &self.children
    }
    pub fn is_inline(&self) -> bool {
        self.kind == ModKind::Inline
    }
}

impl DotAttrs for Module {
    fn dot_attrs<W>(&self, _ctx: &BindgenContext, out: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(out, "<tr><td>ModuleKind</td><td>{:?}</td></tr>", self.kind)
    }
}

impl parse::SubItem for Module {
    fn parse(cur: clang::Cursor, ctx: &mut BindgenContext) -> Result<parse::Result<Self>, parse::Error> {
        use clang_lib::*;
        match cur.kind() {
            CXCursor_Namespace => {
                let id = ctx.module(cur);
                ctx.with_module(id, |ctx| cur.visit(|x| parse_one(ctx, x, Some(id.into()))));
                Ok(parse::Result::AlreadyResolved(id.into()))
            },
            _ => Err(parse::Error::Continue),
        }
    }
}
