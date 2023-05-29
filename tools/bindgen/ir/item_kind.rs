use super::context::BindgenContext;
use super::dot::DotAttrs;
use super::function::Function;
use super::module::Module;
use super::ty::Type;
use super::var::Var;
use std::io;

#[derive(Debug)]
pub enum ItemKind {
    Module(Module),
    Type(Type),
    Function(Function),
    Var(Var),
}

impl ItemKind {
    pub fn as_module(&self) -> Option<&Module> {
        match *self {
            ItemKind::Module(ref x) => Some(x),
            _ => None,
        }
    }
    pub fn kind_name(&self) -> &'static str {
        match *self {
            ItemKind::Module(..) => "Module",
            ItemKind::Type(..) => "Type",
            ItemKind::Function(..) => "Function",
            ItemKind::Var(..) => "Var",
        }
    }
    pub fn is_module(&self) -> bool {
        self.as_module().is_some()
    }
    pub fn as_function(&self) -> Option<&Function> {
        match *self {
            ItemKind::Function(ref x) => Some(x),
            _ => None,
        }
    }
    pub fn is_function(&self) -> bool {
        self.as_function().is_some()
    }
    pub fn expect_function(&self) -> &Function {
        self.as_function().expect("Not a function")
    }
    pub fn as_type(&self) -> Option<&Type> {
        match *self {
            ItemKind::Type(ref x) => Some(x),
            _ => None,
        }
    }
    pub fn as_type_mut(&mut self) -> Option<&mut Type> {
        match *self {
            ItemKind::Type(ref mut x) => Some(x),
            _ => None,
        }
    }
    pub fn is_type(&self) -> bool {
        self.as_type().is_some()
    }
    pub fn expect_type(&self) -> &Type {
        self.as_type().expect("Not a type")
    }
    pub fn as_var(&self) -> Option<&Var> {
        match *self {
            ItemKind::Var(ref v) => Some(v),
            _ => None,
        }
    }
    pub fn is_var(&self) -> bool {
        self.as_var().is_some()
    }
}

impl DotAttrs for ItemKind {
    fn dot_attrs<W>(&self, ctx: &BindgenContext, out: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(out, "<tr><td>kind</td><td>{}</td></tr>", self.kind_name())?;
        match *self {
            ItemKind::Module(ref x) => x.dot_attrs(ctx, out),
            ItemKind::Type(ref x) => x.dot_attrs(ctx, out),
            ItemKind::Function(ref x) => x.dot_attrs(ctx, out),
            ItemKind::Var(ref x) => x.dot_attrs(ctx, out),
        }
    }
}
