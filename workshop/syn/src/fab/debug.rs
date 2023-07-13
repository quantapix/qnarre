use crate::*;
use std::fmt::{self, Debug};
impl Debug for Abi {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Abi");
        formatter.field("extern_", &self.extern_);
        formatter.field("name", &self.name);
        formatter.finish()
    }
}
impl Debug for AngledArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl AngledArgs {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("colon2", &self.colon2);
                formatter.field("lt", &self.lt);
                formatter.field("args", &self.args);
                formatter.field("gt", &self.gt);
                formatter.finish()
            }
        }
        self.debug(formatter, "path::path::AngledArgs")
    }
}
impl Debug for Arm {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Arm");
        formatter.field("attrs", &self.attrs);
        formatter.field("pat", &self.pat);
        formatter.field("guard", &self.guard);
        formatter.field("fat_arrow", &self.fat_arrow);
        formatter.field("body", &self.body);
        formatter.field("comma", &self.comma);
        formatter.finish()
    }
}
impl Debug for AssocConst {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("AssocConst");
        formatter.field("ident", &self.ident);
        formatter.field("gens", &self.gnrs);
        formatter.field("eq", &self.eq);
        formatter.field("value", &self.val);
        formatter.finish()
    }
}
impl Debug for AssocType {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("AssocType");
        formatter.field("ident", &self.ident);
        formatter.field("gens", &self.gnrs);
        formatter.field("eq", &self.eq);
        formatter.field("ty", &self.ty);
        formatter.finish()
    }
}
impl Debug for attr::Style {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("attr::Style::")?;
        match self {
            attr::Style::Outer => formatter.write_str("Outer"),
            attr::Style::Inner(v0) => {
                let mut formatter = formatter.debug_tuple("Inner");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for attr::Attr {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("attr::Attr");
        formatter.field("pound", &self.pound);
        formatter.field("style", &self.style);
        formatter.field("bracket", &self.bracket);
        formatter.field("meta", &self.meta);
        formatter.finish()
    }
}
impl Debug for typ::FnArg {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::FnArg");
        formatter.field("attrs", &self.attrs);
        formatter.field("name", &self.name);
        formatter.field("ty", &self.ty);
        formatter.finish()
    }
}
impl Debug for typ::Variadic {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Variadic");
        formatter.field("attrs", &self.attrs);
        formatter.field("name", &self.name);
        formatter.field("dots", &self.dots);
        formatter.field("comma", &self.comma);
        formatter.finish()
    }
}
impl Debug for BinOp {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("BinOp::")?;
        match self {
            BinOp::Add(v0) => {
                let mut formatter = formatter.debug_tuple("Add");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Sub(v0) => {
                let mut formatter = formatter.debug_tuple("Sub");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Mul(v0) => {
                let mut formatter = formatter.debug_tuple("Mul");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Div(v0) => {
                let mut formatter = formatter.debug_tuple("Div");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Rem(v0) => {
                let mut formatter = formatter.debug_tuple("Rem");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::And(v0) => {
                let mut formatter = formatter.debug_tuple("And");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Or(v0) => {
                let mut formatter = formatter.debug_tuple("Or");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::BitXor(v0) => {
                let mut formatter = formatter.debug_tuple("BitXor");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::BitAnd(v0) => {
                let mut formatter = formatter.debug_tuple("BitAnd");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::BitOr(v0) => {
                let mut formatter = formatter.debug_tuple("BitOr");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Shl(v0) => {
                let mut formatter = formatter.debug_tuple("Shl");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Shr(v0) => {
                let mut formatter = formatter.debug_tuple("Shr");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Eq(v0) => {
                let mut formatter = formatter.debug_tuple("Eq");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Lt(v0) => {
                let mut formatter = formatter.debug_tuple("Lt");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Le(v0) => {
                let mut formatter = formatter.debug_tuple("Le");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Ne(v0) => {
                let mut formatter = formatter.debug_tuple("Ne");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Ge(v0) => {
                let mut formatter = formatter.debug_tuple("Ge");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::Gt(v0) => {
                let mut formatter = formatter.debug_tuple("Gt");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::AddAssign(v0) => {
                let mut formatter = formatter.debug_tuple("AddAssign");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::SubAssign(v0) => {
                let mut formatter = formatter.debug_tuple("SubAssign");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::MulAssign(v0) => {
                let mut formatter = formatter.debug_tuple("MulAssign");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::DivAssign(v0) => {
                let mut formatter = formatter.debug_tuple("DivAssign");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::RemAssign(v0) => {
                let mut formatter = formatter.debug_tuple("RemAssign");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::BitXorAssign(v0) => {
                let mut formatter = formatter.debug_tuple("BitXorAssign");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::BitAndAssign(v0) => {
                let mut formatter = formatter.debug_tuple("BitAndAssign");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::BitOrAssign(v0) => {
                let mut formatter = formatter.debug_tuple("BitOrAssign");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::ShlAssign(v0) => {
                let mut formatter = formatter.debug_tuple("ShlAssign");
                formatter.field(v0);
                formatter.finish()
            },
            BinOp::ShrAssign(v0) => {
                let mut formatter = formatter.debug_tuple("ShrAssign");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for Block {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Block");
        formatter.field("brace", &self.brace);
        formatter.field("stmts", &self.stmts);
        formatter.finish()
    }
}
impl Debug for Bgen::bound::Lifes {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Bgen::bound::Lifes");
        formatter.field("for_", &self.for_);
        formatter.field("lt", &self.lt);
        formatter.field("lifetimes", &self.lifes);
        formatter.field("gt", &self.gt);
        formatter.finish()
    }
}
impl Debug for gen::param::Const {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::param::Const");
        formatter.field("attrs", &self.attrs);
        formatter.field("const_", &self.const_);
        formatter.field("ident", &self.ident);
        formatter.field("colon", &self.colon);
        formatter.field("ty", &self.typ);
        formatter.field("eq", &self.eq);
        formatter.field("default", &self.default);
        formatter.finish()
    }
}
impl Debug for Constraint {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Constraint");
        formatter.field("ident", &self.ident);
        formatter.field("gens", &self.gnrs);
        formatter.field("colon", &self.colon);
        formatter.field("bounds", &self.bounds);
        formatter.finish()
    }
}
impl Debug for Data {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Data::")?;
        match self {
            Data::Struct(v0) => v0.debug(formatter, "Struct"),
            Data::Enum(v0) => v0.debug(formatter, "Enum"),
            Data::Union(v0) => v0.debug(formatter, "Union"),
        }
    }
}
impl Debug for data::Enum {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl data::Enum {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("enum_", &self.enum_);
                formatter.field("brace", &self.brace);
                formatter.field("variants", &self.variants);
                formatter.finish()
            }
        }
        self.debug(formatter, "data::Enum")
    }
}
impl Debug for data::Struct {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl data::Struct {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("struct_", &self.struct_);
                formatter.field("fields", &self.fields);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "data::Struct")
    }
}
impl Debug for data::Union {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl data::Union {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("union_", &self.union_);
                formatter.field("fields", &self.named);
                formatter.finish()
            }
        }
        self.debug(formatter, "data::Union")
    }
}
impl Debug for DeriveInput {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("DeriveInput");
        formatter.field("attrs", &self.attrs);
        formatter.field("vis", &self.vis);
        formatter.field("ident", &self.ident);
        formatter.field("gens", &self.gens);
        formatter.field("data", &self.data);
        formatter.finish()
    }
}
impl Debug for Expr {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Expr::")?;
        match self {
            Expr::Array(v0) => v0.debug(formatter, "Array"),
            Expr::Assign(v0) => v0.debug(formatter, "Assign"),
            Expr::Async(v0) => v0.debug(formatter, "Async"),
            Expr::Await(v0) => v0.debug(formatter, "Await"),
            Expr::Binary(v0) => v0.debug(formatter, "Binary"),
            Expr::Block(v0) => v0.debug(formatter, "Block"),
            Expr::Break(v0) => v0.debug(formatter, "Break"),
            Expr::Call(v0) => v0.debug(formatter, "Call"),
            Expr::Cast(v0) => v0.debug(formatter, "Cast"),
            Expr::Closure(v0) => v0.debug(formatter, "Closure"),
            Expr::Const(v0) => v0.debug(formatter, "Const"),
            Expr::Continue(v0) => v0.debug(formatter, "Continue"),
            Expr::Field(v0) => v0.debug(formatter, "Field"),
            Expr::ForLoop(v0) => v0.debug(formatter, "ForLoop"),
            Expr::Group(v0) => v0.debug(formatter, "Group"),
            Expr::If(v0) => v0.debug(formatter, "If"),
            Expr::Index(v0) => v0.debug(formatter, "Index"),
            Expr::Infer(v0) => v0.debug(formatter, "Infer"),
            Expr::Let(v0) => v0.debug(formatter, "Let"),
            Expr::Lit(v0) => v0.debug(formatter, "Lit"),
            Expr::Loop(v0) => v0.debug(formatter, "Loop"),
            Expr::Macro(v0) => v0.debug(formatter, "Macro"),
            Expr::Match(v0) => v0.debug(formatter, "Match"),
            Expr::MethodCall(v0) => v0.debug(formatter, "MethodCall"),
            Expr::Paren(v0) => v0.debug(formatter, "Paren"),
            Expr::Path(v0) => v0.debug(formatter, "Path"),
            Expr::Range(v0) => v0.debug(formatter, "Range"),
            Expr::Reference(v0) => v0.debug(formatter, "Reference"),
            Expr::Repeat(v0) => v0.debug(formatter, "Repeat"),
            Expr::Return(v0) => v0.debug(formatter, "Return"),
            Expr::Struct(v0) => v0.debug(formatter, "Struct"),
            Expr::Try(v0) => v0.debug(formatter, "Try"),
            Expr::TryBlock(v0) => v0.debug(formatter, "TryBlock"),
            Expr::Tuple(v0) => v0.debug(formatter, "Tuple"),
            Expr::Unary(v0) => v0.debug(formatter, "Unary"),
            Expr::Unsafe(v0) => v0.debug(formatter, "Unsafe"),
            Expr::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
            Expr::While(v0) => v0.debug(formatter, "While"),
            Expr::Yield(v0) => v0.debug(formatter, "Yield"),
            #[cfg(not(feature = "full"))]
            _ => unreachable!(),
        }
    }
}
impl Debug for expr::Array {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Array {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("bracket", &self.bracket);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Array")
    }
}
impl Debug for expr::Assign {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Assign {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("left", &self.left);
                formatter.field("eq", &self.eq);
                formatter.field("right", &self.right);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Assign")
    }
}
impl Debug for expr::Async {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Async {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("async_", &self.async_);
                formatter.field("capture", &self.move_);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Async")
    }
}
impl Debug for expr::Await {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Await {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("base", &self.expr);
                formatter.field("dot", &self.dot);
                formatter.field("await_", &self.await_);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Await")
    }
}
impl Debug for expr::Binary {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Binary {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("left", &self.left);
                formatter.field("op", &self.op);
                formatter.field("right", &self.right);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Binary")
    }
}
impl Debug for expr::Block {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Block {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("label", &self.label);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Block")
    }
}
impl Debug for expr::Break {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Break {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("break_", &self.break_);
                formatter.field("label", &self.label);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Break")
    }
}
impl Debug for expr::Call {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Call {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("func", &self.func);
                formatter.field("paren", &self.paren);
                formatter.field("args", &self.args);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Call")
    }
}
impl Debug for expr::Cast {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Cast {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("expr", &self.expr);
                formatter.field("as_", &self.as_);
                formatter.field("ty", &self.typ);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Cast")
    }
}
impl Debug for expr::Closure {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Closure {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("lifetimes", &self.lifes);
                formatter.field("const_", &self.const_);
                formatter.field("movability", &self.static_);
                formatter.field("asyncness", &self.async_);
                formatter.field("capture", &self.move_);
                formatter.field("or1", &self.or1);
                formatter.field("inputs", &self.inputs);
                formatter.field("or2", &self.or2);
                formatter.field("output", &self.ret);
                formatter.field("body", &self.body);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Closure")
    }
}
impl Debug for expr::Const {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Const {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("const_", &self.const_);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Const")
    }
}
impl Debug for expr::Continue {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Continue {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("continue_", &self.continue_);
                formatter.field("label", &self.label);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Continue")
    }
}
impl Debug for expr::Field {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Field {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("base", &self.base);
                formatter.field("dot", &self.dot);
                formatter.field("member", &self.memb);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Field")
    }
}
impl Debug for expr::ForLoop {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::ForLoop {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("label", &self.label);
                formatter.field("for_", &self.for_);
                formatter.field("pat", &self.pat);
                formatter.field("in_", &self.in_);
                formatter.field("expr", &self.expr);
                formatter.field("body", &self.body);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::ForLoop")
    }
}
impl Debug for expr::Group {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Group {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("group", &self.group);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Group")
    }
}
impl Debug for expr::If {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::If {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("if_", &self.if_);
                formatter.field("cond", &self.cond);
                formatter.field("then_branch", &self.then_branch);
                formatter.field("else_branch", &self.else_branch);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::If")
    }
}
impl Debug for expr::Index {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Index {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("expr", &self.expr);
                formatter.field("bracket", &self.bracket);
                formatter.field("index", &self.index);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Index")
    }
}
impl Debug for expr::Infer {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Infer {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("underscore", &self.underscore);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Infer")
    }
}
impl Debug for expr::Let {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Let {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("let_", &self.let_);
                formatter.field("pat", &self.pat);
                formatter.field("eq", &self.eq);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Let")
    }
}
impl Debug for expr::Lit {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Lit {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("lit", &self.lit);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Lit")
    }
}
impl Debug for expr::Loop {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Loop {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("label", &self.label);
                formatter.field("loop_", &self.loop_);
                formatter.field("body", &self.body);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Loop")
    }
}
impl Debug for expr::Mac {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Mac {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("mac", &self.mac);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Mac")
    }
}
impl Debug for expr::Match {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Match {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("match_", &self.match_);
                formatter.field("expr", &self.expr);
                formatter.field("brace", &self.brace);
                formatter.field("arms", &self.arms);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Match")
    }
}
impl Debug for expr::MethodCall {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::MethodCall {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("receiver", &self.expr);
                formatter.field("dot", &self.dot);
                formatter.field("method", &self.method);
                formatter.field("turbofish", &self.turbofish);
                formatter.field("paren", &self.paren);
                formatter.field("args", &self.args);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::MethodCall")
    }
}
impl Debug for expr::Paren {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Paren {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("paren", &self.paren);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Paren")
    }
}
impl Debug for expr::Pathth {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Pathth {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("qself", &self.qself);
                formatter.field("path", &self.path);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Pathth")
    }
}
impl Debug for expr::Range {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Range {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("start", &self.beg);
                formatter.field("limits", &self.limits);
                formatter.field("end", &self.end);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Range")
    }
}
impl Debug for expr::Ref {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Ref {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("and", &self.and);
                formatter.field("mut_", &self.mut_);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Ref")
    }
}
impl Debug for expr::Repeat {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Repeat {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("bracket", &self.bracket);
                formatter.field("expr", &self.expr);
                formatter.field("semi", &self.semi);
                formatter.field("len", &self.len);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Repeat")
    }
}
impl Debug for expr::Return {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Return {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("return_", &self.return_);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Return")
    }
}
impl Debug for expr::Struct {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Struct {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("qself", &self.qself);
                formatter.field("path", &self.path);
                formatter.field("brace", &self.brace);
                formatter.field("fields", &self.fields);
                formatter.field("dot2", &self.dot2);
                formatter.field("rest", &self.rest);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::ExprStruct")
    }
}
impl Debug for expr::Try {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Try {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("expr", &self.expr);
                formatter.field("question", &self.question);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Try")
    }
}
impl Debug for expr::TryBlock {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::TryBlock {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("try_", &self.try_);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::TryBlock")
    }
}
impl Debug for expr::Tuple {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Tuple {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("paren", &self.paren);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Tuple")
    }
}
impl Debug for expr::Unary {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Unary {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("op", &self.op);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Unary")
    }
}
impl Debug for expr::Unsafe {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Unsafe {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("unsafe_", &self.unsafe_);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Unsafe")
    }
}
impl Debug for expr::While {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::While {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("label", &self.label);
                formatter.field("while_", &self.while_);
                formatter.field("cond", &self.cond);
                formatter.field("body", &self.body);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::While")
    }
}
impl Debug for expr::Yield {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Yield {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("yield_", &self.yield_);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "expr::Yield")
    }
}
impl Debug for data::Field {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("data::Field");
        formatter.field("attrs", &self.attrs);
        formatter.field("vis", &self.vis);
        formatter.field("mut_", &self.mut_);
        formatter.field("ident", &self.ident);
        formatter.field("colon", &self.colon);
        formatter.field("ty", &self.typ);
        formatter.finish()
    }
}
impl Debug for data::Mut {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("FieldMutability::")?;
        match self {
            data::Mut::None => formatter.write_str("None"),
        }
    }
}
impl Debug for pat::Field {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("pat::Field");
        formatter.field("attrs", &self.attrs);
        formatter.field("member", &self.member);
        formatter.field("colon", &self.colon);
        formatter.field("pat", &self.pat);
        formatter.finish()
    }
}
impl Debug for FieldValue {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("FieldValue");
        formatter.field("attrs", &self.attrs);
        formatter.field("member", &self.member);
        formatter.field("colon", &self.colon);
        formatter.field("expr", &self.expr);
        formatter.finish()
    }
}
impl Debug for data::Fields {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("data::Fields::")?;
        match self {
            data::Fields::Named(v0) => v0.debug(formatter, "Named"),
            data::Fields::Unnamed(v0) => v0.debug(formatter, "Unnamed"),
            data::Fields::Unit => formatter.write_str("Unit"),
        }
    }
}
impl Debug for data::Named {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl data::Named {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("brace", &self.brace);
                formatter.field("named", &self.field);
                formatter.finish()
            }
        }
        self.debug(formatter, "data::Named")
    }
}
impl Debug for data::Unnamed {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl data::Unnamed {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("paren", &self.paren);
                formatter.field("unnamed", &self.field);
                formatter.finish()
            }
        }
        self.debug(formatter, "data::Unnamed")
    }
}
impl Debug for File {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("File");
        formatter.field("shebang", &self.shebang);
        formatter.field("attrs", &self.attrs);
        formatter.field("items", &self.items);
        formatter.finish()
    }
}
impl Debug for item::FnArg {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("item::FnArg::")?;
        match self {
            item::FnArg::Receiver(v0) => {
                let mut formatter = formatter.debug_tuple("item::Receiver");
                formatter.field(v0);
                formatter.finish()
            },
            item::FnArg::Type(v0) => {
                let mut formatter = formatter.debug_tuple("Typed");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for item::Foreign::Item {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("item::Foreign::Item::")?;
        match self {
            item::Foreign::Item::Fn(v0) => v0.debug(formatter, "Fn"),
            item::Foreign::Item::Static(v0) => v0.debug(formatter, "Static"),
            item::Foreign::Item::Type(v0) => v0.debug(formatter, "Type"),
            item::Foreign::Item::Macro(v0) => v0.debug(formatter, "Macro"),
            item::Foreign::Item::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for item::Foreign::Fn {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Foreign::Fn {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("sig", &self.sig);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Foreign::Fn")
    }
}
impl Debug for item::Foreign::Mac {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Foreign::Mac {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("mac", &self.mac);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Foreign::Mac")
    }
}
impl Debug for item::Foreign::Static {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Foreign::Static {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("static_", &self.static_);
                formatter.field("mut_", &self.mut_);
                formatter.field("ident", &self.ident);
                formatter.field("colon", &self.colon);
                formatter.field("ty", &self.typ);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Foreign::Static")
    }
}
impl Debug for item::Foreign::Type {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Foreign::Type {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("type", &self.type);
                formatter.field("ident", &self.ident);
                formatter.field("gens", &self.gens);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Foreign::Type")
    }
}
impl Debug for Arg {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("path::Arg::")?;
        match self {
            Arg::Life(v0) => {
                let mut formatter = formatter.debug_tuple("Life");
                formatter.field(v0);
                formatter.finish()
            },
            Arg::Type(v0) => {
                let mut formatter = formatter.debug_tuple("Type");
                formatter.field(v0);
                formatter.finish()
            },
            Arg::Const(v0) => {
                let mut formatter = formatter.debug_tuple("Const");
                formatter.field(v0);
                formatter.finish()
            },
            Arg::AssocType(v0) => {
                let mut formatter = formatter.debug_tuple("AssocType");
                formatter.field(v0);
                formatter.finish()
            },
            Arg::AssocConst(v0) => {
                let mut formatter = formatter.debug_tuple("AssocConst");
                formatter.field(v0);
                formatter.finish()
            },
            Arg::Constraint(v0) => {
                let mut formatter = formatter.debug_tuple("Constraint");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for gen::Param {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("gen::Param::")?;
        match self {
            gen::Param::Life(v0) => {
                let mut formatter = formatter.debug_tuple("Life");
                formatter.field(v0);
                formatter.finish()
            },
            gen::Param::Type(v0) => {
                let mut formatter = formatter.debug_tuple("Type");
                formatter.field(v0);
                formatter.finish()
            },
            gen::Param::Const(v0) => {
                let mut formatter = formatter.debug_tuple("Const");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for gen::Gens {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::Gens");
        formatter.field("lt", &self.lt);
        formatter.field("params", &self.params);
        formatter.field("gt", &self.gt);
        formatter.field("where_clause", &self.where_);
        formatter.finish()
    }
}
impl Debug for item::Impl::Item {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("item::Impl::Item::")?;
        match self {
            item::Impl::Item::Const(v0) => v0.debug(formatter, "Const"),
            item::Impl::Item::Fn(v0) => v0.debug(formatter, "Fn"),
            item::Impl::Item::Type(v0) => v0.debug(formatter, "Type"),
            item::Impl::Item::Macro(v0) => v0.debug(formatter, "Macro"),
            item::Impl::Item::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for item::Impl::Const {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Impl::Const {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("defaultness", &self.default_);
                formatter.field("const_", &self.const_);
                formatter.field("ident", &self.ident);
                formatter.field("gens", &self.gens);
                formatter.field("colon", &self.colon);
                formatter.field("ty", &self.typ);
                formatter.field("eq", &self.eq);
                formatter.field("expr", &self.expr);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Impl::Const")
    }
}
impl Debug for item::Impl::Fn {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Impl::Fn {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("defaultness", &self.default_);
                formatter.field("sig", &self.sig);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Impl::Fn")
    }
}
impl Debug for item::Impl::Mac {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Impl::Mac {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("mac", &self.mac);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Impl::Mac")
    }
}
impl Debug for item::Impl::Type {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Impl::Type {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("defaultness", &self.default_);
                formatter.field("type", &self.type);
                formatter.field("ident", &self.ident);
                formatter.field("gens", &self.gens);
                formatter.field("eq", &self.eq);
                formatter.field("ty", &self.typ);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Impl::Type")
    }
}
impl Debug for item::Impl::Restriction {
    fn fmt(&self, _formatter: &mut fmt::Formatter) -> fmt::Result {
        match *self {}
    }
}
impl Debug for Index {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Index");
        formatter.field("index", &self.index);
        formatter.field("span", &self.span);
        formatter.finish()
    }
}
impl Debug for Item {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Item::")?;
        match self {
            Item::Const(v0) => v0.debug(formatter, "Const"),
            Item::Enum(v0) => v0.debug(formatter, "Enum"),
            Item::ExternCrate(v0) => v0.debug(formatter, "ExternCrate"),
            Item::Fn(v0) => v0.debug(formatter, "Fn"),
            Item::Foreign(v0) => v0.debug(formatter, "Foreign"),
            Item::Impl(v0) => v0.debug(formatter, "Impl"),
            Item::Macro(v0) => v0.debug(formatter, "Macro"),
            Item::Mod(v0) => v0.debug(formatter, "Mod"),
            Item::Static(v0) => v0.debug(formatter, "Static"),
            Item::Struct(v0) => v0.debug(formatter, "Struct"),
            Item::Trait(v0) => v0.debug(formatter, "Trait"),
            Item::TraitAlias(v0) => v0.debug(formatter, "TraitAlias"),
            Item::Type(v0) => v0.debug(formatter, "Type"),
            Item::Union(v0) => v0.debug(formatter, "Union"),
            Item::Use(v0) => v0.debug(formatter, "Use"),
            Item::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for item::Const {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Const {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("const_", &self.const_);
                formatter.field("ident", &self.ident);
                formatter.field("gens", &self.gens);
                formatter.field("colon", &self.colon);
                formatter.field("ty", &self.typ);
                formatter.field("eq", &self.eq);
                formatter.field("expr", &self.expr);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Const")
    }
}
impl Debug for item::Enum {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Enum {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("enum_", &self.enum_);
                formatter.field("ident", &self.ident);
                formatter.field("gens", &self.gens);
                formatter.field("brace", &self.brace);
                formatter.field("variants", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Enum")
    }
}
impl Debug for item::ExternCrate {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::ExternCrate {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("extern_", &self.extern_);
                formatter.field("crate_", &self.crate_);
                formatter.field("ident", &self.ident);
                formatter.field("rename", &self.rename);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::ExternCrate")
    }
}
impl Debug for item::Fn {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Fn {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("sig", &self.sig);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Fn")
    }
}
impl Debug for item::Foreign {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Foreign {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("unsafe_", &self.unsafe_);
                formatter.field("abi", &self.abi);
                formatter.field("brace", &self.brace);
                formatter.field("items", &self.items);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Foreign")
    }
}
impl Debug for item::Impl {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Impl {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("defaultness", &self.default_);
                formatter.field("unsafe_", &self.unsafe_);
                formatter.field("impl_", &self.impl_);
                formatter.field("gens", &self.gens);
                formatter.field("trait_", &self.trait_);
                formatter.field("self_ty", &self.typ);
                formatter.field("brace", &self.brace);
                formatter.field("items", &self.items);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Impl")
    }
}
impl Debug for item::Mac {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Mac {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("ident", &self.ident);
                formatter.field("mac", &self.mac);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Mac")
    }
}
impl Debug for item::Mod {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Mod {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("unsafe_", &self.unsafe_);
                formatter.field("mod_", &self.mod_);
                formatter.field("ident", &self.ident);
                formatter.field("content", &self.gist);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Mod")
    }
}
impl Debug for item::Static {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Static {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("static_", &self.static_);
                formatter.field("mut_", &self.mut_);
                formatter.field("ident", &self.ident);
                formatter.field("colon", &self.colon);
                formatter.field("ty", &self.typ);
                formatter.field("eq", &self.eq);
                formatter.field("expr", &self.expr);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Static")
    }
}
impl Debug for item::Struct {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Struct {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("struct_", &self.struct_);
                formatter.field("ident", &self.ident);
                formatter.field("gens", &self.gens);
                formatter.field("fields", &self.fields);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Struct")
    }
}
impl Debug for item::Trait {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Trait {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("unsafe_", &self.unsafe_);
                formatter.field("auto_", &self.auto_);
                formatter.field("restriction", &self.restriction);
                formatter.field("trait_", &self.trait_);
                formatter.field("ident", &self.ident);
                formatter.field("gens", &self.gens);
                formatter.field("colon", &self.colon);
                formatter.field("supertraits", &self.supers);
                formatter.field("brace", &self.brace);
                formatter.field("items", &self.items);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Trait")
    }
}
impl Debug for item::TraitAlias {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::TraitAlias {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("trait_", &self.trait_);
                formatter.field("ident", &self.ident);
                formatter.field("gens", &self.gens);
                formatter.field("eq", &self.eq);
                formatter.field("bounds", &self.bounds);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::TraitAlias")
    }
}
impl Debug for item::Type {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Type {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("type", &self.type);
                formatter.field("ident", &self.ident);
                formatter.field("gens", &self.gens);
                formatter.field("eq", &self.eq);
                formatter.field("ty", &self.typ);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Type")
    }
}
impl Debug for item::Union {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Union {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("union_", &self.union_);
                formatter.field("ident", &self.ident);
                formatter.field("gens", &self.gens);
                formatter.field("fields", &self.fields);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Union")
    }
}
impl Debug for item::Use {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Use {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("use_", &self.use_);
                formatter.field("leading_colon", &self.colon);
                formatter.field("tree", &self.tree);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Use")
    }
}
impl Debug for Label {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Label");
        formatter.field("name", &self.name);
        formatter.field("colon", &self.colon);
        formatter.finish()
    }
}
impl Debug for Life {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl Life {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("apos", &self.apos);
                formatter.field("ident", &self.ident);
                formatter.finish()
            }
        }
        self.debug(formatter, "Life")
    }
}
impl Debug for gen::param::Life {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::param::Life");
        formatter.field("attrs", &self.attrs);
        formatter.field("life", &self.life);
        formatter.field("colon", &self.colon);
        formatter.field("bounds", &self.bounds);
        formatter.finish()
    }
}
impl Debug for Lit {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Lit::")?;
        match self {
            Lit::Str(v0) => v0.debug(formatter, "Str"),
            Lit::ByteStr(v0) => v0.debug(formatter, "ByteStr"),
            Lit::Byte(v0) => v0.debug(formatter, "Byte"),
            Lit::Char(v0) => v0.debug(formatter, "Char"),
            Lit::Int(v0) => v0.debug(formatter, "Int"),
            Lit::Float(v0) => v0.debug(formatter, "Float"),
            Lit::Bool(v0) => v0.debug(formatter, "Bool"),
            Lit::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for stmt::Local {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl stmt::Local {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("let_", &self.let_);
                formatter.field("pat", &self.pat);
                formatter.field("init", &self.init);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "stmt::Local")
    }
}
impl Debug for stmt::Init {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("stmt::Init");
        formatter.field("eq", &self.eq);
        formatter.field("expr", &self.expr);
        formatter.field("diverge", &self.diverge);
        formatter.finish()
    }
}
impl Debug for Macro {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Macro");
        formatter.field("path", &self.path);
        formatter.field("bang", &self.bang);
        formatter.field("delimiter", &self.delim);
        formatter.field("tokens", &self.toks);
        formatter.finish()
    }
}
impl Debug for tok::Delim {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("MacroDelimiter::")?;
        match self {
            tok::Delim::Paren(v0) => {
                let mut formatter = formatter.debug_tuple("Paren");
                formatter.field(v0);
                formatter.finish()
            },
            tok::Delim::Brace(v0) => {
                let mut formatter = formatter.debug_tuple("Brace");
                formatter.field(v0);
                formatter.finish()
            },
            tok::Delim::Bracket(v0) => {
                let mut formatter = formatter.debug_tuple("Bracket");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for Member {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Member::")?;
        match self {
            Member::Named(v0) => {
                let mut formatter = formatter.debug_tuple("Named");
                formatter.field(v0);
                formatter.finish()
            },
            Member::Unnamed(v0) => {
                let mut formatter = formatter.debug_tuple("Unnamed");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for meta::Meta {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("meta::Meta::")?;
        match self {
            meta::Meta::Path(v0) => v0.debug(formatter, "Path"),
            meta::Meta::List(v0) => v0.debug(formatter, "List"),
            meta::Meta::NameValue(v0) => v0.debug(formatter, "NameValue"),
        }
    }
}
impl Debug for meta::List {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl meta::List {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("path", &self.path);
                formatter.field("delimiter", &self.delim);
                formatter.field("tokens", &self.toks);
                formatter.finish()
            }
        }
        self.debug(formatter, "meta::List")
    }
}
impl Debug for meta::NameValue {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl meta::NameValue {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("path", &self.path);
                formatter.field("eq", &self.eq);
                formatter.field("value", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "meta::NameValue")
    }
}
impl Debug for ParenthesizedArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ParenthesizedArgs {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("paren", &self.paren);
                formatter.field("inputs", &self.ins);
                formatter.field("output", &self.out);
                formatter.finish()
            }
        }
        self.debug(formatter, "path::ParenthesizedArgs")
    }
}
impl Debug for pat::Pat {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("pat::Pat::")?;
        match self {
            pat::Pat::Const(v0) => v0.debug(formatter, "Const"),
            pat::Pat::Ident(v0) => v0.debug(formatter, "Ident"),
            pat::Pat::Lit(v0) => v0.debug(formatter, "Lit"),
            pat::Pat::Mac(v0) => v0.debug(formatter, "Macro"),
            pat::Pat::Or(v0) => v0.debug(formatter, "Or"),
            pat::Pat::Paren(v0) => v0.debug(formatter, "Paren"),
            pat::Pat::Path(v0) => v0.debug(formatter, "Path"),
            pat::Pat::Range(v0) => v0.debug(formatter, "Range"),
            pat::Pat::Ref(v0) => v0.debug(formatter, "Reference"),
            pat::Pat::Rest(v0) => v0.debug(formatter, "Rest"),
            pat::Pat::Slice(v0) => v0.debug(formatter, "Slice"),
            pat::Pat::Struct(v0) => v0.debug(formatter, "Struct"),
            pat::Pat::Tuple(v0) => v0.debug(formatter, "Tuple"),
            pat::Pat::TupleStruct(v0) => v0.debug(formatter, "TupleStruct"),
            pat::Pat::Type(v0) => v0.debug(formatter, "Type"),
            pat::Pat::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
            pat::Pat::Wild(v0) => v0.debug(formatter, "Wild"),
        }
    }
}
impl Debug for pat::Ident {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Ident {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("by_ref", &self.ref_);
                formatter.field("mut_", &self.mut_);
                formatter.field("ident", &self.ident);
                formatter.field("subpat", &self.sub);
                formatter.finish()
            }
        }
        self.debug(formatter, "pat::Ident")
    }
}
impl Debug for pat::Or {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Or {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("leading_vert", &self.vert);
                formatter.field("cases", &self.cases);
                formatter.finish()
            }
        }
        self.debug(formatter, "pat::Or")
    }
}
impl Debug for pat::Paren {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Paren {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("paren", &self.paren);
                formatter.field("pat", &self.pat);
                formatter.finish()
            }
        }
        self.debug(formatter, "pat::Paren")
    }
}
impl Debug for pat::Ref {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Ref {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("and", &self.and);
                formatter.field("mut_", &self.mut_);
                formatter.field("pat", &self.pat);
                formatter.finish()
            }
        }
        self.debug(formatter, "pat::Ref")
    }
}
impl Debug for pat::Rest {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Rest {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("dot2", &self.dot2);
                formatter.finish()
            }
        }
        self.debug(formatter, "pat::Rest")
    }
}
impl Debug for pat::Slice {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Slice {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("bracket", &self.bracket);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "pat::Slice")
    }
}
impl Debug for pat::Struct {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Struct {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("qself", &self.qself);
                formatter.field("path", &self.path);
                formatter.field("brace", &self.brace);
                formatter.field("fields", &self.fields);
                formatter.field("rest", &self.rest);
                formatter.finish()
            }
        }
        self.debug(formatter, "pat::Struct")
    }
}
impl Debug for pat::Tuple {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Tuple {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("paren", &self.paren);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "pat::Tuple")
    }
}
impl Debug for pat::TupleStructuct {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl pat::TupleStructuct {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("qself", &self.qself);
                formatter.field("path", &self.path);
                formatter.field("paren", &self.paren);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "pat::TupleStructuct")
    }
}
impl Debug for pat::Type {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Type {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("pat", &self.pat);
                formatter.field("colon", &self.colon);
                formatter.field("ty", &self.typ);
                formatter.finish()
            }
        }
        self.debug(formatter, "pat::Type")
    }
}
impl Debug for pat::Wild {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Wild {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("underscore", &self.underscore);
                formatter.finish()
            }
        }
        self.debug(formatter, "pat::Wild")
    }
}
impl Debug for Path {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl Path {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("leading_colon", &self.colon);
                formatter.field("segments", &self.segs);
                formatter.finish()
            }
        }
        self.debug(formatter, "Path")
    }
}
impl Debug for Args {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("path::Args::")?;
        match self {
            Args::None => formatter.write_str("None"),
            Args::Angled(v0) => v0.debug(formatter, "AngleBracketed"),
            Args::Parenthesized(v0) => v0.debug(formatter, "Parenthesized"),
        }
    }
}
impl Debug for Segment {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("path::Segment");
        formatter.field("ident", &self.ident);
        formatter.field("arguments", &self.args);
        formatter.finish()
    }
}
impl Debug for gen::Where::Life {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PredicateLifetime");
        formatter.field("life", &self.life);
        formatter.field("colon", &self.colon);
        formatter.field("bounds", &self.bounds);
        formatter.finish()
    }
}
impl Debug for gen::Where::Type {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PredicateType");
        formatter.field("lifetimes", &self.lifes);
        formatter.field("bounded_ty", &self.bounded);
        formatter.field("colon", &self.colon);
        formatter.field("bounds", &self.bounds);
        formatter.finish()
    }
}
impl Debug for QSelf {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("QSelf");
        formatter.field("lt", &self.lt);
        formatter.field("ty", &self.ty);
        formatter.field("position", &self.pos);
        formatter.field("as_", &self.as_);
        formatter.field("gt", &self.gt_);
        formatter.finish()
    }
}
impl Debug for RangeLimits {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("RangeLimits::")?;
        match self {
            RangeLimits::HalfOpen(v0) => {
                let mut formatter = formatter.debug_tuple("HalfOpen");
                formatter.field(v0);
                formatter.finish()
            },
            RangeLimits::Closed(v0) => {
                let mut formatter = formatter.debug_tuple("Closed");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for item::Receiver {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Receiver");
        formatter.field("attrs", &self.attrs);
        formatter.field("reference", &self.ref_);
        formatter.field("mut_", &self.mut_);
        formatter.field("self_", &self.self_);
        formatter.field("colon", &self.colon);
        formatter.field("ty", &self.typ);
        formatter.finish()
    }
}
impl Debug for typ::Ret {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("typ::Ret::")?;
        match self {
            typ::Ret::Default => formatter.write_str("Default"),
            typ::Ret::Type(v0, v1) => {
                let mut formatter = formatter.debug_tuple("Type");
                formatter.field(v0);
                formatter.field(v1);
                formatter.finish()
            },
        }
    }
}
impl Debug for item::Sig {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Sig");
        formatter.field("const_", &self.const_);
        formatter.field("asyncness", &self.async_);
        formatter.field("unsafe_", &self.unsafe_);
        formatter.field("abi", &self.abi);
        formatter.field("fn_", &self.fn_);
        formatter.field("ident", &self.ident);
        formatter.field("gens", &self.gens);
        formatter.field("paren", &self.paren);
        formatter.field("inputs", &self.args);
        formatter.field("vari", &self.vari);
        formatter.field("output", &self.ret);
        formatter.finish()
    }
}
impl Debug for StaticMut {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("StaticMutability::")?;
        match self {
            StaticMut::Mut(v0) => {
                let mut formatter = formatter.debug_tuple("Mut");
                formatter.field(v0);
                formatter.finish()
            },
            StaticMut::None => formatter.write_str("None"),
        }
    }
}
impl Debug for stmt::Stmt {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("stmt::Stmt::")?;
        match self {
            stmt::Stmt::stmt::Local(v0) => v0.debug(formatter, "stmt::Local"),
            stmt::Stmt::Item(v0) => {
                let mut formatter = formatter.debug_tuple("Item");
                formatter.field(v0);
                formatter.finish()
            },
            stmt::Stmt::Expr(v0, v1) => {
                let mut formatter = formatter.debug_tuple("Expr");
                formatter.field(v0);
                formatter.field(v1);
                formatter.finish()
            },
            stmt::Stmt::Mac(v0) => v0.debug(formatter, "Macro"),
        }
    }
}
impl Debug for stmt::Mac {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl stmt::Mac {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("mac", &self.mac);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "stmt::Mac")
    }
}
impl Debug for gen::bound::Trait {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::bound::Trait");
        formatter.field("paren", &self.paren);
        formatter.field("modifier", &self.modif);
        formatter.field("lifetimes", &self.lifes);
        formatter.field("path", &self.path);
        formatter.finish()
    }
}
impl Debug for gen::bound::Modifier {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("gen::bound::Modifier::")?;
        match self {
            gen::bound::Modifier::None => formatter.write_str("None"),
            gen::bound::Modifier::Maybe(v0) => {
                let mut formatter = formatter.debug_tuple("Maybe");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for item::Trait::Item {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("item::Trait::Item::")?;
        match self {
            item::Trait::Item::Const(v0) => v0.debug(formatter, "Const"),
            item::Trait::Item::Fn(v0) => v0.debug(formatter, "Fn"),
            item::Trait::Item::Type(v0) => v0.debug(formatter, "Type"),
            item::Trait::Item::Macro(v0) => v0.debug(formatter, "Macro"),
            item::Trait::Item::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for item::Trait::Const {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Trait::Const {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("const_", &self.const_);
                formatter.field("ident", &self.ident);
                formatter.field("gens", &self.gens);
                formatter.field("colon", &self.colon);
                formatter.field("ty", &self.typ);
                formatter.field("default", &self.default);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Trait::Const")
    }
}
impl Debug for item::Trait::Fn {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Trait::Fn {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("sig", &self.sig);
                formatter.field("default", &self.default);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Trait::Fn")
    }
}
impl Debug for item::Trait::Mac {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl item::Trait::Mac {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("mac", &self.mac);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "item::Trait::Mac")
    }
}
impl Debug for item::Trait::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Trait::Type {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut y = f.debug_struct(name);
                y.field("attrs", &self.attrs);
                y.field("type", &self.type);
                y.field("ident", &self.ident);
                y.field("gens", &self.gens);
                y.field("colon", &self.colon);
                y.field("bounds", &self.bounds);
                y.field("default", &self.default);
                y.field("semi", &self.semi);
                y.finish()
            }
        }
        self.debug(f, "item::Trait::Type")
    }
}
impl Debug for typ::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Type::")?;
        match self {
            typ::Type::Array(v0) => v0.debug(f, "Array"),
            typ::Type::Fn(v0) => v0.debug(f, "Fn"),
            typ::Type::Group(v0) => v0.debug(f, "Group"),
            typ::Type::Impl(v0) => v0.debug(f, "ImplTrait"),
            typ::Type::Infer(v0) => v0.debug(f, "Infer"),
            typ::Type::Mac(v0) => v0.debug(f, "Macro"),
            typ::Type::Never(v0) => v0.debug(f, "Never"),
            typ::Type::Paren(v0) => v0.debug(f, "Paren"),
            typ::Type::Path(v0) => v0.debug(f, "Path"),
            typ::Type::Ptr(v0) => v0.debug(f, "Ptr"),
            typ::Type::Ref(v0) => v0.debug(f, "Reference"),
            typ::Type::Slice(v0) => v0.debug(f, "Slice"),
            typ::Type::Trait(v0) => v0.debug(f, "TraitObject"),
            typ::Type::Tuple(v0) => v0.debug(f, "Tuple"),
            typ::Type::Verbatim(v0) => {
                let mut y = f.debug_tuple("Verbatim");
                y.field(v0);
                y.finish()
            },
        }
    }
}
impl Debug for typ::Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Array {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut y = f.debug_struct(name);
                y.field("bracket", &self.bracket);
                y.field("elem", &self.elem);
                y.field("semi", &self.semi);
                y.field("len", &self.len);
                y.finish()
            }
        }
        self.debug(f, "typ::Array")
    }
}
impl Debug for typ::Fn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Fn {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut y = f.debug_struct(name);
                y.field("lifetimes", &self.lifes);
                y.field("unsafe_", &self.unsafe_);
                y.field("abi", &self.abi);
                y.field("fn_", &self.fn_);
                y.field("paren", &self.paren);
                y.field("inputs", &self.args);
                y.field("vari", &self.vari);
                y.field("output", &self.ret);
                y.finish()
            }
        }
        self.debug(f, "typ::Fn")
    }
}
impl Debug for typ::Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Group {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut y = f.debug_struct(name);
                y.field("group", &self.group);
                y.field("elem", &self.elem);
                y.finish()
            }
        }
        self.debug(f, "typ::Group")
    }
}
impl Debug for typ::Impl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Impl {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut y = f.debug_struct(name);
                y.field("impl_", &self.impl_);
                y.field("bounds", &self.bounds);
                y.finish()
            }
        }
        self.debug(f, "typ::Impl")
    }
}
impl Debug for typ::Infer {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Infer {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("underscore", &self.underscore);
                formatter.finish()
            }
        }
        self.debug(formatter, "typ::Infer")
    }
}
impl Debug for typ::Mac {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Mac {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("mac", &self.mac);
                formatter.finish()
            }
        }
        self.debug(formatter, "typ::Mac")
    }
}
impl Debug for typ::Never {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Never {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("bang", &self.bang);
                formatter.finish()
            }
        }
        self.debug(formatter, "typ::Never")
    }
}
impl Debug for gen::param::Type {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::param::Type");
        formatter.field("attrs", &self.attrs);
        formatter.field("ident", &self.ident);
        formatter.field("colon", &self.colon);
        formatter.field("bounds", &self.bounds);
        formatter.field("eq", &self.eq);
        formatter.field("default", &self.default);
        formatter.finish()
    }
}
impl Debug for gen::bound::Type {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("gen::bound::Type::")?;
        match self {
            gen::bound::Type::Trait(v0) => {
                let mut formatter = formatter.debug_tuple("Trait");
                formatter.field(v0);
                formatter.finish()
            },
            gen::bound::Type::Life(v0) => v0.debug(formatter, "Life"),
            gen::bound::Type::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for typ::Paren {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Paren {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("paren", &self.paren);
                formatter.field("elem", &self.elem);
                formatter.finish()
            }
        }
        self.debug(formatter, "typ::Paren")
    }
}
impl Debug for typ::Path {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Path {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("qself", &self.qself);
                formatter.field("path", &self.path);
                formatter.finish()
            }
        }
        self.debug(formatter, "typ::Path")
    }
}
impl Debug for typ::Ptr {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Ptr {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("star", &self.star);
                formatter.field("const_", &self.const_);
                formatter.field("mut_", &self.mut_);
                formatter.field("elem", &self.elem);
                formatter.finish()
            }
        }
        self.debug(formatter, "typ::Ptr")
    }
}
impl Debug for typ::Ref {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Ref {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("and", &self.and);
                formatter.field("life", &self.life);
                formatter.field("mut_", &self.mut_);
                formatter.field("elem", &self.elem);
                formatter.finish()
            }
        }
        self.debug(formatter, "typ::Ref")
    }
}
impl Debug for typ::Slice {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Slice {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("bracket", &self.bracket);
                formatter.field("elem", &self.elem);
                formatter.finish()
            }
        }
        self.debug(formatter, "typ::Slice")
    }
}
impl Debug for typ::Trait {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Trait {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("dyn_", &self.dyn_);
                formatter.field("bounds", &self.bounds);
                formatter.finish()
            }
        }
        self.debug(formatter, "typ::Trait")
    }
}
impl Debug for typ::Tuple {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Tuple {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("paren", &self.paren);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "typ::Tuple")
    }
}
impl Debug for UnOp {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("UnOp::")?;
        match self {
            UnOp::Deref(v0) => {
                let mut formatter = formatter.debug_tuple("Deref");
                formatter.field(v0);
                formatter.finish()
            },
            UnOp::Not(v0) => {
                let mut formatter = formatter.debug_tuple("Not");
                formatter.field(v0);
                formatter.finish()
            },
            UnOp::Neg(v0) => {
                let mut formatter = formatter.debug_tuple("Neg");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for item::Use::Glob {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Use::Glob");
        formatter.field("star", &self.star);
        formatter.finish()
    }
}
impl Debug for item::Use::Group {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Use::Group");
        formatter.field("brace", &self.brace);
        formatter.field("items", &self.elems);
        formatter.finish()
    }
}
impl Debug for item::Use::Name {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Use::Name");
        formatter.field("ident", &self.ident);
        formatter.finish()
    }
}
impl Debug for item::Use::Path {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Use::Path");
        formatter.field("ident", &self.ident);
        formatter.field("colon2", &self.colon2);
        formatter.field("tree", &self.tree);
        formatter.finish()
    }
}
impl Debug for item::Use::Rename {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Use::Rename");
        formatter.field("ident", &self.ident);
        formatter.field("as_", &self.as_);
        formatter.field("rename", &self.rename);
        formatter.finish()
    }
}
impl Debug for item::Use::Tree {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("item::Use::Tree::")?;
        match self {
            item::Use::Tree::Path(v0) => {
                let mut formatter = formatter.debug_tuple("Path");
                formatter.field(v0);
                formatter.finish()
            },
            item::Use::Tree::Name(v0) => {
                let mut formatter = formatter.debug_tuple("Name");
                formatter.field(v0);
                formatter.finish()
            },
            item::Use::Tree::Rename(v0) => {
                let mut formatter = formatter.debug_tuple("Rename");
                formatter.field(v0);
                formatter.finish()
            },
            item::Use::Tree::Glob(v0) => {
                let mut formatter = formatter.debug_tuple("Glob");
                formatter.field(v0);
                formatter.finish()
            },
            item::Use::Tree::Group(v0) => {
                let mut formatter = formatter.debug_tuple("Group");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for item::Variadic {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Variadic");
        formatter.field("attrs", &self.attrs);
        formatter.field("pat", &self.pat);
        formatter.field("dots", &self.dots);
        formatter.field("comma", &self.comma);
        formatter.finish()
    }
}
impl Debug for data::Variant {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("data::Variant");
        formatter.field("attrs", &self.attrs);
        formatter.field("ident", &self.ident);
        formatter.field("fields", &self.fields);
        formatter.field("discriminant", &self.discriminant);
        formatter.finish()
    }
}
impl Debug for VisRestricted {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl VisRestricted {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("pub_", &self.pub_);
                formatter.field("paren", &self.paren);
                formatter.field("in_", &self.in_);
                formatter.field("path", &self.path);
                formatter.finish()
            }
        }
        self.debug(formatter, "VisRestricted")
    }
}
impl Debug for Visibility {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Visibility::")?;
        match self {
            Visibility::Public(v0) => {
                let mut formatter = formatter.debug_tuple("Public");
                formatter.field(v0);
                formatter.finish()
            },
            Visibility::Restricted(v0) => v0.debug(formatter, "Restricted"),
            Visibility::Inherited => formatter.write_str("Inherited"),
        }
    }
}
impl Debug for gen::Where {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::Where");
        formatter.field("where_", &self.where_);
        formatter.field("predicates", &self.preds);
        formatter.finish()
    }
}
impl Debug for gen::Where::Pred {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("WherePredicate::")?;
        match self {
            gen::Where::Pred::Life(v0) => {
                let mut formatter = formatter.debug_tuple("Life");
                formatter.field(v0);
                formatter.finish()
            },
            gen::Where::Pred::Type(v0) => {
                let mut formatter = formatter.debug_tuple("Type");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
