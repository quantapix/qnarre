use crate::*;
use std::fmt::{self, Debug};
impl Debug for Abi {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Abi");
        formatter.field("extern_token", &self.extern_);
        formatter.field("name", &self.name);
        formatter.finish()
    }
}
impl Debug for AngleBracketedGenericArguments {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl AngleBracketedGenericArguments {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("colon2_token", &self.colon2_token);
                formatter.field("lt_token", &self.lt);
                formatter.field("args", &self.args);
                formatter.field("gt_token", &self.gt);
                formatter.finish()
            }
        }
        self.debug(formatter, "AngleBracketedGenericArguments")
    }
}
impl Debug for Arm {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Arm");
        formatter.field("attrs", &self.attrs);
        formatter.field("pat", &self.pat);
        formatter.field("guard", &self.guard);
        formatter.field("fat_arrow_token", &self.fat_arrow_token);
        formatter.field("body", &self.body);
        formatter.field("comma", &self.comma);
        formatter.finish()
    }
}
impl Debug for AssocConst {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("AssocConst");
        formatter.field("ident", &self.ident);
        formatter.field("generics", &self.generics);
        formatter.field("eq_token", &self.equal);
        formatter.field("value", &self.value);
        formatter.finish()
    }
}
impl Debug for AssocType {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("AssocType");
        formatter.field("ident", &self.ident);
        formatter.field("generics", &self.generics);
        formatter.field("eq_token", &self.equal);
        formatter.field("ty", &self.ty);
        formatter.finish()
    }
}
impl Debug for AttrStyle {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("AttrStyle::")?;
        match self {
            AttrStyle::Outer => formatter.write_str("Outer"),
            AttrStyle::Inner(v0) => {
                let mut formatter = formatter.debug_tuple("Inner");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for Attribute {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Attribute");
        formatter.field("pound_token", &self.pound);
        formatter.field("style", &self.style);
        formatter.field("bracket_token", &self.bracket);
        formatter.field("meta", &self.meta);
        formatter.finish()
    }
}
impl Debug for BareFnArg {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("BareFnArg");
        formatter.field("attrs", &self.attrs);
        formatter.field("name", &self.name);
        formatter.field("ty", &self.ty);
        formatter.finish()
    }
}
impl Debug for BareVariadic {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("BareVariadic");
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
        formatter.field("brace_token", &self.brace);
        formatter.field("stmts", &self.stmts);
        formatter.finish()
    }
}
impl Debug for BoundLifetimes {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("BoundLifetimes");
        formatter.field("for_token", &self.for_);
        formatter.field("lt_token", &self.lt);
        formatter.field("lifetimes", &self.lifetimes);
        formatter.field("gt_token", &self.gt);
        formatter.finish()
    }
}
impl Debug for ConstParam {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ConstParam");
        formatter.field("attrs", &self.attrs);
        formatter.field("const_token", &self.const_);
        formatter.field("ident", &self.ident);
        formatter.field("colon_token", &self.colon);
        formatter.field("ty", &self.ty);
        formatter.field("eq_token", &self.equal);
        formatter.field("default", &self.default);
        formatter.finish()
    }
}
impl Debug for Constraint {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Constraint");
        formatter.field("ident", &self.ident);
        formatter.field("generics", &self.generics);
        formatter.field("colon_token", &self.colon);
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
impl Debug for DataEnum {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl DataEnum {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("enum_token", &self.enum_);
                formatter.field("brace_token", &self.brace);
                formatter.field("variants", &self.variants);
                formatter.finish()
            }
        }
        self.debug(formatter, "DataEnum")
    }
}
impl Debug for DataStruct {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl DataStruct {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("struct_token", &self.struct_);
                formatter.field("fields", &self.fields);
                formatter.field("semi_token", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "DataStruct")
    }
}
impl Debug for DataUnion {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl DataUnion {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("union_token", &self.union_);
                formatter.field("fields", &self.fields);
                formatter.finish()
            }
        }
        self.debug(formatter, "DataUnion")
    }
}
impl Debug for DeriveInput {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("DeriveInput");
        formatter.field("attrs", &self.attrs);
        formatter.field("vis", &self.vis);
        formatter.field("ident", &self.ident);
        formatter.field("generics", &self.generics);
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
impl Debug for ExprArray {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprArray {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("bracket_token", &self.bracket_token);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprArray")
    }
}
impl Debug for ExprAssign {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprAssign {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("left", &self.left);
                formatter.field("eq_token", &self.eq_token);
                formatter.field("right", &self.right);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprAssign")
    }
}
impl Debug for ExprAsync {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprAsync {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("async_token", &self.async_token);
                formatter.field("capture", &self.capture);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprAsync")
    }
}
impl Debug for ExprAwait {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprAwait {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("base", &self.base);
                formatter.field("dot_token", &self.dot_token);
                formatter.field("await_token", &self.await_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprAwait")
    }
}
impl Debug for ExprBinary {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprBinary {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("left", &self.left);
                formatter.field("op", &self.op);
                formatter.field("right", &self.right);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprBinary")
    }
}
impl Debug for ExprBlock {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprBlock {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("label", &self.label);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprBlock")
    }
}
impl Debug for ExprBreak {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprBreak {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("break_token", &self.break_token);
                formatter.field("label", &self.label);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprBreak")
    }
}
impl Debug for ExprCall {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprCall {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("func", &self.func);
                formatter.field("paren_token", &self.paren_token);
                formatter.field("args", &self.args);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprCall")
    }
}
impl Debug for ExprCast {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprCast {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("expr", &self.expr);
                formatter.field("as_token", &self.as_token);
                formatter.field("ty", &self.ty);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprCast")
    }
}
impl Debug for ExprClosure {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprClosure {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("lifetimes", &self.lifetimes);
                formatter.field("constness", &self.constness);
                formatter.field("movability", &self.movability);
                formatter.field("asyncness", &self.asyncness);
                formatter.field("capture", &self.capture);
                formatter.field("or1_token", &self.or1_token);
                formatter.field("inputs", &self.inputs);
                formatter.field("or2_token", &self.or2_token);
                formatter.field("output", &self.output);
                formatter.field("body", &self.body);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprClosure")
    }
}
impl Debug for ExprConst {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprConst {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("const_token", &self.const_token);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprConst")
    }
}
impl Debug for ExprContinue {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprContinue {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("continue_token", &self.continue_token);
                formatter.field("label", &self.label);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprContinue")
    }
}
impl Debug for ExprField {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprField {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("base", &self.base);
                formatter.field("dot_token", &self.dot_token);
                formatter.field("member", &self.member);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprField")
    }
}
impl Debug for ExprForLoop {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprForLoop {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("label", &self.label);
                formatter.field("for_token", &self.for_token);
                formatter.field("pat", &self.pat);
                formatter.field("in_token", &self.in_token);
                formatter.field("expr", &self.expr);
                formatter.field("body", &self.body);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprForLoop")
    }
}
impl Debug for ExprGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprGroup {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("group_token", &self.group_token);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprGroup")
    }
}
impl Debug for ExprIf {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprIf {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("if_token", &self.if_token);
                formatter.field("cond", &self.cond);
                formatter.field("then_branch", &self.then_branch);
                formatter.field("else_branch", &self.else_branch);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprIf")
    }
}
impl Debug for ExprIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprIndex {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("expr", &self.expr);
                formatter.field("bracket_token", &self.bracket_token);
                formatter.field("index", &self.index);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprIndex")
    }
}
impl Debug for ExprInfer {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprInfer {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("underscore_token", &self.underscore_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprInfer")
    }
}
impl Debug for ExprLet {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprLet {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("let_token", &self.let_token);
                formatter.field("pat", &self.pat);
                formatter.field("eq_token", &self.eq_token);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprLet")
    }
}
impl Debug for ExprLit {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprLit {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("lit", &self.lit);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprLit")
    }
}
impl Debug for ExprLoop {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprLoop {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("label", &self.label);
                formatter.field("loop_token", &self.loop_token);
                formatter.field("body", &self.body);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprLoop")
    }
}
impl Debug for ExprMacro {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprMacro {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("mac", &self.mac);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprMacro")
    }
}
impl Debug for ExprMatch {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprMatch {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("match_token", &self.match_token);
                formatter.field("expr", &self.expr);
                formatter.field("brace_token", &self.brace_token);
                formatter.field("arms", &self.arms);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprMatch")
    }
}
impl Debug for ExprMethodCall {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprMethodCall {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("receiver", &self.receiver);
                formatter.field("dot_token", &self.dot_token);
                formatter.field("method", &self.method);
                formatter.field("turbofish", &self.turbofish);
                formatter.field("paren_token", &self.paren_token);
                formatter.field("args", &self.args);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprMethodCall")
    }
}
impl Debug for ExprParen {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprParen {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("paren_token", &self.paren_token);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprParen")
    }
}
impl Debug for ExprPath {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprPath {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("qself", &self.qself);
                formatter.field("path", &self.path);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprPath")
    }
}
impl Debug for ExprRange {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprRange {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("start", &self.start);
                formatter.field("limits", &self.limits);
                formatter.field("end", &self.end);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprRange")
    }
}
impl Debug for ExprReference {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprReference {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("and_token", &self.and_token);
                formatter.field("mutability", &self.mutability);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprReference")
    }
}
impl Debug for ExprRepeat {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprRepeat {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("bracket_token", &self.bracket_token);
                formatter.field("expr", &self.expr);
                formatter.field("semi_token", &self.semi_token);
                formatter.field("len", &self.len);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprRepeat")
    }
}
impl Debug for ExprReturn {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprReturn {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("return_token", &self.return_token);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprReturn")
    }
}
impl Debug for ExprStruct {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprStruct {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("qself", &self.qself);
                formatter.field("path", &self.path);
                formatter.field("brace_token", &self.brace_token);
                formatter.field("fields", &self.fields);
                formatter.field("dot2_token", &self.dot2_token);
                formatter.field("rest", &self.rest);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprStruct")
    }
}
impl Debug for ExprTry {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprTry {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("expr", &self.expr);
                formatter.field("question_token", &self.question_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprTry")
    }
}
impl Debug for ExprTryBlock {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprTryBlock {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("try_token", &self.try_token);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprTryBlock")
    }
}
impl Debug for ExprTuple {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprTuple {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("paren_token", &self.paren_token);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprTuple")
    }
}
impl Debug for ExprUnary {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprUnary {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("op", &self.op);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprUnary")
    }
}
impl Debug for ExprUnsafe {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprUnsafe {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("unsafe_token", &self.unsafe_token);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprUnsafe")
    }
}
impl Debug for ExprWhile {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprWhile {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("label", &self.label);
                formatter.field("while_token", &self.while_token);
                formatter.field("cond", &self.cond);
                formatter.field("body", &self.body);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprWhile")
    }
}
impl Debug for ExprYield {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ExprYield {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("yield_token", &self.yield_token);
                formatter.field("expr", &self.expr);
                formatter.finish()
            }
        }
        self.debug(formatter, "ExprYield")
    }
}
impl Debug for Field {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Field");
        formatter.field("attrs", &self.attrs);
        formatter.field("vis", &self.vis);
        formatter.field("mutability", &self.mutability);
        formatter.field("ident", &self.ident);
        formatter.field("colon_token", &self.colon_token);
        formatter.field("ty", &self.ty);
        formatter.finish()
    }
}
impl Debug for FieldMut {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("FieldMutability::")?;
        match self {
            FieldMut::None => formatter.write_str("None"),
        }
    }
}
impl Debug for FieldPat {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("FieldPat");
        formatter.field("attrs", &self.attrs);
        formatter.field("member", &self.member);
        formatter.field("colon_token", &self.colon);
        formatter.field("pat", &self.pat);
        formatter.finish()
    }
}
impl Debug for FieldValue {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("FieldValue");
        formatter.field("attrs", &self.attrs);
        formatter.field("member", &self.member);
        formatter.field("colon_token", &self.colon_token);
        formatter.field("expr", &self.expr);
        formatter.finish()
    }
}
impl Debug for Fields {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Fields::")?;
        match self {
            Fields::Named(v0) => v0.debug(formatter, "Named"),
            Fields::Unnamed(v0) => v0.debug(formatter, "Unnamed"),
            Fields::Unit => formatter.write_str("Unit"),
        }
    }
}
impl Debug for FieldsNamed {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl FieldsNamed {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("brace_token", &self.brace_token);
                formatter.field("named", &self.named);
                formatter.finish()
            }
        }
        self.debug(formatter, "FieldsNamed")
    }
}
impl Debug for FieldsUnnamed {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl FieldsUnnamed {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("paren_token", &self.paren_token);
                formatter.field("unnamed", &self.unnamed);
                formatter.finish()
            }
        }
        self.debug(formatter, "FieldsUnnamed")
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
impl Debug for FnArg {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("FnArg::")?;
        match self {
            FnArg::Receiver(v0) => {
                let mut formatter = formatter.debug_tuple("Receiver");
                formatter.field(v0);
                formatter.finish()
            },
            FnArg::Typed(v0) => {
                let mut formatter = formatter.debug_tuple("Typed");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for ForeignItem {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("ForeignItem::")?;
        match self {
            ForeignItem::Fn(v0) => v0.debug(formatter, "Fn"),
            ForeignItem::Static(v0) => v0.debug(formatter, "Static"),
            ForeignItem::Type(v0) => v0.debug(formatter, "Type"),
            ForeignItem::Macro(v0) => v0.debug(formatter, "Macro"),
            ForeignItem::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for ForeignItemFn {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ForeignItemFn {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("sig", &self.sig);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ForeignItemFn")
    }
}
impl Debug for ForeignItemMacro {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ForeignItemMacro {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("mac", &self.mac);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ForeignItemMacro")
    }
}
impl Debug for ForeignItemStatic {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ForeignItemStatic {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("static_token", &self.static_token);
                formatter.field("mutability", &self.mutability);
                formatter.field("ident", &self.ident);
                formatter.field("colon_token", &self.colon_token);
                formatter.field("ty", &self.ty);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ForeignItemStatic")
    }
}
impl Debug for ForeignItemType {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ForeignItemType {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("type_token", &self.type_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ForeignItemType")
    }
}
impl Debug for GenericArgument {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("GenericArgument::")?;
        match self {
            GenericArgument::Lifetime(v0) => {
                let mut formatter = formatter.debug_tuple("Lifetime");
                formatter.field(v0);
                formatter.finish()
            },
            GenericArgument::Type(v0) => {
                let mut formatter = formatter.debug_tuple("Type");
                formatter.field(v0);
                formatter.finish()
            },
            GenericArgument::Const(v0) => {
                let mut formatter = formatter.debug_tuple("Const");
                formatter.field(v0);
                formatter.finish()
            },
            GenericArgument::AssocType(v0) => {
                let mut formatter = formatter.debug_tuple("AssocType");
                formatter.field(v0);
                formatter.finish()
            },
            GenericArgument::AssocConst(v0) => {
                let mut formatter = formatter.debug_tuple("AssocConst");
                formatter.field(v0);
                formatter.finish()
            },
            GenericArgument::Constraint(v0) => {
                let mut formatter = formatter.debug_tuple("Constraint");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for GenericParam {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("GenericParam::")?;
        match self {
            GenericParam::Lifetime(v0) => {
                let mut formatter = formatter.debug_tuple("Lifetime");
                formatter.field(v0);
                formatter.finish()
            },
            GenericParam::Type(v0) => {
                let mut formatter = formatter.debug_tuple("Type");
                formatter.field(v0);
                formatter.finish()
            },
            GenericParam::Const(v0) => {
                let mut formatter = formatter.debug_tuple("Const");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for Generics {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Generics");
        formatter.field("lt_token", &self.lt);
        formatter.field("params", &self.params);
        formatter.field("gt_token", &self.gt);
        formatter.field("where_clause", &self.clause);
        formatter.finish()
    }
}
impl Debug for ImplItem {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("ImplItem::")?;
        match self {
            ImplItem::Const(v0) => v0.debug(formatter, "Const"),
            ImplItem::Fn(v0) => v0.debug(formatter, "Fn"),
            ImplItem::Type(v0) => v0.debug(formatter, "Type"),
            ImplItem::Macro(v0) => v0.debug(formatter, "Macro"),
            ImplItem::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for ImplItemConst {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ImplItemConst {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("defaultness", &self.defaultness);
                formatter.field("const_token", &self.const_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("colon_token", &self.colon_token);
                formatter.field("ty", &self.ty);
                formatter.field("eq_token", &self.eq_token);
                formatter.field("expr", &self.expr);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ImplItemConst")
    }
}
impl Debug for ImplItemFn {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ImplItemFn {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("defaultness", &self.defaultness);
                formatter.field("sig", &self.sig);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "ImplItemFn")
    }
}
impl Debug for ImplItemMacro {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ImplItemMacro {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("mac", &self.mac);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ImplItemMacro")
    }
}
impl Debug for ImplItemType {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ImplItemType {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("defaultness", &self.defaultness);
                formatter.field("type_token", &self.type_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("eq_token", &self.eq_token);
                formatter.field("ty", &self.ty);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ImplItemType")
    }
}
impl Debug for ImplRestriction {
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
            Item::ForeignMod(v0) => v0.debug(formatter, "ForeignMod"),
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
impl Debug for ItemConst {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemConst {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("const_token", &self.const_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("colon_token", &self.colon_token);
                formatter.field("ty", &self.ty);
                formatter.field("eq_token", &self.eq_token);
                formatter.field("expr", &self.expr);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemConst")
    }
}
impl Debug for ItemEnum {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemEnum {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("enum_token", &self.enum_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("brace_token", &self.brace_token);
                formatter.field("variants", &self.variants);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemEnum")
    }
}
impl Debug for ItemExternCrate {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemExternCrate {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("extern_token", &self.extern_token);
                formatter.field("crate_token", &self.crate_token);
                formatter.field("ident", &self.ident);
                formatter.field("rename", &self.rename);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemExternCrate")
    }
}
impl Debug for ItemFn {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemFn {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("sig", &self.sig);
                formatter.field("block", &self.block);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemFn")
    }
}
impl Debug for ItemForeignMod {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemForeignMod {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("unsafety", &self.unsafety);
                formatter.field("abi", &self.abi);
                formatter.field("brace_token", &self.brace_token);
                formatter.field("items", &self.items);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemForeignMod")
    }
}
impl Debug for ItemImpl {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemImpl {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("defaultness", &self.defaultness);
                formatter.field("unsafety", &self.unsafety);
                formatter.field("impl_token", &self.impl_token);
                formatter.field("generics", &self.generics);
                formatter.field("trait_", &self.trait_);
                formatter.field("self_ty", &self.self_ty);
                formatter.field("brace_token", &self.brace_token);
                formatter.field("items", &self.items);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemImpl")
    }
}
impl Debug for ItemMacro {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemMacro {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("ident", &self.ident);
                formatter.field("mac", &self.mac);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemMacro")
    }
}
impl Debug for ItemMod {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemMod {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("unsafety", &self.unsafety);
                formatter.field("mod_token", &self.mod_token);
                formatter.field("ident", &self.ident);
                formatter.field("content", &self.content);
                formatter.field("semi", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemMod")
    }
}
impl Debug for ItemStatic {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemStatic {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("static_token", &self.static_token);
                formatter.field("mutability", &self.mutability);
                formatter.field("ident", &self.ident);
                formatter.field("colon_token", &self.colon_token);
                formatter.field("ty", &self.ty);
                formatter.field("eq_token", &self.eq_token);
                formatter.field("expr", &self.expr);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemStatic")
    }
}
impl Debug for ItemStruct {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemStruct {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("struct_token", &self.struct_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("fields", &self.fields);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemStruct")
    }
}
impl Debug for ItemTrait {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemTrait {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("unsafety", &self.unsafety);
                formatter.field("auto_token", &self.auto_token);
                formatter.field("restriction", &self.restriction);
                formatter.field("trait_token", &self.trait_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("colon_token", &self.colon_token);
                formatter.field("supertraits", &self.supertraits);
                formatter.field("brace_token", &self.brace_token);
                formatter.field("items", &self.items);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemTrait")
    }
}
impl Debug for ItemTraitAlias {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemTraitAlias {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("trait_token", &self.trait_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("eq_token", &self.eq_token);
                formatter.field("bounds", &self.bounds);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemTraitAlias")
    }
}
impl Debug for ItemType {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemType {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("type_token", &self.type_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("eq_token", &self.eq_token);
                formatter.field("ty", &self.ty);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemType")
    }
}
impl Debug for ItemUnion {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemUnion {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("union_token", &self.union_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("fields", &self.fields);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemUnion")
    }
}
impl Debug for ItemUse {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ItemUse {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("vis", &self.vis);
                formatter.field("use_token", &self.use_token);
                formatter.field("leading_colon", &self.leading_colon);
                formatter.field("tree", &self.tree);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "ItemUse")
    }
}
impl Debug for Label {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Label");
        formatter.field("name", &self.name);
        formatter.field("colon_token", &self.colon_token);
        formatter.finish()
    }
}
impl Debug for Lifetime {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl Lifetime {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("apostrophe", &self.apostrophe);
                formatter.field("ident", &self.ident);
                formatter.finish()
            }
        }
        self.debug(formatter, "Lifetime")
    }
}
impl Debug for LifetimeParam {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("LifetimeParam");
        formatter.field("attrs", &self.attrs);
        formatter.field("lifetime", &self.lifetime);
        formatter.field("colon_token", &self.colon);
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
impl Debug for Local {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl Local {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("let_token", &self.let_);
                formatter.field("pat", &self.pat);
                formatter.field("init", &self.init);
                formatter.field("semi_token", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "Local")
    }
}
impl Debug for LocalInit {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("LocalInit");
        formatter.field("eq_token", &self.equal);
        formatter.field("expr", &self.expr);
        formatter.field("diverge", &self.diverge);
        formatter.finish()
    }
}
impl Debug for Macro {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Macro");
        formatter.field("path", &self.path);
        formatter.field("bang_token", &self.bang);
        formatter.field("delimiter", &self.delim);
        formatter.field("tokens", &self.toks);
        formatter.finish()
    }
}
impl Debug for MacroDelimiter {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("MacroDelimiter::")?;
        match self {
            MacroDelimiter::Paren(v0) => {
                let mut formatter = formatter.debug_tuple("Paren");
                formatter.field(v0);
                formatter.finish()
            },
            MacroDelimiter::Brace(v0) => {
                let mut formatter = formatter.debug_tuple("Brace");
                formatter.field(v0);
                formatter.finish()
            },
            MacroDelimiter::Bracket(v0) => {
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
impl Debug for Meta {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Meta::")?;
        match self {
            Meta::Path(v0) => v0.debug(formatter, "Path"),
            Meta::List(v0) => v0.debug(formatter, "List"),
            Meta::NameValue(v0) => v0.debug(formatter, "NameValue"),
        }
    }
}
impl Debug for MetaList {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl MetaList {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("path", &self.path);
                formatter.field("delimiter", &self.delim);
                formatter.field("tokens", &self.toks);
                formatter.finish()
            }
        }
        self.debug(formatter, "MetaList")
    }
}
impl Debug for MetaNameValue {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl MetaNameValue {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("path", &self.path);
                formatter.field("eq_token", &self.equal);
                formatter.field("value", &self.val);
                formatter.finish()
            }
        }
        self.debug(formatter, "MetaNameValue")
    }
}
impl Debug for ParenthesizedGenericArguments {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl ParenthesizedGenericArguments {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("paren_token", &self.paren);
                formatter.field("inputs", &self.inputs);
                formatter.field("output", &self.output);
                formatter.finish()
            }
        }
        self.debug(formatter, "ParenthesizedGenericArguments")
    }
}
impl Debug for Pat {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Pat::")?;
        match self {
            Pat::Const(v0) => v0.debug(formatter, "Const"),
            Pat::Ident(v0) => v0.debug(formatter, "Ident"),
            Pat::Lit(v0) => v0.debug(formatter, "Lit"),
            Pat::Macro(v0) => v0.debug(formatter, "Macro"),
            Pat::Or(v0) => v0.debug(formatter, "Or"),
            Pat::Paren(v0) => v0.debug(formatter, "Paren"),
            Pat::Path(v0) => v0.debug(formatter, "Path"),
            Pat::Range(v0) => v0.debug(formatter, "Range"),
            Pat::Reference(v0) => v0.debug(formatter, "Reference"),
            Pat::Rest(v0) => v0.debug(formatter, "Rest"),
            Pat::Slice(v0) => v0.debug(formatter, "Slice"),
            Pat::Struct(v0) => v0.debug(formatter, "Struct"),
            Pat::Tuple(v0) => v0.debug(formatter, "Tuple"),
            Pat::TupleStruct(v0) => v0.debug(formatter, "TupleStruct"),
            Pat::Type(v0) => v0.debug(formatter, "Type"),
            Pat::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
            Pat::Wild(v0) => v0.debug(formatter, "Wild"),
        }
    }
}
impl Debug for PatIdent {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl PatIdent {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("by_ref", &self.ref_);
                formatter.field("mutability", &self.mut_);
                formatter.field("ident", &self.ident);
                formatter.field("subpat", &self.subpat);
                formatter.finish()
            }
        }
        self.debug(formatter, "PatIdent")
    }
}
impl Debug for PatOr {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl PatOr {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("leading_vert", &self.leading_vert);
                formatter.field("cases", &self.cases);
                formatter.finish()
            }
        }
        self.debug(formatter, "PatOr")
    }
}
impl Debug for PatParen {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl PatParen {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("paren_token", &self.paren);
                formatter.field("pat", &self.pat);
                formatter.finish()
            }
        }
        self.debug(formatter, "PatParen")
    }
}
impl Debug for PatReference {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl PatReference {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("and_token", &self.and_);
                formatter.field("mutability", &self.mutability);
                formatter.field("pat", &self.pat);
                formatter.finish()
            }
        }
        self.debug(formatter, "PatReference")
    }
}
impl Debug for PatRest {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl PatRest {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("dot2_token", &self.dot2);
                formatter.finish()
            }
        }
        self.debug(formatter, "PatRest")
    }
}
impl Debug for PatSlice {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl PatSlice {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("bracket_token", &self.bracket);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "PatSlice")
    }
}
impl Debug for PatStruct {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl PatStruct {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("qself", &self.qself);
                formatter.field("path", &self.path);
                formatter.field("brace_token", &self.brace);
                formatter.field("fields", &self.fields);
                formatter.field("rest", &self.rest);
                formatter.finish()
            }
        }
        self.debug(formatter, "PatStruct")
    }
}
impl Debug for PatTuple {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl PatTuple {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("paren_token", &self.paren);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "PatTuple")
    }
}
impl Debug for PatTupleStruct {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl PatTupleStruct {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("qself", &self.qself);
                formatter.field("path", &self.path);
                formatter.field("paren_token", &self.paren);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "PatTupleStruct")
    }
}
impl Debug for PatType {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl PatType {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("pat", &self.pat);
                formatter.field("colon_token", &self.colon);
                formatter.field("ty", &self.ty);
                formatter.finish()
            }
        }
        self.debug(formatter, "PatType")
    }
}
impl Debug for PatWild {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl PatWild {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("underscore_token", &self.underscore);
                formatter.finish()
            }
        }
        self.debug(formatter, "PatWild")
    }
}
impl Debug for Path {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl Path {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("leading_colon", &self.leading_colon);
                formatter.field("segments", &self.segments);
                formatter.finish()
            }
        }
        self.debug(formatter, "Path")
    }
}
impl Debug for PathArguments {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("PathArguments::")?;
        match self {
            PathArguments::None => formatter.write_str("None"),
            PathArguments::AngleBracketed(v0) => v0.debug(formatter, "AngleBracketed"),
            PathArguments::Parenthesized(v0) => v0.debug(formatter, "Parenthesized"),
        }
    }
}
impl Debug for PathSegment {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PathSegment");
        formatter.field("ident", &self.ident);
        formatter.field("arguments", &self.arguments);
        formatter.finish()
    }
}
impl Debug for PredLifetime {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PredicateLifetime");
        formatter.field("lifetime", &self.lifetime);
        formatter.field("colon_token", &self.colon);
        formatter.field("bounds", &self.bounds);
        formatter.finish()
    }
}
impl Debug for PredType {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PredicateType");
        formatter.field("lifetimes", &self.lifetimes);
        formatter.field("bounded_ty", &self.bounded_ty);
        formatter.field("colon_token", &self.colon);
        formatter.field("bounds", &self.bounds);
        formatter.finish()
    }
}
impl Debug for QSelf {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("QSelf");
        formatter.field("lt_token", &self.lt);
        formatter.field("ty", &self.ty);
        formatter.field("position", &self.position);
        formatter.field("as_token", &self.as_);
        formatter.field("gt_token", &self.gt_);
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
impl Debug for Receiver {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Receiver");
        formatter.field("attrs", &self.attrs);
        formatter.field("reference", &self.reference);
        formatter.field("mutability", &self.mutability);
        formatter.field("self_token", &self.self_token);
        formatter.field("colon_token", &self.colon_token);
        formatter.field("ty", &self.ty);
        formatter.finish()
    }
}
impl Debug for ReturnType {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("ReturnType::")?;
        match self {
            ReturnType::Default => formatter.write_str("Default"),
            ReturnType::Type(v0, v1) => {
                let mut formatter = formatter.debug_tuple("Type");
                formatter.field(v0);
                formatter.field(v1);
                formatter.finish()
            },
        }
    }
}
impl Debug for Signature {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Signature");
        formatter.field("constness", &self.constness);
        formatter.field("asyncness", &self.asyncness);
        formatter.field("unsafety", &self.unsafety);
        formatter.field("abi", &self.abi);
        formatter.field("fn_token", &self.fn_token);
        formatter.field("ident", &self.ident);
        formatter.field("generics", &self.generics);
        formatter.field("paren_token", &self.paren_token);
        formatter.field("inputs", &self.inputs);
        formatter.field("variadic", &self.variadic);
        formatter.field("output", &self.output);
        formatter.finish()
    }
}
impl Debug for StaticMutability {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("StaticMutability::")?;
        match self {
            StaticMutability::Mut(v0) => {
                let mut formatter = formatter.debug_tuple("Mut");
                formatter.field(v0);
                formatter.finish()
            },
            StaticMutability::None => formatter.write_str("None"),
        }
    }
}
impl Debug for Stmt {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Stmt::")?;
        match self {
            Stmt::Local(v0) => v0.debug(formatter, "Local"),
            Stmt::Item(v0) => {
                let mut formatter = formatter.debug_tuple("Item");
                formatter.field(v0);
                formatter.finish()
            },
            Stmt::Expr(v0, v1) => {
                let mut formatter = formatter.debug_tuple("Expr");
                formatter.field(v0);
                formatter.field(v1);
                formatter.finish()
            },
            Stmt::Macro(v0) => v0.debug(formatter, "Macro"),
        }
    }
}
impl Debug for StmtMacro {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl StmtMacro {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("mac", &self.mac);
                formatter.field("semi_token", &self.semi);
                formatter.finish()
            }
        }
        self.debug(formatter, "StmtMacro")
    }
}
impl Debug for TraitBound {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TraitBound");
        formatter.field("paren_token", &self.paren);
        formatter.field("modifier", &self.modifier);
        formatter.field("lifetimes", &self.lifetimes);
        formatter.field("path", &self.path);
        formatter.finish()
    }
}
impl Debug for TraitBoundModifier {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("TraitBoundModifier::")?;
        match self {
            TraitBoundModifier::None => formatter.write_str("None"),
            TraitBoundModifier::Maybe(v0) => {
                let mut formatter = formatter.debug_tuple("Maybe");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for TraitItem {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("TraitItem::")?;
        match self {
            TraitItem::Const(v0) => v0.debug(formatter, "Const"),
            TraitItem::Fn(v0) => v0.debug(formatter, "Fn"),
            TraitItem::Type(v0) => v0.debug(formatter, "Type"),
            TraitItem::Macro(v0) => v0.debug(formatter, "Macro"),
            TraitItem::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for TraitItemConst {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TraitItemConst {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("const_token", &self.const_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("colon_token", &self.colon_token);
                formatter.field("ty", &self.ty);
                formatter.field("default", &self.default);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "TraitItemConst")
    }
}
impl Debug for TraitItemFn {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TraitItemFn {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("sig", &self.sig);
                formatter.field("default", &self.default);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "TraitItemFn")
    }
}
impl Debug for TraitItemMacro {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TraitItemMacro {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("mac", &self.mac);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "TraitItemMacro")
    }
}
impl Debug for TraitItemType {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TraitItemType {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("attrs", &self.attrs);
                formatter.field("type_token", &self.type_token);
                formatter.field("ident", &self.ident);
                formatter.field("generics", &self.generics);
                formatter.field("colon_token", &self.colon_token);
                formatter.field("bounds", &self.bounds);
                formatter.field("default", &self.default);
                formatter.field("semi_token", &self.semi_token);
                formatter.finish()
            }
        }
        self.debug(formatter, "TraitItemType")
    }
}
impl Debug for Type {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Type::")?;
        match self {
            Type::Array(v0) => v0.debug(formatter, "Array"),
            Type::BareFn(v0) => v0.debug(formatter, "BareFn"),
            Type::Group(v0) => v0.debug(formatter, "Group"),
            Type::ImplTrait(v0) => v0.debug(formatter, "ImplTrait"),
            Type::Infer(v0) => v0.debug(formatter, "Infer"),
            Type::Macro(v0) => v0.debug(formatter, "Macro"),
            Type::Never(v0) => v0.debug(formatter, "Never"),
            Type::Paren(v0) => v0.debug(formatter, "Paren"),
            Type::Path(v0) => v0.debug(formatter, "Path"),
            Type::Ptr(v0) => v0.debug(formatter, "Ptr"),
            Type::Reference(v0) => v0.debug(formatter, "Reference"),
            Type::Slice(v0) => v0.debug(formatter, "Slice"),
            Type::TraitObject(v0) => v0.debug(formatter, "TraitObject"),
            Type::Tuple(v0) => v0.debug(formatter, "Tuple"),
            Type::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for TypeArray {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeArray {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("bracket_token", &self.bracket_token);
                formatter.field("elem", &self.elem);
                formatter.field("semi_token", &self.semi);
                formatter.field("len", &self.len);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeArray")
    }
}
impl Debug for TypeBareFn {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeBareFn {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("lifetimes", &self.lifetimes);
                formatter.field("unsafety", &self.unsafe_);
                formatter.field("abi", &self.abi);
                formatter.field("fn_token", &self.fn_);
                formatter.field("paren_token", &self.paren);
                formatter.field("inputs", &self.inputs);
                formatter.field("variadic", &self.variadic);
                formatter.field("output", &self.output);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeBareFn")
    }
}
impl Debug for TypeGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeGroup {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("group_token", &self.group);
                formatter.field("elem", &self.elem);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeGroup")
    }
}
impl Debug for TypeImplTrait {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeImplTrait {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("impl_token", &self.impl_);
                formatter.field("bounds", &self.bounds);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeImplTrait")
    }
}
impl Debug for TypeInfer {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeInfer {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("underscore_token", &self.underscore);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeInfer")
    }
}
impl Debug for TypeMacro {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeMacro {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("mac", &self.mac);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeMacro")
    }
}
impl Debug for TypeNever {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeNever {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("bang_token", &self.bang);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeNever")
    }
}
impl Debug for TypeParam {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeParam");
        formatter.field("attrs", &self.attrs);
        formatter.field("ident", &self.ident);
        formatter.field("colon_token", &self.colon);
        formatter.field("bounds", &self.bounds);
        formatter.field("eq_token", &self.equal);
        formatter.field("default", &self.default);
        formatter.finish()
    }
}
impl Debug for TypeParamBound {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("TypeParamBound::")?;
        match self {
            TypeParamBound::Trait(v0) => {
                let mut formatter = formatter.debug_tuple("Trait");
                formatter.field(v0);
                formatter.finish()
            },
            TypeParamBound::Lifetime(v0) => v0.debug(formatter, "Lifetime"),
            TypeParamBound::Verbatim(v0) => {
                let mut formatter = formatter.debug_tuple("Verbatim");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for TypeParen {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeParen {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("paren_token", &self.paren_token);
                formatter.field("elem", &self.elem);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeParen")
    }
}
impl Debug for TypePath {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypePath {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("qself", &self.qself);
                formatter.field("path", &self.path);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypePath")
    }
}
impl Debug for TypePtr {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypePtr {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("star_token", &self.star);
                formatter.field("const_token", &self.const_);
                formatter.field("mutability", &self.mut_);
                formatter.field("elem", &self.elem);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypePtr")
    }
}
impl Debug for TypeReference {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeReference {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("and_token", &self.and_);
                formatter.field("lifetime", &self.lifetime);
                formatter.field("mutability", &self.mut_);
                formatter.field("elem", &self.elem);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeReference")
    }
}
impl Debug for TypeSlice {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeSlice {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("bracket_token", &self.bracket);
                formatter.field("elem", &self.elem);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeSlice")
    }
}
impl Debug for TypeTraitObject {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeTraitObject {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("dyn_token", &self.dyn_);
                formatter.field("bounds", &self.bounds);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeTraitObject")
    }
}
impl Debug for TypeTuple {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        impl TypeTuple {
            fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut formatter = formatter.debug_struct(name);
                formatter.field("paren_token", &self.paren);
                formatter.field("elems", &self.elems);
                formatter.finish()
            }
        }
        self.debug(formatter, "TypeTuple")
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
impl Debug for UseGlob {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("UseGlob");
        formatter.field("star_token", &self.star_token);
        formatter.finish()
    }
}
impl Debug for UseGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("UseGroup");
        formatter.field("brace_token", &self.brace_token);
        formatter.field("items", &self.items);
        formatter.finish()
    }
}
impl Debug for UseName {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("UseName");
        formatter.field("ident", &self.ident);
        formatter.finish()
    }
}
impl Debug for UsePath {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("UsePath");
        formatter.field("ident", &self.ident);
        formatter.field("colon2_token", &self.colon2_token);
        formatter.field("tree", &self.tree);
        formatter.finish()
    }
}
impl Debug for UseRename {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("UseRename");
        formatter.field("ident", &self.ident);
        formatter.field("as_token", &self.as_token);
        formatter.field("rename", &self.rename);
        formatter.finish()
    }
}
impl Debug for UseTree {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("UseTree::")?;
        match self {
            UseTree::Path(v0) => {
                let mut formatter = formatter.debug_tuple("Path");
                formatter.field(v0);
                formatter.finish()
            },
            UseTree::Name(v0) => {
                let mut formatter = formatter.debug_tuple("Name");
                formatter.field(v0);
                formatter.finish()
            },
            UseTree::Rename(v0) => {
                let mut formatter = formatter.debug_tuple("Rename");
                formatter.field(v0);
                formatter.finish()
            },
            UseTree::Glob(v0) => {
                let mut formatter = formatter.debug_tuple("Glob");
                formatter.field(v0);
                formatter.finish()
            },
            UseTree::Group(v0) => {
                let mut formatter = formatter.debug_tuple("Group");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
impl Debug for Variadic {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Variadic");
        formatter.field("attrs", &self.attrs);
        formatter.field("pat", &self.pat);
        formatter.field("dots", &self.dots);
        formatter.field("comma", &self.comma);
        formatter.finish()
    }
}
impl Debug for Variant {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Variant");
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
                formatter.field("pub_token", &self.pub_);
                formatter.field("paren_token", &self.paren);
                formatter.field("in_token", &self.in_);
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
impl Debug for WhereClause {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("WhereClause");
        formatter.field("where_token", &self.where_);
        formatter.field("predicates", &self.preds);
        formatter.finish()
    }
}
impl Debug for WherePred {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("WherePredicate::")?;
        match self {
            WherePred::Lifetime(v0) => {
                let mut formatter = formatter.debug_tuple("Lifetime");
                formatter.field(v0);
                formatter.finish()
            },
            WherePred::Type(v0) => {
                let mut formatter = formatter.debug_tuple("Type");
                formatter.field(v0);
                formatter.finish()
            },
        }
    }
}
