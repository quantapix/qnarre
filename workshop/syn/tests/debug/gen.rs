#![allow(clippy::match_wildcard_for_single_variants)]
use super::{Lite, Present};
use ref_cast::RefCast;
use std::fmt::{self, Debug, Display};
impl Debug for Lite<syn::Abi> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Abi");
        if let Some(val) = &self.value.name {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::lit::Str);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("name", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::AngledArgs> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("path::path::AngledArgs");
        if self.value.colon2.is_some() {
            f.field("colon2", &Present);
        }
        if !self.value.args.is_empty() {
            f.field("args", Lite(&self.value.args));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::Arm> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Arm");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("pat", Lite(&self.value.pat));
        if let Some(val) = &self.value.guard {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::If, Box<syn::Expr>));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("guard", Print::ref_cast(val));
        }
        f.field("body", Lite(&self.value.body));
        if self.value.comma.is_some() {
            f.field("comma", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::AssocConst> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("AssocConst");
        f.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.gnrs {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngledArgs);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("gens", Print::ref_cast(val));
        }
        f.field("value", Lite(&self.value.val));
        f.finish()
    }
}
impl Debug for Lite<syn::AssocType> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("AssocType");
        f.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.gnrs {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngledArgs);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("gens", Print::ref_cast(val));
        }
        f.field("ty", Lite(&self.value.ty));
        f.finish()
    }
}
impl Debug for Lite<syn::attr::Style> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::attr::Style::Outer => f.write_str("attr::Style::Outer"),
            syn::attr::Style::Inner(_val) => {
                f.write_str("attr::Style::Inner")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::attr::Attr> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("attr::Attr");
        f.field("style", Lite(&self.value.style));
        f.field("meta", Lite(&self.value.meta));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::FnArg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::FnArg");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.name {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((proc_macro2::Ident, syn::tok::Colon));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("name", Print::ref_cast(val));
        }
        f.field("ty", Lite(&self.value.typ));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Variadic> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Variadic");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.name {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((proc_macro2::Ident, syn::tok::Colon));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("name", Print::ref_cast(val));
        }
        if self.value.comma.is_some() {
            f.field("comma", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::BinOp> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::BinOp::Add(_val) => {
                f.write_str("BinOp::Add")?;
                Ok(())
            },
            syn::BinOp::Sub(_val) => {
                f.write_str("BinOp::Sub")?;
                Ok(())
            },
            syn::BinOp::Mul(_val) => {
                f.write_str("BinOp::Mul")?;
                Ok(())
            },
            syn::BinOp::Div(_val) => {
                f.write_str("BinOp::Div")?;
                Ok(())
            },
            syn::BinOp::Rem(_val) => {
                f.write_str("BinOp::Rem")?;
                Ok(())
            },
            syn::BinOp::And(_val) => {
                f.write_str("BinOp::And")?;
                Ok(())
            },
            syn::BinOp::Or(_val) => {
                f.write_str("BinOp::Or")?;
                Ok(())
            },
            syn::BinOp::BitXor(_val) => {
                f.write_str("BinOp::BitXor")?;
                Ok(())
            },
            syn::BinOp::BitAnd(_val) => {
                f.write_str("BinOp::BitAnd")?;
                Ok(())
            },
            syn::BinOp::BitOr(_val) => {
                f.write_str("BinOp::BitOr")?;
                Ok(())
            },
            syn::BinOp::Shl(_val) => {
                f.write_str("BinOp::Shl")?;
                Ok(())
            },
            syn::BinOp::Shr(_val) => {
                f.write_str("BinOp::Shr")?;
                Ok(())
            },
            syn::BinOp::Eq(_val) => {
                f.write_str("BinOp::Eq")?;
                Ok(())
            },
            syn::BinOp::Lt(_val) => {
                f.write_str("BinOp::Lt")?;
                Ok(())
            },
            syn::BinOp::Le(_val) => {
                f.write_str("BinOp::Le")?;
                Ok(())
            },
            syn::BinOp::Ne(_val) => {
                f.write_str("BinOp::Ne")?;
                Ok(())
            },
            syn::BinOp::Ge(_val) => {
                f.write_str("BinOp::Ge")?;
                Ok(())
            },
            syn::BinOp::Gt(_val) => {
                f.write_str("BinOp::Gt")?;
                Ok(())
            },
            syn::BinOp::AddAssign(_val) => {
                f.write_str("BinOp::AddAssign")?;
                Ok(())
            },
            syn::BinOp::SubAssign(_val) => {
                f.write_str("BinOp::SubAssign")?;
                Ok(())
            },
            syn::BinOp::MulAssign(_val) => {
                f.write_str("BinOp::MulAssign")?;
                Ok(())
            },
            syn::BinOp::DivAssign(_val) => {
                f.write_str("BinOp::DivAssign")?;
                Ok(())
            },
            syn::BinOp::RemAssign(_val) => {
                f.write_str("BinOp::RemAssign")?;
                Ok(())
            },
            syn::BinOp::BitXorAssign(_val) => {
                f.write_str("BinOp::BitXorAssign")?;
                Ok(())
            },
            syn::BinOp::BitAndAssign(_val) => {
                f.write_str("BinOp::BitAndAssign")?;
                Ok(())
            },
            syn::BinOp::BitOrAssign(_val) => {
                f.write_str("BinOp::BitOrAssign")?;
                Ok(())
            },
            syn::BinOp::ShlAssign(_val) => {
                f.write_str("BinOp::ShlAssign")?;
                Ok(())
            },
            syn::BinOp::ShrAssign(_val) => {
                f.write_str("BinOp::ShrAssign")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::Block> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Block");
        if !self.value.stmts.is_empty() {
            f.field("stmts", Lite(&self.value.stmts));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::Bgen::bound::Lifes> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Bgen::bound::Lifes");
        if !self.value.lifes.is_empty() {
            f.field("lifetimes", Lite(&self.value.lifes));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::gen::param::Const> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::param::Const");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("ident", Lite(&self.value.ident));
        f.field("ty", Lite(&self.value.typ));
        if self.value.eq.is_some() {
            f.field("eq", &Present);
        }
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Expr);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("default", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::Constraint> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Constraint");
        f.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.gnrs {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngledArgs);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("gens", Print::ref_cast(val));
        }
        if !self.value.bounds.is_empty() {
            f.field("bounds", Lite(&self.value.bounds));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::Data> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Data::Struct(_val) => {
                let mut f = f.debug_struct("Data::Struct");
                f.field("fields", Lite(&_val.fields));
                if _val.semi.is_some() {
                    f.field("semi", &Present);
                }
                f.finish()
            },
            syn::Data::Enum(_val) => {
                let mut f = f.debug_struct("Data::Enum");
                if !_val.variants.is_empty() {
                    f.field("variants", Lite(&_val.variants));
                }
                f.finish()
            },
            syn::Data::Union(_val) => {
                let mut f = f.debug_struct("Data::Union");
                f.field("fields", Lite(&_val.fields));
                f.finish()
            },
        }
    }
}
impl Debug for Lite<syn::DataEnum> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("DataEnum");
        if !self.value.variants.is_empty() {
            f.field("variants", Lite(&self.value.variants));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::DataStruct> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("DataStruct");
        f.field("fields", Lite(&self.value.fields));
        if self.value.semi.is_some() {
            f.field("semi", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::DataUnion> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("DataUnion");
        f.field("fields", Lite(&self.value.fields));
        f.finish()
    }
}
impl Debug for Lite<syn::DeriveInput> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("DeriveInput");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        f.field("data", Lite(&self.value.data));
        f.finish()
    }
}
impl Debug for Lite<syn::Expr> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Expr::Array(_val) => {
                let mut f = f.debug_struct("Expr::Array");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    f.field("elems", Lite(&_val.elems));
                }
                f.finish()
            },
            syn::Expr::Assign(_val) => {
                let mut f = f.debug_struct("Expr::Assign");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("left", Lite(&_val.left));
                f.field("right", Lite(&_val.right));
                f.finish()
            },
            syn::Expr::Async(_val) => {
                let mut f = f.debug_struct("Expr::Async");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if _val.move_.is_some() {
                    f.field("capture", &Present);
                }
                f.field("block", Lite(&_val.block));
                f.finish()
            },
            syn::Expr::Await(_val) => {
                let mut f = f.debug_struct("Expr::Await");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("base", Lite(&_val.expr));
                f.finish()
            },
            syn::Expr::Binary(_val) => {
                let mut f = f.debug_struct("Expr::Binary");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("left", Lite(&_val.left));
                f.field("op", Lite(&_val.op));
                f.field("right", Lite(&_val.right));
                f.finish()
            },
            syn::Expr::Block(_val) => {
                let mut f = f.debug_struct("Expr::Block");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("label", Print::ref_cast(val));
                }
                f.field("block", Lite(&_val.block));
                f.finish()
            },
            syn::Expr::Break(_val) => {
                let mut f = f.debug_struct("Expr::Break");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Life);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("label", Print::ref_cast(val));
                }
                if let Some(val) = &_val.expr {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("expr", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::Expr::Call(_val) => {
                let mut f = f.debug_struct("Expr::Call");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("func", Lite(&_val.func));
                if !_val.args.is_empty() {
                    f.field("args", Lite(&_val.args));
                }
                f.finish()
            },
            syn::Expr::Cast(_val) => {
                let mut f = f.debug_struct("Expr::Cast");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("expr", Lite(&_val.expr));
                f.field("ty", Lite(&_val.typ));
                f.finish()
            },
            syn::Expr::Closure(_val) => {
                let mut f = f.debug_struct("Expr::Closure");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.lifes {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Bgen::bound::Lifes);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("lifetimes", Print::ref_cast(val));
                }
                if _val.const_.is_some() {
                    f.field("const_", &Present);
                }
                if _val.static_.is_some() {
                    f.field("movability", &Present);
                }
                if _val.async_.is_some() {
                    f.field("asyncness", &Present);
                }
                if _val.move_.is_some() {
                    f.field("capture", &Present);
                }
                if !_val.inputs.is_empty() {
                    f.field("inputs", Lite(&_val.inputs));
                }
                f.field("output", Lite(&_val.ret));
                f.field("body", Lite(&_val.body));
                f.finish()
            },
            syn::Expr::Const(_val) => {
                let mut f = f.debug_struct("Expr::Const");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("block", Lite(&_val.block));
                f.finish()
            },
            syn::Expr::Continue(_val) => {
                let mut f = f.debug_struct("Expr::Continue");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Life);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("label", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::Expr::Field(_val) => {
                let mut f = f.debug_struct("Expr::Field");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("base", Lite(&_val.base));
                f.field("member", Lite(&_val.member));
                f.finish()
            },
            syn::Expr::ForLoop(_val) => {
                let mut f = f.debug_struct("Expr::ForLoop");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("label", Print::ref_cast(val));
                }
                f.field("pat", Lite(&_val.pat));
                f.field("expr", Lite(&_val.expr));
                f.field("body", Lite(&_val.body));
                f.finish()
            },
            syn::Expr::Group(_val) => {
                let mut f = f.debug_struct("Expr::Group");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("expr", Lite(&_val.expr));
                f.finish()
            },
            syn::Expr::If(_val) => {
                let mut f = f.debug_struct("Expr::If");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("cond", Lite(&_val.cond));
                f.field("then_branch", Lite(&_val.then_branch));
                if let Some(val) = &_val.else_branch {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Else, Box<syn::Expr>));
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("else_branch", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::Expr::Index(_val) => {
                let mut f = f.debug_struct("Expr::Index");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("expr", Lite(&_val.expr));
                f.field("index", Lite(&_val.index));
                f.finish()
            },
            syn::Expr::Infer(_val) => {
                let mut f = f.debug_struct("Expr::Infer");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.finish()
            },
            syn::Expr::Let(_val) => {
                let mut f = f.debug_struct("Expr::Let");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("pat", Lite(&_val.pat));
                f.field("expr", Lite(&_val.expr));
                f.finish()
            },
            syn::Expr::Lit(_val) => {
                let mut f = f.debug_struct("Expr::Lit");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("lit", Lite(&_val.lit));
                f.finish()
            },
            syn::Expr::Loop(_val) => {
                let mut f = f.debug_struct("Expr::Loop");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("label", Print::ref_cast(val));
                }
                f.field("body", Lite(&_val.body));
                f.finish()
            },
            syn::Expr::Macro(_val) => {
                let mut f = f.debug_struct("Expr::Macro");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("mac", Lite(&_val.mac));
                f.finish()
            },
            syn::Expr::Match(_val) => {
                let mut f = f.debug_struct("Expr::Match");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("expr", Lite(&_val.expr));
                if !_val.arms.is_empty() {
                    f.field("arms", Lite(&_val.arms));
                }
                f.finish()
            },
            syn::Expr::MethodCall(_val) => {
                let mut f = f.debug_struct("Expr::MethodCall");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("receiver", Lite(&_val.receiver));
                f.field("method", Lite(&_val.method));
                if let Some(val) = &_val.turbofish {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::AngledArgs);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("turbofish", Print::ref_cast(val));
                }
                if !_val.args.is_empty() {
                    f.field("args", Lite(&_val.args));
                }
                f.finish()
            },
            syn::Expr::Paren(_val) => {
                let mut f = f.debug_struct("Expr::Paren");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("expr", Lite(&_val.expr));
                f.finish()
            },
            syn::Expr::Path(_val) => {
                let mut f = f.debug_struct("Expr::Path");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("qself", Print::ref_cast(val));
                }
                f.field("path", Lite(&_val.path));
                f.finish()
            },
            syn::Expr::Range(_val) => {
                let mut f = f.debug_struct("Expr::Range");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.start {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("start", Print::ref_cast(val));
                }
                f.field("limits", Lite(&_val.limits));
                if let Some(val) = &_val.end {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("end", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::Expr::Reference(_val) => {
                let mut f = f.debug_struct("Expr::Reference");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if _val.mut_.is_some() {
                    f.field("mut_", &Present);
                }
                f.field("expr", Lite(&_val.expr));
                f.finish()
            },
            syn::Expr::Repeat(_val) => {
                let mut f = f.debug_struct("Expr::Repeat");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("expr", Lite(&_val.expr));
                f.field("len", Lite(&_val.len));
                f.finish()
            },
            syn::Expr::Return(_val) => {
                let mut f = f.debug_struct("Expr::Return");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.expr {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("expr", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::Expr::Struct(_val) => {
                let mut f = f.debug_struct("Expr::Struct");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("qself", Print::ref_cast(val));
                }
                f.field("path", Lite(&_val.path));
                if !_val.fields.is_empty() {
                    f.field("fields", Lite(&_val.fields));
                }
                if _val.dot2.is_some() {
                    f.field("dot2", &Present);
                }
                if let Some(val) = &_val.rest {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("rest", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::Expr::Try(_val) => {
                let mut f = f.debug_struct("Expr::Try");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("expr", Lite(&_val.expr));
                f.finish()
            },
            syn::Expr::TryBlock(_val) => {
                let mut f = f.debug_struct("Expr::TryBlock");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("block", Lite(&_val.block));
                f.finish()
            },
            syn::Expr::Tuple(_val) => {
                let mut f = f.debug_struct("Expr::Tuple");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    f.field("elems", Lite(&_val.elems));
                }
                f.finish()
            },
            syn::Expr::Unary(_val) => {
                let mut f = f.debug_struct("Expr::Unary");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("op", Lite(&_val.op));
                f.field("expr", Lite(&_val.expr));
                f.finish()
            },
            syn::Expr::Unsafe(_val) => {
                let mut f = f.debug_struct("Expr::Unsafe");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("block", Lite(&_val.block));
                f.finish()
            },
            syn::Expr::Stream(_val) => {
                f.write_str("Expr::Stream")?;
                f.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                f.write_str("`)")?;
                Ok(())
            },
            syn::Expr::While(_val) => {
                let mut f = f.debug_struct("Expr::While");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("label", Print::ref_cast(val));
                }
                f.field("cond", Lite(&_val.cond));
                f.field("body", Lite(&_val.body));
                f.finish()
            },
            syn::Expr::Yield(_val) => {
                let mut f = f.debug_struct("Expr::Yield");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.expr {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("expr", Print::ref_cast(val));
                }
                f.finish()
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::expr::Array> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Array");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            f.field("elems", Lite(&self.value.elems));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Assign> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Assign");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("left", Lite(&self.value.left));
        f.field("right", Lite(&self.value.right));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Async> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Async");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.move_.is_some() {
            f.field("capture", &Present);
        }
        f.field("block", Lite(&self.value.block));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Await> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Await");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("base", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Binary> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Binary");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("left", Lite(&self.value.left));
        f.field("op", Lite(&self.value.op));
        f.field("right", Lite(&self.value.right));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Block> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Block");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("label", Print::ref_cast(val));
        }
        f.field("block", Lite(&self.value.block));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Break> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Break");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Life);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("label", Print::ref_cast(val));
        }
        if let Some(val) = &self.value.expr {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("expr", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Call> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Call");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("func", Lite(&self.value.func));
        if !self.value.args.is_empty() {
            f.field("args", Lite(&self.value.args));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Cast> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Cast");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("expr", Lite(&self.value.expr));
        f.field("ty", Lite(&self.value.typ));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Closure> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Closure");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.lifes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Bgen::bound::Lifes);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("lifetimes", Print::ref_cast(val));
        }
        if self.value.const_.is_some() {
            f.field("const_", &Present);
        }
        if self.value.static_.is_some() {
            f.field("movability", &Present);
        }
        if self.value.async_.is_some() {
            f.field("asyncness", &Present);
        }
        if self.value.move_.is_some() {
            f.field("capture", &Present);
        }
        if !self.value.inputs.is_empty() {
            f.field("inputs", Lite(&self.value.inputs));
        }
        f.field("output", Lite(&self.value.ret));
        f.field("body", Lite(&self.value.body));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Const> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Const");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("block", Lite(&self.value.block));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Continue> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Continue");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Life);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("label", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Field> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Field");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("base", Lite(&self.value.base));
        f.field("member", Lite(&self.value.memb));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::ForLoop> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::ForLoop");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("label", Print::ref_cast(val));
        }
        f.field("pat", Lite(&self.value.pat));
        f.field("expr", Lite(&self.value.expr));
        f.field("body", Lite(&self.value.body));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Group> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Group");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("expr", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::If> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::If");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("cond", Lite(&self.value.cond));
        f.field("then_branch", Lite(&self.value.then_branch));
        if let Some(val) = &self.value.else_branch {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Else, Box<syn::Expr>));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("else_branch", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Index> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Index");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("expr", Lite(&self.value.expr));
        f.field("index", Lite(&self.value.index));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Infer> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Infer");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Let> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Let");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("pat", Lite(&self.value.pat));
        f.field("expr", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Lit> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Lit");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("lit", Lite(&self.value.lit));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Loop> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Loop");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("label", Print::ref_cast(val));
        }
        f.field("body", Lite(&self.value.body));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Mac> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Mac");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("mac", Lite(&self.value.mac));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Match> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Match");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("expr", Lite(&self.value.expr));
        if !self.value.arms.is_empty() {
            f.field("arms", Lite(&self.value.arms));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::MethodCall> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::MethodCall");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("receiver", Lite(&self.value.expr));
        f.field("method", Lite(&self.value.method));
        if let Some(val) = &self.value.turbofish {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngledArgs);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("turbofish", Print::ref_cast(val));
        }
        if !self.value.args.is_empty() {
            f.field("args", Lite(&self.value.args));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Paren> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Paren");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("expr", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Path> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Path");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("qself", Print::ref_cast(val));
        }
        f.field("path", Lite(&self.value.path));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Range> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Range");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.beg {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("start", Print::ref_cast(val));
        }
        f.field("limits", Lite(&self.value.limits));
        if let Some(val) = &self.value.end {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("end", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Ref> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Ref");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.mut_.is_some() {
            f.field("mut_", &Present);
        }
        f.field("expr", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Repeat> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Repeat");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("expr", Lite(&self.value.expr));
        f.field("len", Lite(&self.value.len));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Return> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Return");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.expr {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("expr", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::expr::ExprStruct> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::expr::ExprStruct");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("qself", Print::ref_cast(val));
        }
        f.field("path", Lite(&self.value.path));
        if !self.value.fields.is_empty() {
            f.field("fields", Lite(&self.value.fields));
        }
        if self.value.dot2.is_some() {
            f.field("dot2", &Present);
        }
        if let Some(val) = &self.value.rest {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("rest", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Try> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Try");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("expr", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::TryBlock> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::TryBlock");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("block", Lite(&self.value.block));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Tuple> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Tuple");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            f.field("elems", Lite(&self.value.elems));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Unary> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Unary");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("op", Lite(&self.value.op));
        f.field("expr", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Unsafe> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Unsafe");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("block", Lite(&self.value.block));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::While> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::While");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("label", Print::ref_cast(val));
        }
        f.field("cond", Lite(&self.value.cond));
        f.field("body", Lite(&self.value.body));
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Yield> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("expr::Yield");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.expr {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("expr", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::Field> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Field");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        match self.value.mut_ {
            syn::FieldMut::None => {},
            _ => {
                f.field("mut_", Lite(&self.value.mut_));
            },
        }
        if let Some(val) = &self.value.ident {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(proc_macro2::Ident);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("ident", Print::ref_cast(val));
        }
        if self.value.colon.is_some() {
            f.field("colon", &Present);
        }
        f.field("ty", Lite(&self.value.typ));
        f.finish()
    }
}
impl Debug for Lite<syn::FieldMut> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::FieldMut::None => f.write_str("Mut::None"),
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::patt::Field> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::Field");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("member", Lite(&self.value.member));
        if self.value.colon.is_some() {
            f.field("colon", &Present);
        }
        f.field("pat", Lite(&self.value.patt));
        f.finish()
    }
}
impl Debug for Lite<syn::FieldValue> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("FieldValue");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("member", Lite(&self.value.member));
        if self.value.colon.is_some() {
            f.field("colon", &Present);
        }
        f.field("expr", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::Fields> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Fields::Named(_val) => {
                let mut f = f.debug_struct("Fields::Named");
                if !_val.field.is_empty() {
                    f.field("fields", Lite(&_val.fields));
                }
                f.finish()
            },
            syn::Fields::Unnamed(_val) => {
                let mut f = f.debug_struct("Fields::Unnamed");
                if !_val.field.is_empty() {
                    f.field("fields", Lite(&_val.fields));
                }
                f.finish()
            },
            syn::Fields::Unit => f.write_str("Fields::Unit"),
        }
    }
}
impl Debug for Lite<syn::FieldsNamed> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("FieldsNamed");
        if !self.value.field.is_empty() {
            f.field("fields", Lite(&self.value.fields));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::FieldsUnnamed> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("FieldsUnnamed");
        if !self.value.field.is_empty() {
            f.field("fields", Lite(&self.value.fields));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::File> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::File");
        if let Some(val) = &self.value.shebang {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(String);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("shebang", Print::ref_cast(val));
        }
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.items.is_empty() {
            f.field("items", Lite(&self.value.items));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::FnArg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::item::FnArg::Receiver(_val) => {
                f.write_str("item::FnArg::Receiver")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::item::FnArg::Type(_val) => {
                f.write_str("item::FnArg::Typed")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::item::Foreign::Item> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::item::Foreign::Item::Fn(_val) => {
                let mut f = f.debug_struct("item::Foreign::Item::Fn");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                f.field("sig", Lite(&_val.sig));
                f.finish()
            },
            syn::item::Foreign::Item::Static(_val) => {
                let mut f = f.debug_struct("item::Foreign::Item::Static");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                match _val.mut_ {
                    syn::StaticMut::None => {},
                    _ => {
                        f.field("mut_", Lite(&_val.mut_));
                    },
                }
                f.field("ident", Lite(&_val.ident));
                f.field("ty", Lite(&_val.typ));
                f.finish()
            },
            syn::item::Foreign::Item::Type(_val) => {
                let mut f = f.debug_struct("item::Foreign::Item::Type");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                f.finish()
            },
            syn::item::Foreign::Item::Macro(_val) => {
                let mut f = f.debug_struct("item::Foreign::Item::Macro");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("mac", Lite(&_val.mac));
                if _val.semi.is_some() {
                    f.field("semi", &Present);
                }
                f.finish()
            },
            syn::item::Foreign::Item::Stream(_val) => {
                f.write_str("item::Foreign::Item::Stream")?;
                f.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                f.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::item::Foreign::Fn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Foreign::Fn");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        f.field("sig", Lite(&self.value.sig));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Foreign::Mac> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Foreign::Mac");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("mac", Lite(&self.value.mac));
        if self.value.semi.is_some() {
            f.field("semi", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Foreign::Static> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Foreign::Static");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        match self.value.mut_ {
            syn::StaticMut::None => {},
            _ => {
                f.field("mut_", Lite(&self.value.mut_));
            },
        }
        f.field("ident", Lite(&self.value.ident));
        f.field("ty", Lite(&self.value.typ));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Foreign::Type> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Foreign::Type");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        f.finish()
    }
}
impl Debug for Lite<syn::Arg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Arg::Life(_val) => {
                f.write_str("path::Arg::Life")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::Arg::Type(_val) => {
                f.write_str("path::Arg::Type")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::Arg::Const(_val) => {
                f.write_str("path::Arg::Const")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::Arg::AssocType(_val) => {
                f.write_str("path::Arg::AssocType")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::Arg::AssocConst(_val) => {
                f.write_str("path::Arg::AssocConst")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::Arg::Constraint(_val) => {
                f.write_str("path::Arg::Constraint")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::gen::Param> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::gen::Param::Life(_val) => {
                f.write_str("gen::Param::Life")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::gen::Param::Type(_val) => {
                f.write_str("gen::Param::Type")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::gen::Param::Const(_val) => {
                f.write_str("gen::Param::Const")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::gen::Gens> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::Gens");
        if self.value.lt.is_some() {
            f.field("lt", &Present);
        }
        if !self.value.params.is_empty() {
            f.field("params", Lite(&self.value.params));
        }
        if self.value.gt.is_some() {
            f.field("gt", &Present);
        }
        if let Some(val) = &self.value.where_ {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::gen::Where);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("where_clause", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Impl::Item> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::item::Impl::Item::Const(_val) => {
                let mut f = f.debug_struct("item::Impl::Item::Const");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                if _val.default_.is_some() {
                    f.field("defaultness", &Present);
                }
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                f.field("ty", Lite(&_val.typ));
                f.field("expr", Lite(&_val.expr));
                f.finish()
            },
            syn::item::Impl::Item::Fn(_val) => {
                let mut f = f.debug_struct("item::Impl::Item::Fn");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                if _val.default_.is_some() {
                    f.field("defaultness", &Present);
                }
                f.field("sig", Lite(&_val.sig));
                f.field("block", Lite(&_val.block));
                f.finish()
            },
            syn::item::Impl::Item::Type(_val) => {
                let mut f = f.debug_struct("item::Impl::Item::Type");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                if _val.default_.is_some() {
                    f.field("defaultness", &Present);
                }
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                f.field("ty", Lite(&_val.typ));
                f.finish()
            },
            syn::item::Impl::Item::Macro(_val) => {
                let mut f = f.debug_struct("item::Impl::Item::Macro");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("mac", Lite(&_val.mac));
                if _val.semi.is_some() {
                    f.field("semi", &Present);
                }
                f.finish()
            },
            syn::item::Impl::Item::Stream(_val) => {
                f.write_str("item::Impl::Item::Stream")?;
                f.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                f.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::item::Impl::Const> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Impl::Const");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        if self.value.default_.is_some() {
            f.field("defaultness", &Present);
        }
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        f.field("ty", Lite(&self.value.typ));
        f.field("expr", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Impl::Fn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Impl::Fn");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        if self.value.default_.is_some() {
            f.field("defaultness", &Present);
        }
        f.field("sig", Lite(&self.value.sig));
        f.field("block", Lite(&self.value.block));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Impl::Mac> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Impl::Mac");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("mac", Lite(&self.value.mac));
        if self.value.semi.is_some() {
            f.field("semi", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Impl::Type> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Impl::Type");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        if self.value.default_.is_some() {
            f.field("defaultness", &Present);
        }
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        f.field("ty", Lite(&self.value.typ));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Impl::Restriction> {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unreachable!()
    }
}
impl Debug for Lite<syn::Index> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Index");
        f.field("index", Lite(&self.value.index));
        f.finish()
    }
}
impl Debug for Lite<syn::Item> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Item::Const(_val) => {
                let mut f = f.debug_struct("Item::Const");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                f.field("ty", Lite(&_val.typ));
                f.field("expr", Lite(&_val.expr));
                f.finish()
            },
            syn::Item::Enum(_val) => {
                let mut f = f.debug_struct("Item::Enum");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                if !_val.variants.is_empty() {
                    f.field("variants", Lite(&_val.variants));
                }
                f.finish()
            },
            syn::Item::Extern(_val) => {
                let mut f = f.debug_struct("Item::Extern");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                f.field("ident", Lite(&_val.ident));
                if let Some(val) = &_val.rename {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::As, proc_macro2::Ident));
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("rename", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::Item::Fn(_val) => {
                let mut f = f.debug_struct("Item::Fn");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                f.field("sig", Lite(&_val.sig));
                f.field("block", Lite(&_val.block));
                f.finish()
            },
            syn::Item::Foreign(_val) => {
                let mut f = f.debug_struct("Item::Foreign");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if _val.unsafe_.is_some() {
                    f.field("unsafe_", &Present);
                }
                f.field("abi", Lite(&_val.abi));
                if !_val.items.is_empty() {
                    f.field("items", Lite(&_val.items));
                }
                f.finish()
            },
            syn::Item::Impl(_val) => {
                let mut f = f.debug_struct("Item::Impl");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if _val.default_.is_some() {
                    f.field("defaultness", &Present);
                }
                if _val.unsafe_.is_some() {
                    f.field("unsafe_", &Present);
                }
                f.field("gens", Lite(&_val.gens));
                if let Some(val) = &_val.trait_ {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((Option<syn::tok::Not>, syn::Path, syn::tok::For));
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(
                                &(
                                    &super::Option {
                                        present: self.0 .0.is_some(),
                                    },
                                    Lite(&self.0 .1),
                                ),
                                formatter,
                            )?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("trait_", Print::ref_cast(val));
                }
                f.field("self_ty", Lite(&_val.typ));
                if !_val.items.is_empty() {
                    f.field("items", Lite(&_val.items));
                }
                f.finish()
            },
            syn::Item::Macro(_val) => {
                let mut f = f.debug_struct("Item::Macro");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.ident {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(proc_macro2::Ident);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("ident", Print::ref_cast(val));
                }
                f.field("mac", Lite(&_val.mac));
                if _val.semi.is_some() {
                    f.field("semi", &Present);
                }
                f.finish()
            },
            syn::Item::Mod(_val) => {
                let mut f = f.debug_struct("Item::Mod");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                if _val.unsafe_.is_some() {
                    f.field("unsafe_", &Present);
                }
                f.field("ident", Lite(&_val.ident));
                if let Some(val) = &_val.gist {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Brace, Vec<syn::Item>));
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("content", Print::ref_cast(val));
                }
                if _val.semi.is_some() {
                    f.field("semi", &Present);
                }
                f.finish()
            },
            syn::Item::Static(_val) => {
                let mut f = f.debug_struct("Item::Static");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                match _val.mut_ {
                    syn::StaticMut::None => {},
                    _ => {
                        f.field("mut_", Lite(&_val.mut_));
                    },
                }
                f.field("ident", Lite(&_val.ident));
                f.field("ty", Lite(&_val.typ));
                f.field("expr", Lite(&_val.expr));
                f.finish()
            },
            syn::Item::Struct(_val) => {
                let mut f = f.debug_struct("Item::Struct");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                f.field("fields", Lite(&_val.fields));
                if _val.semi.is_some() {
                    f.field("semi", &Present);
                }
                f.finish()
            },
            syn::Item::Trait(_val) => {
                let mut f = f.debug_struct("Item::Trait");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                if _val.unsafe_.is_some() {
                    f.field("unsafe_", &Present);
                }
                if _val.auto_.is_some() {
                    f.field("auto_", &Present);
                }
                if let Some(val) = &_val.restriction {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::item::Impl::Restriction);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("restriction", Print::ref_cast(val));
                }
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                if _val.colon.is_some() {
                    f.field("colon", &Present);
                }
                if !_val.supertraits.is_empty() {
                    f.field("supertraits", Lite(&_val.supertraits));
                }
                if !_val.items.is_empty() {
                    f.field("items", Lite(&_val.items));
                }
                f.finish()
            },
            syn::Item::TraitAlias(_val) => {
                let mut f = f.debug_struct("Item::TraitAlias");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                if !_val.bounds.is_empty() {
                    f.field("bounds", Lite(&_val.bounds));
                }
                f.finish()
            },
            syn::Item::Type(_val) => {
                let mut f = f.debug_struct("Item::Type");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                f.field("ty", Lite(&_val.typ));
                f.finish()
            },
            syn::Item::Union(_val) => {
                let mut f = f.debug_struct("Item::Union");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                f.field("fields", Lite(&_val.fields));
                f.finish()
            },
            syn::Item::Use(_val) => {
                let mut f = f.debug_struct("Item::Use");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("vis", Lite(&_val.vis));
                if _val.leading_colon.is_some() {
                    f.field("leading_colon", &Present);
                }
                f.field("tree", Lite(&_val.tree));
                f.finish()
            },
            syn::Item::Stream(_val) => {
                f.write_str("Item::Stream")?;
                f.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                f.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::item::Const> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Const");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        f.field("ty", Lite(&self.value.typ));
        f.field("expr", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Enum> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Enum");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        if !self.value.variants.is_empty() {
            f.field("variants", Lite(&self.value.variants));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Extern> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Extern");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        f.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.rename {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::As, proc_macro2::Ident));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("rename", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Fn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Fn");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        f.field("sig", Lite(&self.value.sig));
        f.field("block", Lite(&self.value.block));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Foreign> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Foreign");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.unsafe_.is_some() {
            f.field("unsafe_", &Present);
        }
        f.field("abi", Lite(&self.value.abi));
        if !self.value.items.is_empty() {
            f.field("items", Lite(&self.value.items));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Impl> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Impl");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.default_.is_some() {
            f.field("defaultness", &Present);
        }
        if self.value.unsafe_.is_some() {
            f.field("unsafe_", &Present);
        }
        f.field("gens", Lite(&self.value.gens));
        if let Some(val) = &self.value.trait_ {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((Option<syn::tok::Not>, syn::Path, syn::tok::For));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(
                        &(
                            &super::Option {
                                present: self.0 .0.is_some(),
                            },
                            Lite(&self.0 .1),
                        ),
                        formatter,
                    )?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("trait_", Print::ref_cast(val));
        }
        f.field("self_ty", Lite(&self.value.typ));
        if !self.value.items.is_empty() {
            f.field("items", Lite(&self.value.items));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Mac> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Mac");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.ident {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(proc_macro2::Ident);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("ident", Print::ref_cast(val));
        }
        f.field("mac", Lite(&self.value.mac));
        if self.value.semi.is_some() {
            f.field("semi", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Mod> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Mod");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        if self.value.unsafe_.is_some() {
            f.field("unsafe_", &Present);
        }
        f.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.items {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Brace, Vec<syn::Item>));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("content", Print::ref_cast(val));
        }
        if self.value.semi.is_some() {
            f.field("semi", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Static> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Static");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        match self.value.mut_ {
            syn::StaticMut::None => {},
            _ => {
                f.field("mut_", Lite(&self.value.mut_));
            },
        }
        f.field("ident", Lite(&self.value.ident));
        f.field("ty", Lite(&self.value.typ));
        f.field("expr", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Struct> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Struct");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        f.field("fields", Lite(&self.value.fields));
        if self.value.semi.is_some() {
            f.field("semi", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Trait> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Trait");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        if self.value.unsafe_.is_some() {
            f.field("unsafe_", &Present);
        }
        if self.value.auto_.is_some() {
            f.field("auto_", &Present);
        }
        if let Some(val) = &self.value.restriction {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::item::Impl::Restriction);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("restriction", Print::ref_cast(val));
        }
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        if self.value.colon.is_some() {
            f.field("colon", &Present);
        }
        if !self.value.supers.is_empty() {
            f.field("supertraits", Lite(&self.value.supers));
        }
        if !self.value.items.is_empty() {
            f.field("items", Lite(&self.value.items));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::TraitAlias> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::TraitAlias");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        if !self.value.bounds.is_empty() {
            f.field("bounds", Lite(&self.value.bounds));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Type> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Type");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        f.field("ty", Lite(&self.value.typ));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Union> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Union");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        f.field("fields", Lite(&self.value.fields));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Use> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Use");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("vis", Lite(&self.value.vis));
        if self.value.colon.is_some() {
            f.field("leading_colon", &Present);
        }
        f.field("tree", Lite(&self.value.tree));
        f.finish()
    }
}
impl Debug for Lite<syn::Label> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Label");
        f.field("name", Lite(&self.value.name));
        f.finish()
    }
}
impl Debug for Lite<syn::Life> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Life");
        f.field("ident", Lite(&self.value.ident));
        f.finish()
    }
}
impl Debug for Lite<syn::gen::param::Life> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::param::Life");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("life", Lite(&self.value.life));
        if self.value.colon.is_some() {
            f.field("colon", &Present);
        }
        if !self.value.bounds.is_empty() {
            f.field("bounds", Lite(&self.value.bounds));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::Lit> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Lit::Str(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::ByteStr(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::Byte(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::Char(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::Int(_val) => write!(formatter, "{}", _val),
            syn::Lit::Float(_val) => write!(formatter, "{}", _val),
            syn::Lit::Bool(_val) => {
                let mut f = f.debug_struct("Lit::Bool");
                f.field("value", Lite(&_val.val));
                f.finish()
            },
            syn::Lit::Stream(_val) => {
                f.write_str("Lit::Stream")?;
                f.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                f.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::lit::Bool> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("lit::Bool");
        f.field("value", Lite(&self.value.val));
        f.finish()
    }
}
impl Debug for Lite<syn::lit::Byte> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::lit::ByteStr> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::lit::Char> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::lit::Float> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}", &self.value)
    }
}
impl Debug for Lite<syn::lit::Int> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}", &self.value)
    }
}
impl Debug for Lite<syn::lit::Str> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::stmt::Local> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("stmt::Local");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("pat", Lite(&self.value.pat));
        if let Some(val) = &self.value.init {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::stmt::Init);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("init", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::stmt::Init> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("stmt::Init");
        f.field("expr", Lite(&self.value.expr));
        if let Some(val) = &self.value.diverge {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Else, Box<syn::Expr>));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("diverge", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::Macro> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Macro");
        f.field("path", Lite(&self.value.path));
        f.field("delimiter", Lite(&self.value.delim));
        f.field("tokens", Lite(&self.value.toks));
        f.finish()
    }
}
impl Debug for Lite<syn::tok::Delim> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::tok::Delim::Paren(_val) => {
                f.write_str("MacroDelimiter::Paren")?;
                Ok(())
            },
            syn::tok::Delim::Brace(_val) => {
                f.write_str("MacroDelimiter::Brace")?;
                Ok(())
            },
            syn::tok::Delim::Bracket(_val) => {
                f.write_str("MacroDelimiter::Bracket")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::Member> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Member::Named(_val) => {
                f.write_str("Member::Named")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::Member::Unnamed(_val) => {
                f.write_str("Member::Unnamed")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::meta::Meta> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::meta::Meta::Path(_val) => {
                let mut f = f.debug_struct("meta::Meta::Path");
                if _val.colon.is_some() {
                    f.field("leading_colon", &Present);
                }
                if !_val.segs.is_empty() {
                    f.field("segments", Lite(&_val.segs));
                }
                f.finish()
            },
            syn::meta::Meta::List(_val) => {
                let mut f = f.debug_struct("meta::Meta::List");
                f.field("path", Lite(&_val.path));
                f.field("delimiter", Lite(&_val.delim));
                f.field("tokens", Lite(&_val.toks));
                f.finish()
            },
            syn::meta::Meta::NameValue(_val) => {
                let mut f = f.debug_struct("meta::Meta::NameValue");
                f.field("path", Lite(&_val.path));
                f.field("value", Lite(&_val.expr));
                f.finish()
            },
        }
    }
}
impl Debug for Lite<syn::meta::List> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("meta::List");
        f.field("path", Lite(&self.value.path));
        f.field("delimiter", Lite(&self.value.delim));
        f.field("tokens", Lite(&self.value.toks));
        f.finish()
    }
}
impl Debug for Lite<syn::meta::NameValue> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("meta::NameValue");
        f.field("path", Lite(&self.value.path));
        f.field("value", Lite(&self.value.expr));
        f.finish()
    }
}
impl Debug for Lite<syn::ParenthesizedArgs> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("path::ParenthesizedArgs");
        if !self.value.ins.is_empty() {
            f.field("inputs", Lite(&self.value.ins));
        }
        f.field("output", Lite(&self.value.out));
        f.finish()
    }
}
impl Debug for Lite<syn::pat::Pat> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::pat::Pat::Const(_val) => {
                f.write_str("pat::Pat::Const")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::pat::Pat::Ident(_val) => {
                let mut f = f.debug_struct("pat::Pat::Ident");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if _val.ref_.is_some() {
                    f.field("by_ref", &Present);
                }
                if _val.mut_.is_some() {
                    f.field("mut_", &Present);
                }
                f.field("ident", Lite(&_val.ident));
                if let Some(val) = &_val.sub {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::At, Box<syn::pat::Pat>));
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("subpat", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::pat::Pat::Lit(_val) => {
                f.write_str("pat::Pat::Lit")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::pat::Pat::Mac(_val) => {
                f.write_str("pat::Pat::Macro")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::pat::Pat::Or(_val) => {
                let mut f = f.debug_struct("pat::Pat::Or");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if _val.vert.is_some() {
                    f.field("leading_vert", &Present);
                }
                if !_val.cases.is_empty() {
                    f.field("cases", Lite(&_val.cases));
                }
                f.finish()
            },
            syn::pat::Pat::Paren(_val) => {
                let mut f = f.debug_struct("pat::Pat::Paren");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("pat", Lite(&_val.patt));
                f.finish()
            },
            syn::pat::Pat::Path(_val) => {
                f.write_str("pat::Pat::Path")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::pat::Pat::Range(_val) => {
                f.write_str("pat::Pat::Range")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::pat::Pat::Ref(_val) => {
                let mut f = f.debug_struct("pat::Pat::Reference");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if _val.mut_.is_some() {
                    f.field("mut_", &Present);
                }
                f.field("pat", Lite(&_val.patt));
                f.finish()
            },
            syn::pat::Pat::Rest(_val) => {
                let mut f = f.debug_struct("pat::Pat::Rest");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.finish()
            },
            syn::pat::Pat::Slice(_val) => {
                let mut f = f.debug_struct("pat::Pat::Slice");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    f.field("elems", Lite(&_val.elems));
                }
                f.finish()
            },
            syn::pat::Pat::Struct(_val) => {
                let mut f = f.debug_struct("pat::Pat::Struct");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("qself", Print::ref_cast(val));
                }
                f.field("path", Lite(&_val.path));
                if !_val.fields.is_empty() {
                    f.field("fields", Lite(&_val.fields));
                }
                if let Some(val) = &_val.rest {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::patt::Rest);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("rest", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::pat::Pat::Tuple(_val) => {
                let mut f = f.debug_struct("pat::Pat::Tuple");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    f.field("elems", Lite(&_val.elems));
                }
                f.finish()
            },
            syn::pat::Pat::TupleStruct(_val) => {
                let mut f = f.debug_struct("pat::Pat::TupleStruct");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("qself", Print::ref_cast(val));
                }
                f.field("path", Lite(&_val.path));
                if !_val.elems.is_empty() {
                    f.field("elems", Lite(&_val.elems));
                }
                f.finish()
            },
            syn::pat::Pat::Type(_val) => {
                let mut f = f.debug_struct("pat::Pat::Type");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("pat", Lite(&_val.patt));
                f.field("ty", Lite(&_val.typ));
                f.finish()
            },
            syn::pat::Pat::Stream(_val) => {
                f.write_str("pat::Pat::Stream")?;
                f.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                f.write_str("`)")?;
                Ok(())
            },
            syn::pat::Pat::Wild(_val) => {
                let mut f = f.debug_struct("pat::Pat::Wild");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.finish()
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::patt::Ident> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::Ident");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.ref_.is_some() {
            f.field("by_ref", &Present);
        }
        if self.value.mut_.is_some() {
            f.field("mut_", &Present);
        }
        f.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.sub {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::At, Box<syn::pat::Pat>));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("subpat", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::patt::Or> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::Or");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.vert.is_some() {
            f.field("leading_vert", &Present);
        }
        if !self.value.cases.is_empty() {
            f.field("cases", Lite(&self.value.cases));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::patt::Paren> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::Paren");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("pat", Lite(&self.value.patt));
        f.finish()
    }
}
impl Debug for Lite<syn::patt::Ref> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::Ref");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.mut_.is_some() {
            f.field("mut_", &Present);
        }
        f.field("pat", Lite(&self.value.patt));
        f.finish()
    }
}
impl Debug for Lite<syn::patt::Rest> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::Rest");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::patt::Slice> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::Slice");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            f.field("elems", Lite(&self.value.elems));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::patt::Struct> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::Struct");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("qself", Print::ref_cast(val));
        }
        f.field("path", Lite(&self.value.path));
        if !self.value.fields.is_empty() {
            f.field("fields", Lite(&self.value.fields));
        }
        if let Some(val) = &self.value.rest {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::patt::Rest);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("rest", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::patt::Tuple> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::Tuple");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            f.field("elems", Lite(&self.value.elems));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::patt::TupleStruct> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::TupleStruct");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("qself", Print::ref_cast(val));
        }
        f.field("path", Lite(&self.value.path));
        if !self.value.elems.is_empty() {
            f.field("elems", Lite(&self.value.elems));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::patt::Type> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::Type");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("pat", Lite(&self.value.patt));
        f.field("ty", Lite(&self.value.typ));
        f.finish()
    }
}
impl Debug for Lite<syn::patt::Wild> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("patt::Wild");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::Path> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Path");
        if self.value.colon.is_some() {
            f.field("leading_colon", &Present);
        }
        if !self.value.segs.is_empty() {
            f.field("segments", Lite(&self.value.segs));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::Args> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Args::None => f.write_str("path::Args::None"),
            syn::Args::Angled(_val) => {
                let mut f = f.debug_struct("path::Args::AngleBracketed");
                if _val.colon2.is_some() {
                    f.field("colon2", &Present);
                }
                if !_val.args.is_empty() {
                    f.field("args", Lite(&_val.args));
                }
                f.finish()
            },
            syn::Args::Parenthesized(_val) => {
                let mut f = f.debug_struct("path::Args::Parenthesized");
                if !_val.ins.is_empty() {
                    f.field("inputs", Lite(&_val.ins));
                }
                f.field("output", Lite(&_val.out));
                f.finish()
            },
        }
    }
}
impl Debug for Lite<syn::Segment> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("path::Segment");
        f.field("ident", Lite(&self.value.ident));
        match self.value.args {
            syn::Args::None => {},
            _ => {
                f.field("arguments", Lite(&self.value.args));
            },
        }
        f.finish()
    }
}
impl Debug for Lite<syn::gen::Where::Life> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("PredicateLifetime");
        f.field("life", Lite(&self.value.life));
        if !self.value.bounds.is_empty() {
            f.field("bounds", Lite(&self.value.bounds));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::gen::Where::Type> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("PredicateType");
        if let Some(val) = &self.value.lifes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Bgen::bound::Lifes);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("lifetimes", Print::ref_cast(val));
        }
        f.field("bounded", Lite(&self.value.bounded));
        if !self.value.bounds.is_empty() {
            f.field("bounds", Lite(&self.value.bounds));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::QSelf> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("QSelf");
        f.field("ty", Lite(&self.value.ty));
        f.field("position", Lite(&self.value.pos));
        if self.value.as_.is_some() {
            f.field("as_", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::expr::Limits> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::expr::Limits::HalfOpen(_val) => {
                f.write_str("expr::Limits::HalfOpen")?;
                Ok(())
            },
            syn::expr::Limits::Closed(_val) => {
                f.write_str("expr::Limits::Closed")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::item::Receiver> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Receiver");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.ref_ {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::And, Option<syn::Life>));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(
                        {
                            #[derive(RefCast)]
                            #[repr(transparent)]
                            struct Print(Option<syn::Life>);
                            impl Debug for Print {
                                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                                    match &self.0 {
                                        Some(_val) => {
                                            f.write_str("Some(")?;
                                            Debug::fmt(Lite(_val), formatter)?;
                                            f.write_str(")")?;
                                            Ok(())
                                        },
                                        None => f.write_str("None"),
                                    }
                                }
                            }
                            Print::ref_cast(&self.0 .1)
                        },
                        formatter,
                    )?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("reference", Print::ref_cast(val));
        }
        if self.value.mut_.is_some() {
            f.field("mut_", &Present);
        }
        if self.value.colon.is_some() {
            f.field("colon", &Present);
        }
        f.field("ty", Lite(&self.value.typ));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Ret> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::typ::Ret::Default => f.write_str("typ::Ret::Default"),
            syn::typ::Ret::Type(_v0, _v1) => {
                let mut f = f.debug_tuple("typ::Ret::Type");
                f.field(Lite(_v1));
                f.finish()
            },
        }
    }
}
impl Debug for Lite<syn::item::Sig> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Sig");
        if self.value.const_.is_some() {
            f.field("const_", &Present);
        }
        if self.value.async_.is_some() {
            f.field("asyncness", &Present);
        }
        if self.value.unsafe_.is_some() {
            f.field("unsafe_", &Present);
        }
        if let Some(val) = &self.value.abi {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Abi);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("abi", Print::ref_cast(val));
        }
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        if !self.value.args.is_empty() {
            f.field("inputs", Lite(&self.value.args));
        }
        if let Some(val) = &self.value.vari {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::item::Variadic);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("vari", Print::ref_cast(val));
        }
        f.field("output", Lite(&self.value.ret));
        f.finish()
    }
}
impl Debug for Lite<syn::StaticMut> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::StaticMut::Mut(_val) => {
                f.write_str("StaticMutability::Mut")?;
                Ok(())
            },
            syn::StaticMut::None => f.write_str("StaticMutability::None"),
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::stmt::Stmt> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::stmt::Stmt::stmt::Local(_val) => {
                let mut f = f.debug_struct("stmt::Stmt::stmt::Local");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("pat", Lite(&_val.pat));
                if let Some(val) = &_val.init {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::stmt::Init);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("init", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::stmt::Stmt::Item(_val) => {
                f.write_str("stmt::Stmt::Item")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::stmt::Stmt::Expr(_v0, _v1) => {
                let mut f = f.debug_tuple("stmt::Stmt::Expr");
                f.field(Lite(_v0));
                f.field(&super::Option { present: _v1.is_some() });
                f.finish()
            },
            syn::stmt::Stmt::Mac(_val) => {
                let mut f = f.debug_struct("stmt::Stmt::Macro");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("mac", Lite(&_val.mac));
                if _val.semi.is_some() {
                    f.field("semi", &Present);
                }
                f.finish()
            },
        }
    }
}
impl Debug for Lite<syn::stmt::Mac> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("stmt::Mac");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("mac", Lite(&self.value.mac));
        if self.value.semi.is_some() {
            f.field("semi", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::gen::bound::Trait> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::bound::Trait");
        if self.value.paren.is_some() {
            f.field("paren", &Present);
        }
        match self.value.modif {
            syn::gen::bound::Modifier::None => {},
            _ => {
                f.field("modifier", Lite(&self.value.modif));
            },
        }
        if let Some(val) = &self.value.lifes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Bgen::bound::Lifes);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("lifetimes", Print::ref_cast(val));
        }
        f.field("path", Lite(&self.value.path));
        f.finish()
    }
}
impl Debug for Lite<syn::gen::bound::Modifier> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::gen::bound::Modifier::None => f.write_str("gen::bound::Modifier::None"),
            syn::gen::bound::Modifier::Maybe(_val) => {
                f.write_str("gen::bound::Modifier::Maybe")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::item::Trait::Item> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::item::Trait::Item::Const(_val) => {
                let mut f = f.debug_struct("item::Trait::Item::Const");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                f.field("ty", Lite(&_val.typ));
                if let Some(val) = &_val.default {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Eq, syn::Expr));
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("default", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::item::Trait::Item::Fn(_val) => {
                let mut f = f.debug_struct("item::Trait::Item::Fn");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("sig", Lite(&_val.sig));
                if let Some(val) = &_val.default {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Block);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("default", Print::ref_cast(val));
                }
                if _val.semi.is_some() {
                    f.field("semi", &Present);
                }
                f.finish()
            },
            syn::item::Trait::Item::Type(_val) => {
                let mut f = f.debug_struct("item::Trait::Item::Type");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("ident", Lite(&_val.ident));
                f.field("gens", Lite(&_val.gens));
                if _val.colon.is_some() {
                    f.field("colon", &Present);
                }
                if !_val.bounds.is_empty() {
                    f.field("bounds", Lite(&_val.bounds));
                }
                if let Some(val) = &_val.default {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Eq, syn::typ::Type));
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("default", Print::ref_cast(val));
                }
                f.finish()
            },
            syn::item::Trait::Item::Macro(_val) => {
                let mut f = f.debug_struct("item::Trait::Item::Macro");
                if !_val.attrs.is_empty() {
                    f.field("attrs", Lite(&_val.attrs));
                }
                f.field("mac", Lite(&_val.mac));
                if _val.semi.is_some() {
                    f.field("semi", &Present);
                }
                f.finish()
            },
            syn::item::Trait::Item::Stream(_val) => {
                f.write_str("item::Trait::Item::Stream")?;
                f.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                f.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::item::Trait::Const> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Trait::Const");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        f.field("ty", Lite(&self.value.typ));
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Eq, syn::Expr));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("default", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Trait::Fn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Trait::Fn");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("sig", Lite(&self.value.sig));
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Block);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("default", Print::ref_cast(val));
        }
        if self.value.semi.is_some() {
            f.field("semi", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Trait::Mac> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Trait::Mac");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("mac", Lite(&self.value.mac));
        if self.value.semi.is_some() {
            f.field("semi", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Trait::Type> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Trait::Type");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("ident", Lite(&self.value.ident));
        f.field("gens", Lite(&self.value.gens));
        if self.value.colon.is_some() {
            f.field("colon", &Present);
        }
        if !self.value.bounds.is_empty() {
            f.field("bounds", Lite(&self.value.bounds));
        }
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Eq, syn::typ::Type));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("default", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Type> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::typ::Type::Array(_val) => {
                let mut f = f.debug_struct("Type::Array");
                f.field("elem", Lite(&_val.elem));
                f.field("len", Lite(&_val.len));
                f.finish()
            },
            syn::typ::Type::Fn(_val) => {
                let mut f = f.debug_struct("Type::Fn");
                if let Some(val) = &_val.lifes {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Bgen::bound::Lifes);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("lifetimes", Print::ref_cast(val));
                }
                if _val.unsafe_.is_some() {
                    f.field("unsafe_", &Present);
                }
                if let Some(val) = &_val.abi {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Abi);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("abi", Print::ref_cast(val));
                }
                if !_val.args.is_empty() {
                    f.field("inputs", Lite(&_val.args));
                }
                if let Some(val) = &_val.vari {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::typ::Variadic);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("vari", Print::ref_cast(val));
                }
                f.field("output", Lite(&_val.ret));
                f.finish()
            },
            syn::typ::Type::Group(_val) => {
                let mut f = f.debug_struct("Type::Group");
                f.field("elem", Lite(&_val.elem));
                f.finish()
            },
            syn::typ::Type::Impl(_val) => {
                let mut f = f.debug_struct("Type::ImplTrait");
                if !_val.bounds.is_empty() {
                    f.field("bounds", Lite(&_val.bounds));
                }
                f.finish()
            },
            syn::typ::Type::Infer(_val) => {
                let mut f = f.debug_struct("Type::Infer");
                f.finish()
            },
            syn::typ::Type::Mac(_val) => {
                let mut f = f.debug_struct("Type::Macro");
                f.field("mac", Lite(&_val.mac));
                f.finish()
            },
            syn::typ::Type::Never(_val) => {
                let mut f = f.debug_struct("Type::Never");
                f.finish()
            },
            syn::typ::Type::Paren(_val) => {
                let mut f = f.debug_struct("Type::Paren");
                f.field("elem", Lite(&_val.elem));
                f.finish()
            },
            syn::typ::Type::Path(_val) => {
                let mut f = f.debug_struct("Type::Path");
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("qself", Print::ref_cast(val));
                }
                f.field("path", Lite(&_val.path));
                f.finish()
            },
            syn::typ::Type::Ptr(_val) => {
                let mut f = f.debug_struct("Type::Ptr");
                if _val.const_.is_some() {
                    f.field("const_", &Present);
                }
                if _val.mut_.is_some() {
                    f.field("mut_", &Present);
                }
                f.field("elem", Lite(&_val.elem));
                f.finish()
            },
            syn::typ::Type::Ref(_val) => {
                let mut f = f.debug_struct("Type::Reference");
                if let Some(val) = &_val.life {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Life);
                    impl Debug for Print {
                        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            f.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            f.write_str(")")?;
                            Ok(())
                        }
                    }
                    f.field("life", Print::ref_cast(val));
                }
                if _val.mut_.is_some() {
                    f.field("mut_", &Present);
                }
                f.field("elem", Lite(&_val.elem));
                f.finish()
            },
            syn::typ::Type::Slice(_val) => {
                let mut f = f.debug_struct("Type::Slice");
                f.field("elem", Lite(&_val.elem));
                f.finish()
            },
            syn::typ::Type::Trait(_val) => {
                let mut f = f.debug_struct("Type::TraitObject");
                if _val.dyn_.is_some() {
                    f.field("dyn_", &Present);
                }
                if !_val.bounds.is_empty() {
                    f.field("bounds", Lite(&_val.bounds));
                }
                f.finish()
            },
            syn::typ::Type::Tuple(_val) => {
                let mut f = f.debug_struct("Type::Tuple");
                if !_val.elems.is_empty() {
                    f.field("elems", Lite(&_val.elems));
                }
                f.finish()
            },
            syn::typ::Type::Stream(_val) => {
                f.write_str("Type::Stream")?;
                f.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                f.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::typ::Array> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Array");
        f.field("elem", Lite(&self.value.elem));
        f.field("len", Lite(&self.value.len));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Fn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Fn");
        if let Some(val) = &self.value.lifes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Bgen::bound::Lifes);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("lifetimes", Print::ref_cast(val));
        }
        if self.value.unsafe_.is_some() {
            f.field("unsafe_", &Present);
        }
        if let Some(val) = &self.value.abi {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Abi);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("abi", Print::ref_cast(val));
        }
        if !self.value.args.is_empty() {
            f.field("inputs", Lite(&self.value.args));
        }
        if let Some(val) = &self.value.vari {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::typ::Variadic);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("vari", Print::ref_cast(val));
        }
        f.field("output", Lite(&self.value.ret));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Group> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Group");
        f.field("elem", Lite(&self.value.elem));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Impl> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Impl");
        if !self.value.bounds.is_empty() {
            f.field("bounds", Lite(&self.value.bounds));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Infer> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Infer");
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Mac> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Mac");
        f.field("mac", Lite(&self.value.mac));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Never> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Never");
        f.finish()
    }
}
impl Debug for Lite<syn::gen::param::Type> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::param::Type");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("ident", Lite(&self.value.ident));
        if self.value.colon.is_some() {
            f.field("colon", &Present);
        }
        if !self.value.bounds.is_empty() {
            f.field("bounds", Lite(&self.value.bounds));
        }
        if self.value.eq.is_some() {
            f.field("eq", &Present);
        }
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::typ::Type);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("default", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::gen::bound::Type> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::gen::bound::Type::Trait(_val) => {
                f.write_str("gen::bound::Type::Trait")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::gen::bound::Type::Life(_val) => {
                let mut f = f.debug_struct("gen::bound::Type::Life");
                f.field("ident", Lite(&_val.ident));
                f.finish()
            },
            syn::gen::bound::Type::Stream(_val) => {
                f.write_str("gen::bound::Type::Stream")?;
                f.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                f.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::typ::Paren> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Paren");
        f.field("elem", Lite(&self.value.elem));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Path> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Path");
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("qself", Print::ref_cast(val));
        }
        f.field("path", Lite(&self.value.path));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Ptr> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Ptr");
        if self.value.const_.is_some() {
            f.field("const_", &Present);
        }
        if self.value.mut_.is_some() {
            f.field("mut_", &Present);
        }
        f.field("elem", Lite(&self.value.elem));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Ref> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Ref");
        if let Some(val) = &self.value.life {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Life);
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("life", Print::ref_cast(val));
        }
        if self.value.mut_.is_some() {
            f.field("mut_", &Present);
        }
        f.field("elem", Lite(&self.value.elem));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Slice> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Slice");
        f.field("elem", Lite(&self.value.elem));
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Trait> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Trait");
        if self.value.dyn_.is_some() {
            f.field("dyn_", &Present);
        }
        if !self.value.bounds.is_empty() {
            f.field("bounds", Lite(&self.value.bounds));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::typ::Tuple> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Tuple");
        if !self.value.elems.is_empty() {
            f.field("elems", Lite(&self.value.elems));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::UnOp> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::UnOp::Deref(_val) => {
                f.write_str("UnOp::Deref")?;
                Ok(())
            },
            syn::UnOp::Not(_val) => {
                f.write_str("UnOp::Not")?;
                Ok(())
            },
            syn::UnOp::Neg(_val) => {
                f.write_str("UnOp::Neg")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::item::Use::Glob> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Use::Glob");
        f.finish()
    }
}
impl Debug for Lite<syn::item::Use::Group> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Use::Group");
        if !self.value.elems.is_empty() {
            f.field("items", Lite(&self.value.elems));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::item::Use::Name> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Use::Name");
        f.field("ident", Lite(&self.value.ident));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Use::Path> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Use::Path");
        f.field("ident", Lite(&self.value.ident));
        f.field("tree", Lite(&self.value.tree));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Use::Rename> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Use::Rename");
        f.field("ident", Lite(&self.value.ident));
        f.field("rename", Lite(&self.value.rename));
        f.finish()
    }
}
impl Debug for Lite<syn::item::Use::Tree> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::item::Use::Tree::Path(_val) => {
                f.write_str("item::Use::Tree::Path")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::item::Use::Tree::Name(_val) => {
                f.write_str("item::Use::Tree::Name")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::item::Use::Tree::Rename(_val) => {
                f.write_str("item::Use::Tree::Rename")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::item::Use::Tree::Glob(_val) => {
                f.write_str("item::Use::Tree::Glob")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::item::Use::Tree::Group(_val) => {
                f.write_str("item::Use::Tree::Group")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::item::Variadic> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Variadic");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.pat {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((Box<syn::pat::Pat>, syn::tok::Colon));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .0), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("pat", Print::ref_cast(val));
        }
        if self.value.comma.is_some() {
            f.field("comma", &Present);
        }
        f.finish()
    }
}
impl Debug for Lite<syn::Variant> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Variant");
        if !self.value.attrs.is_empty() {
            f.field("attrs", Lite(&self.value.attrs));
        }
        f.field("ident", Lite(&self.value.ident));
        f.field("fields", Lite(&self.value.fields));
        if let Some(val) = &self.value.discriminant {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Eq, syn::Expr));
            impl Debug for Print {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    f.write_str(")")?;
                    Ok(())
                }
            }
            f.field("discriminant", Print::ref_cast(val));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::data::Restricted> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("data::Restricted");
        if self.value.in_.is_some() {
            f.field("in_", &Present);
        }
        f.field("path", Lite(&self.value.path));
        f.finish()
    }
}
impl Debug for Lite<syn::data::Visibility> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::data::Visibility::Public(_val) => {
                f.write_str("data::Visibility::Public")?;
                Ok(())
            },
            syn::data::Visibility::Restricted(_val) => {
                let mut f = f.debug_struct("data::Visibility::Restricted");
                if _val.in_.is_some() {
                    f.field("in_", &Present);
                }
                f.field("path", Lite(&_val.path));
                f.finish()
            },
            syn::data::Visibility::Inherited => f.write_str("data::Visibility::Inherited"),
        }
    }
}
impl Debug for Lite<syn::gen::Where> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::Where");
        if !self.value.preds.is_empty() {
            f.field("predicates", Lite(&self.value.preds));
        }
        f.finish()
    }
}
impl Debug for Lite<syn::gen::Where::Pred> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::gen::Where::Pred::Life(_val) => {
                f.write_str("WherePredicate::Life")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            syn::gen::Where::Pred::Type(_val) => {
                f.write_str("WherePredicate::Type")?;
                f.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                f.write_str(")")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
