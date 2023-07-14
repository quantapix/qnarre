#![allow(clippy::match_wildcard_for_single_variants)]
use super::{Lite, Present};
use ref_cast::RefCast;
use std::fmt::{self, Debug, Display};
impl Debug for Lite<syn::Abi> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Abi");
        if let Some(val) = &self.value.name {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::lit::Str);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("name", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::AngledArgs> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("path::path::AngledArgs");
        if self.value.colon2.is_some() {
            formatter.field("colon2", &Present);
        }
        if !self.value.args.is_empty() {
            formatter.field("args", Lite(&self.value.args));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Arm> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Arm");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("pat", Lite(&self.value.pat));
        if let Some(val) = &self.value.guard {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::If, Box<syn::Expr>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("guard", Print::ref_cast(val));
        }
        formatter.field("body", Lite(&self.value.body));
        if self.value.comma.is_some() {
            formatter.field("comma", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::AssocConst> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("AssocConst");
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.gnrs {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngledArgs);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("gens", Print::ref_cast(val));
        }
        formatter.field("value", Lite(&self.value.val));
        formatter.finish()
    }
}
impl Debug for Lite<syn::AssocType> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("AssocType");
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.gnrs {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngledArgs);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("gens", Print::ref_cast(val));
        }
        formatter.field("ty", Lite(&self.value.ty));
        formatter.finish()
    }
}
impl Debug for Lite<syn::attr::Style> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::attr::Style::Outer => formatter.write_str("attr::Style::Outer"),
            syn::attr::Style::Inner(_val) => {
                formatter.write_str("attr::Style::Inner")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::attr::Attr> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("attr::Attr");
        formatter.field("style", Lite(&self.value.style));
        formatter.field("meta", Lite(&self.value.meta));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::FnArg> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::FnArg");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.name {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((proc_macro2::Ident, syn::tok::Colon));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("name", Print::ref_cast(val));
        }
        formatter.field("ty", Lite(&self.value.ty));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Variadic> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Variadic");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.name {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((proc_macro2::Ident, syn::tok::Colon));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("name", Print::ref_cast(val));
        }
        if self.value.comma.is_some() {
            formatter.field("comma", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::BinOp> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::BinOp::Add(_val) => {
                formatter.write_str("BinOp::Add")?;
                Ok(())
            },
            syn::BinOp::Sub(_val) => {
                formatter.write_str("BinOp::Sub")?;
                Ok(())
            },
            syn::BinOp::Mul(_val) => {
                formatter.write_str("BinOp::Mul")?;
                Ok(())
            },
            syn::BinOp::Div(_val) => {
                formatter.write_str("BinOp::Div")?;
                Ok(())
            },
            syn::BinOp::Rem(_val) => {
                formatter.write_str("BinOp::Rem")?;
                Ok(())
            },
            syn::BinOp::And(_val) => {
                formatter.write_str("BinOp::And")?;
                Ok(())
            },
            syn::BinOp::Or(_val) => {
                formatter.write_str("BinOp::Or")?;
                Ok(())
            },
            syn::BinOp::BitXor(_val) => {
                formatter.write_str("BinOp::BitXor")?;
                Ok(())
            },
            syn::BinOp::BitAnd(_val) => {
                formatter.write_str("BinOp::BitAnd")?;
                Ok(())
            },
            syn::BinOp::BitOr(_val) => {
                formatter.write_str("BinOp::BitOr")?;
                Ok(())
            },
            syn::BinOp::Shl(_val) => {
                formatter.write_str("BinOp::Shl")?;
                Ok(())
            },
            syn::BinOp::Shr(_val) => {
                formatter.write_str("BinOp::Shr")?;
                Ok(())
            },
            syn::BinOp::Eq(_val) => {
                formatter.write_str("BinOp::Eq")?;
                Ok(())
            },
            syn::BinOp::Lt(_val) => {
                formatter.write_str("BinOp::Lt")?;
                Ok(())
            },
            syn::BinOp::Le(_val) => {
                formatter.write_str("BinOp::Le")?;
                Ok(())
            },
            syn::BinOp::Ne(_val) => {
                formatter.write_str("BinOp::Ne")?;
                Ok(())
            },
            syn::BinOp::Ge(_val) => {
                formatter.write_str("BinOp::Ge")?;
                Ok(())
            },
            syn::BinOp::Gt(_val) => {
                formatter.write_str("BinOp::Gt")?;
                Ok(())
            },
            syn::BinOp::AddAssign(_val) => {
                formatter.write_str("BinOp::AddAssign")?;
                Ok(())
            },
            syn::BinOp::SubAssign(_val) => {
                formatter.write_str("BinOp::SubAssign")?;
                Ok(())
            },
            syn::BinOp::MulAssign(_val) => {
                formatter.write_str("BinOp::MulAssign")?;
                Ok(())
            },
            syn::BinOp::DivAssign(_val) => {
                formatter.write_str("BinOp::DivAssign")?;
                Ok(())
            },
            syn::BinOp::RemAssign(_val) => {
                formatter.write_str("BinOp::RemAssign")?;
                Ok(())
            },
            syn::BinOp::BitXorAssign(_val) => {
                formatter.write_str("BinOp::BitXorAssign")?;
                Ok(())
            },
            syn::BinOp::BitAndAssign(_val) => {
                formatter.write_str("BinOp::BitAndAssign")?;
                Ok(())
            },
            syn::BinOp::BitOrAssign(_val) => {
                formatter.write_str("BinOp::BitOrAssign")?;
                Ok(())
            },
            syn::BinOp::ShlAssign(_val) => {
                formatter.write_str("BinOp::ShlAssign")?;
                Ok(())
            },
            syn::BinOp::ShrAssign(_val) => {
                formatter.write_str("BinOp::ShrAssign")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::Block> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Block");
        if !self.value.stmts.is_empty() {
            formatter.field("stmts", Lite(&self.value.stmts));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Bgen::bound::Lifes> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Bgen::bound::Lifes");
        if !self.value.lifes.is_empty() {
            formatter.field("lifetimes", Lite(&self.value.lifes));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::gen::param::Const> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::param::Const");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("ty", Lite(&self.value.typ));
        if self.value.eq.is_some() {
            formatter.field("eq", &Present);
        }
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Expr);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("default", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Constraint> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Constraint");
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.gnrs {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngledArgs);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("gens", Print::ref_cast(val));
        }
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Data> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Data::Struct(_val) => {
                let mut formatter = formatter.debug_struct("Data::Struct");
                formatter.field("fields", Lite(&_val.fields));
                if _val.semi.is_some() {
                    formatter.field("semi", &Present);
                }
                formatter.finish()
            },
            syn::Data::Enum(_val) => {
                let mut formatter = formatter.debug_struct("Data::Enum");
                if !_val.variants.is_empty() {
                    formatter.field("variants", Lite(&_val.variants));
                }
                formatter.finish()
            },
            syn::Data::Union(_val) => {
                let mut formatter = formatter.debug_struct("Data::Union");
                formatter.field("fields", Lite(&_val.fields));
                formatter.finish()
            },
        }
    }
}
impl Debug for Lite<syn::DataEnum> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("DataEnum");
        if !self.value.variants.is_empty() {
            formatter.field("variants", Lite(&self.value.variants));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::DataStruct> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("DataStruct");
        formatter.field("fields", Lite(&self.value.fields));
        if self.value.semi.is_some() {
            formatter.field("semi", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::DataUnion> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("DataUnion");
        formatter.field("fields", Lite(&self.value.fields));
        formatter.finish()
    }
}
impl Debug for Lite<syn::DeriveInput> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("DeriveInput");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        formatter.field("data", Lite(&self.value.data));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Expr> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Expr::Array(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Array");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::Expr::Assign(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Assign");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("left", Lite(&_val.left));
                formatter.field("right", Lite(&_val.right));
                formatter.finish()
            },
            syn::Expr::Async(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Async");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.move_.is_some() {
                    formatter.field("capture", &Present);
                }
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Expr::Await(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Await");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("base", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::Binary(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Binary");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("left", Lite(&_val.left));
                formatter.field("op", Lite(&_val.op));
                formatter.field("right", Lite(&_val.right));
                formatter.finish()
            },
            syn::Expr::Block(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Block");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Expr::Break(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Break");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Life);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                if let Some(val) = &_val.expr {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("expr", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Call(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Call");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("func", Lite(&_val.func));
                if !_val.args.is_empty() {
                    formatter.field("args", Lite(&_val.args));
                }
                formatter.finish()
            },
            syn::Expr::Cast(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Cast");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.field("ty", Lite(&_val.typ));
                formatter.finish()
            },
            syn::Expr::Closure(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Closure");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.lifes {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Bgen::bound::Lifes);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("lifetimes", Print::ref_cast(val));
                }
                if _val.const_.is_some() {
                    formatter.field("const_", &Present);
                }
                if _val.static_.is_some() {
                    formatter.field("movability", &Present);
                }
                if _val.async_.is_some() {
                    formatter.field("asyncness", &Present);
                }
                if _val.move_.is_some() {
                    formatter.field("capture", &Present);
                }
                if !_val.inputs.is_empty() {
                    formatter.field("inputs", Lite(&_val.inputs));
                }
                formatter.field("output", Lite(&_val.ret));
                formatter.field("body", Lite(&_val.body));
                formatter.finish()
            },
            syn::Expr::Const(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Const");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Expr::Continue(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Continue");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Life);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Field(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Field");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("base", Lite(&_val.base));
                formatter.field("member", Lite(&_val.member));
                formatter.finish()
            },
            syn::Expr::ForLoop(_val) => {
                let mut formatter = formatter.debug_struct("Expr::ForLoop");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                formatter.field("pat", Lite(&_val.pat));
                formatter.field("expr", Lite(&_val.expr));
                formatter.field("body", Lite(&_val.body));
                formatter.finish()
            },
            syn::Expr::Group(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Group");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::If(_val) => {
                let mut formatter = formatter.debug_struct("Expr::If");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("cond", Lite(&_val.cond));
                formatter.field("then_branch", Lite(&_val.then_branch));
                if let Some(val) = &_val.else_branch {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Else, Box<syn::Expr>));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("else_branch", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Index(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Index");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.field("index", Lite(&_val.index));
                formatter.finish()
            },
            syn::Expr::Infer(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Infer");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.finish()
            },
            syn::Expr::Let(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Let");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("pat", Lite(&_val.pat));
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::Lit(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Lit");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("lit", Lite(&_val.lit));
                formatter.finish()
            },
            syn::Expr::Loop(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Loop");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                formatter.field("body", Lite(&_val.body));
                formatter.finish()
            },
            syn::Expr::Macro(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("mac", Lite(&_val.mac));
                formatter.finish()
            },
            syn::Expr::Match(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Match");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                if !_val.arms.is_empty() {
                    formatter.field("arms", Lite(&_val.arms));
                }
                formatter.finish()
            },
            syn::Expr::MethodCall(_val) => {
                let mut formatter = formatter.debug_struct("Expr::MethodCall");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("receiver", Lite(&_val.receiver));
                formatter.field("method", Lite(&_val.method));
                if let Some(val) = &_val.turbofish {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::AngledArgs);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("turbofish", Print::ref_cast(val));
                }
                if !_val.args.is_empty() {
                    formatter.field("args", Lite(&_val.args));
                }
                formatter.finish()
            },
            syn::Expr::Paren(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Paren");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::Path(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Path");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("qself", Print::ref_cast(val));
                }
                formatter.field("path", Lite(&_val.path));
                formatter.finish()
            },
            syn::Expr::Range(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Range");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.start {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("start", Print::ref_cast(val));
                }
                formatter.field("limits", Lite(&_val.limits));
                if let Some(val) = &_val.end {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("end", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Reference(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Reference");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.mut_.is_some() {
                    formatter.field("mut_", &Present);
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::Repeat(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Repeat");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.field("len", Lite(&_val.len));
                formatter.finish()
            },
            syn::Expr::Return(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Return");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.expr {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("expr", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Struct(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Struct");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("qself", Print::ref_cast(val));
                }
                formatter.field("path", Lite(&_val.path));
                if !_val.fields.is_empty() {
                    formatter.field("fields", Lite(&_val.fields));
                }
                if _val.dot2.is_some() {
                    formatter.field("dot2", &Present);
                }
                if let Some(val) = &_val.rest {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("rest", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Try(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Try");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::TryBlock(_val) => {
                let mut formatter = formatter.debug_struct("Expr::TryBlock");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Expr::Tuple(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Tuple");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::Expr::Unary(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Unary");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("op", Lite(&_val.op));
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::Unsafe(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Unsafe");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Expr::Verbatim(_val) => {
                formatter.write_str("Expr::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            syn::Expr::While(_val) => {
                let mut formatter = formatter.debug_struct("Expr::While");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                formatter.field("cond", Lite(&_val.cond));
                formatter.field("body", Lite(&_val.body));
                formatter.finish()
            },
            syn::Expr::Yield(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Yield");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.expr {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("expr", Print::ref_cast(val));
                }
                formatter.finish()
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::expr::Array> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Array");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Assign> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Assign");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("left", Lite(&self.value.left));
        formatter.field("right", Lite(&self.value.right));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Async> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Async");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.move_.is_some() {
            formatter.field("capture", &Present);
        }
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Await> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Await");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("base", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Binary> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Binary");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("left", Lite(&self.value.left));
        formatter.field("op", Lite(&self.value.op));
        formatter.field("right", Lite(&self.value.right));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Block> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Block");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Break> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Break");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Life);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        if let Some(val) = &self.value.expr {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("expr", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Call> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Call");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("func", Lite(&self.value.func));
        if !self.value.args.is_empty() {
            formatter.field("args", Lite(&self.value.args));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Cast> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Cast");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.field("ty", Lite(&self.value.typ));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Closure> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Closure");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.lifes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Bgen::bound::Lifes);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("lifetimes", Print::ref_cast(val));
        }
        if self.value.const_.is_some() {
            formatter.field("const_", &Present);
        }
        if self.value.static_.is_some() {
            formatter.field("movability", &Present);
        }
        if self.value.async_.is_some() {
            formatter.field("asyncness", &Present);
        }
        if self.value.move_.is_some() {
            formatter.field("capture", &Present);
        }
        if !self.value.inputs.is_empty() {
            formatter.field("inputs", Lite(&self.value.inputs));
        }
        formatter.field("output", Lite(&self.value.ret));
        formatter.field("body", Lite(&self.value.body));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Const> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Const");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Continue> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Continue");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Life);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Field> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Field");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("base", Lite(&self.value.base));
        formatter.field("member", Lite(&self.value.memb));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::ForLoop> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::ForLoop");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        formatter.field("pat", Lite(&self.value.pat));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.field("body", Lite(&self.value.body));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Group> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Group");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::If> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::If");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("cond", Lite(&self.value.cond));
        formatter.field("then_branch", Lite(&self.value.then_branch));
        if let Some(val) = &self.value.else_branch {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Else, Box<syn::Expr>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("else_branch", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Index> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Index");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.field("index", Lite(&self.value.index));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Infer> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Infer");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Let> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Let");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("pat", Lite(&self.value.pat));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Lit> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Lit");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("lit", Lite(&self.value.lit));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Loop> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Loop");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        formatter.field("body", Lite(&self.value.body));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Mac> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Mac");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("mac", Lite(&self.value.mac));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Match> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Match");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        if !self.value.arms.is_empty() {
            formatter.field("arms", Lite(&self.value.arms));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::MethodCall> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::MethodCall");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("receiver", Lite(&self.value.expr));
        formatter.field("method", Lite(&self.value.method));
        if let Some(val) = &self.value.turbofish {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngledArgs);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("turbofish", Print::ref_cast(val));
        }
        if !self.value.args.is_empty() {
            formatter.field("args", Lite(&self.value.args));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Paren> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Paren");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Path> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Path");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("qself", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Range> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Range");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.beg {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("start", Print::ref_cast(val));
        }
        formatter.field("limits", Lite(&self.value.limits));
        if let Some(val) = &self.value.end {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("end", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Ref> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Ref");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.mut_.is_some() {
            formatter.field("mut_", &Present);
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Repeat> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Repeat");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.field("len", Lite(&self.value.len));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Return> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Return");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.expr {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("expr", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::expr::ExprStruct> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::expr::ExprStruct");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("qself", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        if !self.value.fields.is_empty() {
            formatter.field("fields", Lite(&self.value.fields));
        }
        if self.value.dot2.is_some() {
            formatter.field("dot2", &Present);
        }
        if let Some(val) = &self.value.rest {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("rest", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Try> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Try");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::TryBlock> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::TryBlock");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Tuple> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Tuple");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Unary> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Unary");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("op", Lite(&self.value.op));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Unsafe> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Unsafe");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::While> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::While");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        formatter.field("cond", Lite(&self.value.cond));
        formatter.field("body", Lite(&self.value.body));
        formatter.finish()
    }
}
impl Debug for Lite<syn::expr::Yield> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("expr::Yield");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.expr {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("expr", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Field> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Field");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        match self.value.mut_ {
            syn::FieldMut::None => {},
            _ => {
                formatter.field("mut_", Lite(&self.value.mut_));
            },
        }
        if let Some(val) = &self.value.ident {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(proc_macro2::Ident);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("ident", Print::ref_cast(val));
        }
        if self.value.colon.is_some() {
            formatter.field("colon", &Present);
        }
        formatter.field("ty", Lite(&self.value.typ));
        formatter.finish()
    }
}
impl Debug for Lite<syn::FieldMut> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::FieldMut::None => formatter.write_str("Mut::None"),
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::patt::Field> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::Field");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("member", Lite(&self.value.member));
        if self.value.colon.is_some() {
            formatter.field("colon", &Present);
        }
        formatter.field("pat", Lite(&self.value.patt));
        formatter.finish()
    }
}
impl Debug for Lite<syn::FieldValue> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("FieldValue");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("member", Lite(&self.value.member));
        if self.value.colon.is_some() {
            formatter.field("colon", &Present);
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Fields> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Fields::Named(_val) => {
                let mut formatter = formatter.debug_struct("Fields::Named");
                if !_val.field.is_empty() {
                    formatter.field("fields", Lite(&_val.fields));
                }
                formatter.finish()
            },
            syn::Fields::Unnamed(_val) => {
                let mut formatter = formatter.debug_struct("Fields::Unnamed");
                if !_val.field.is_empty() {
                    formatter.field("fields", Lite(&_val.fields));
                }
                formatter.finish()
            },
            syn::Fields::Unit => formatter.write_str("Fields::Unit"),
        }
    }
}
impl Debug for Lite<syn::FieldsNamed> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("FieldsNamed");
        if !self.value.field.is_empty() {
            formatter.field("fields", Lite(&self.value.fields));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::FieldsUnnamed> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("FieldsUnnamed");
        if !self.value.field.is_empty() {
            formatter.field("fields", Lite(&self.value.fields));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::File> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::File");
        if let Some(val) = &self.value.shebang {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(String);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("shebang", Print::ref_cast(val));
        }
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.items.is_empty() {
            formatter.field("items", Lite(&self.value.items));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::FnArg> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::item::FnArg::Receiver(_val) => {
                formatter.write_str("item::FnArg::Receiver")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::item::FnArg::Type(_val) => {
                formatter.write_str("item::FnArg::Typed")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::item::Foreign::Item> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::item::Foreign::Item::Fn(_val) => {
                let mut formatter = formatter.debug_struct("item::Foreign::Item::Fn");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("sig", Lite(&_val.sig));
                formatter.finish()
            },
            syn::item::Foreign::Item::Static(_val) => {
                let mut formatter = formatter.debug_struct("item::Foreign::Item::Static");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                match _val.mut_ {
                    syn::StaticMut::None => {},
                    _ => {
                        formatter.field("mut_", Lite(&_val.mut_));
                    },
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("ty", Lite(&_val.typ));
                formatter.finish()
            },
            syn::item::Foreign::Item::Type(_val) => {
                let mut formatter = formatter.debug_struct("item::Foreign::Item::Type");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                formatter.finish()
            },
            syn::item::Foreign::Item::Macro(_val) => {
                let mut formatter = formatter.debug_struct("item::Foreign::Item::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("mac", Lite(&_val.mac));
                if _val.semi.is_some() {
                    formatter.field("semi", &Present);
                }
                formatter.finish()
            },
            syn::item::Foreign::Item::Verbatim(_val) => {
                formatter.write_str("item::Foreign::Item::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::item::Foreign::Fn> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Foreign::Fn");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("sig", Lite(&self.value.sig));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Foreign::Mac> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Foreign::Mac");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("mac", Lite(&self.value.mac));
        if self.value.semi.is_some() {
            formatter.field("semi", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Foreign::Static> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Foreign::Static");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        match self.value.mut_ {
            syn::StaticMut::None => {},
            _ => {
                formatter.field("mut_", Lite(&self.value.mut_));
            },
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("ty", Lite(&self.value.typ));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Foreign::Type> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Foreign::Type");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Arg> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Arg::Life(_val) => {
                formatter.write_str("path::Arg::Life")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Arg::Type(_val) => {
                formatter.write_str("path::Arg::Type")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Arg::Const(_val) => {
                formatter.write_str("path::Arg::Const")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Arg::AssocType(_val) => {
                formatter.write_str("path::Arg::AssocType")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Arg::AssocConst(_val) => {
                formatter.write_str("path::Arg::AssocConst")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Arg::Constraint(_val) => {
                formatter.write_str("path::Arg::Constraint")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::gen::Param> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::gen::Param::Life(_val) => {
                formatter.write_str("gen::Param::Life")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::gen::Param::Type(_val) => {
                formatter.write_str("gen::Param::Type")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::gen::Param::Const(_val) => {
                formatter.write_str("gen::Param::Const")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::gen::Gens> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::Gens");
        if self.value.lt.is_some() {
            formatter.field("lt", &Present);
        }
        if !self.value.params.is_empty() {
            formatter.field("params", Lite(&self.value.params));
        }
        if self.value.gt.is_some() {
            formatter.field("gt", &Present);
        }
        if let Some(val) = &self.value.where_ {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::gen::Where);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("where_clause", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Impl::Item> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::item::Impl::Item::Const(_val) => {
                let mut formatter = formatter.debug_struct("item::Impl::Item::Const");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.default_.is_some() {
                    formatter.field("defaultness", &Present);
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                formatter.field("ty", Lite(&_val.typ));
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::item::Impl::Item::Fn(_val) => {
                let mut formatter = formatter.debug_struct("item::Impl::Item::Fn");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.default_.is_some() {
                    formatter.field("defaultness", &Present);
                }
                formatter.field("sig", Lite(&_val.sig));
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::item::Impl::Item::Type(_val) => {
                let mut formatter = formatter.debug_struct("item::Impl::Item::Type");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.default_.is_some() {
                    formatter.field("defaultness", &Present);
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                formatter.field("ty", Lite(&_val.typ));
                formatter.finish()
            },
            syn::item::Impl::Item::Macro(_val) => {
                let mut formatter = formatter.debug_struct("item::Impl::Item::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("mac", Lite(&_val.mac));
                if _val.semi.is_some() {
                    formatter.field("semi", &Present);
                }
                formatter.finish()
            },
            syn::item::Impl::Item::Verbatim(_val) => {
                formatter.write_str("item::Impl::Item::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::item::Impl::Const> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Impl::Const");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.default_.is_some() {
            formatter.field("defaultness", &Present);
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        formatter.field("ty", Lite(&self.value.typ));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Impl::Fn> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Impl::Fn");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.default_.is_some() {
            formatter.field("defaultness", &Present);
        }
        formatter.field("sig", Lite(&self.value.sig));
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Impl::Mac> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Impl::Mac");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("mac", Lite(&self.value.mac));
        if self.value.semi.is_some() {
            formatter.field("semi", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Impl::Type> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Impl::Type");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.default_.is_some() {
            formatter.field("defaultness", &Present);
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        formatter.field("ty", Lite(&self.value.typ));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Impl::Restriction> {
    fn fmt(&self, _formatter: &mut fmt::Formatter) -> fmt::Result {
        unreachable!()
    }
}
impl Debug for Lite<syn::Index> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Index");
        formatter.field("index", Lite(&self.value.index));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Item> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Item::Const(_val) => {
                let mut formatter = formatter.debug_struct("Item::Const");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                formatter.field("ty", Lite(&_val.typ));
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Item::Enum(_val) => {
                let mut formatter = formatter.debug_struct("Item::Enum");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                if !_val.variants.is_empty() {
                    formatter.field("variants", Lite(&_val.variants));
                }
                formatter.finish()
            },
            syn::Item::Extern(_val) => {
                let mut formatter = formatter.debug_struct("Item::Extern");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                if let Some(val) = &_val.rename {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::As, proc_macro2::Ident));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("rename", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Item::Fn(_val) => {
                let mut formatter = formatter.debug_struct("Item::Fn");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("sig", Lite(&_val.sig));
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Item::Foreign(_val) => {
                let mut formatter = formatter.debug_struct("Item::Foreign");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.unsafe_.is_some() {
                    formatter.field("unsafe_", &Present);
                }
                formatter.field("abi", Lite(&_val.abi));
                if !_val.items.is_empty() {
                    formatter.field("items", Lite(&_val.items));
                }
                formatter.finish()
            },
            syn::Item::Impl(_val) => {
                let mut formatter = formatter.debug_struct("Item::Impl");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.default_.is_some() {
                    formatter.field("defaultness", &Present);
                }
                if _val.unsafe_.is_some() {
                    formatter.field("unsafe_", &Present);
                }
                formatter.field("gens", Lite(&_val.gens));
                if let Some(val) = &_val.trait_ {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((Option<syn::tok::Not>, syn::Path, syn::tok::For));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(
                                &(
                                    &super::Option {
                                        present: self.0 .0.is_some(),
                                    },
                                    Lite(&self.0 .1),
                                ),
                                formatter,
                            )?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("trait_", Print::ref_cast(val));
                }
                formatter.field("self_ty", Lite(&_val.typ));
                if !_val.items.is_empty() {
                    formatter.field("items", Lite(&_val.items));
                }
                formatter.finish()
            },
            syn::Item::Macro(_val) => {
                let mut formatter = formatter.debug_struct("Item::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.ident {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(proc_macro2::Ident);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("ident", Print::ref_cast(val));
                }
                formatter.field("mac", Lite(&_val.mac));
                if _val.semi.is_some() {
                    formatter.field("semi", &Present);
                }
                formatter.finish()
            },
            syn::Item::Mod(_val) => {
                let mut formatter = formatter.debug_struct("Item::Mod");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.unsafe_.is_some() {
                    formatter.field("unsafe_", &Present);
                }
                formatter.field("ident", Lite(&_val.ident));
                if let Some(val) = &_val.gist {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Brace, Vec<syn::Item>));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("content", Print::ref_cast(val));
                }
                if _val.semi.is_some() {
                    formatter.field("semi", &Present);
                }
                formatter.finish()
            },
            syn::Item::Static(_val) => {
                let mut formatter = formatter.debug_struct("Item::Static");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                match _val.mut_ {
                    syn::StaticMut::None => {},
                    _ => {
                        formatter.field("mut_", Lite(&_val.mut_));
                    },
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("ty", Lite(&_val.typ));
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Item::Struct(_val) => {
                let mut formatter = formatter.debug_struct("Item::Struct");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                formatter.field("fields", Lite(&_val.fields));
                if _val.semi.is_some() {
                    formatter.field("semi", &Present);
                }
                formatter.finish()
            },
            syn::Item::Trait(_val) => {
                let mut formatter = formatter.debug_struct("Item::Trait");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.unsafe_.is_some() {
                    formatter.field("unsafe_", &Present);
                }
                if _val.auto_.is_some() {
                    formatter.field("auto_", &Present);
                }
                if let Some(val) = &_val.restriction {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::item::Impl::Restriction);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("restriction", Print::ref_cast(val));
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                if _val.colon.is_some() {
                    formatter.field("colon", &Present);
                }
                if !_val.supertraits.is_empty() {
                    formatter.field("supertraits", Lite(&_val.supertraits));
                }
                if !_val.items.is_empty() {
                    formatter.field("items", Lite(&_val.items));
                }
                formatter.finish()
            },
            syn::Item::TraitAlias(_val) => {
                let mut formatter = formatter.debug_struct("Item::TraitAlias");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                if !_val.bounds.is_empty() {
                    formatter.field("bounds", Lite(&_val.bounds));
                }
                formatter.finish()
            },
            syn::Item::Type(_val) => {
                let mut formatter = formatter.debug_struct("Item::Type");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                formatter.field("ty", Lite(&_val.typ));
                formatter.finish()
            },
            syn::Item::Union(_val) => {
                let mut formatter = formatter.debug_struct("Item::Union");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                formatter.field("fields", Lite(&_val.fields));
                formatter.finish()
            },
            syn::Item::Use(_val) => {
                let mut formatter = formatter.debug_struct("Item::Use");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.leading_colon.is_some() {
                    formatter.field("leading_colon", &Present);
                }
                formatter.field("tree", Lite(&_val.tree));
                formatter.finish()
            },
            syn::Item::Verbatim(_val) => {
                formatter.write_str("Item::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::item::Const> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Const");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        formatter.field("ty", Lite(&self.value.typ));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Enum> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Enum");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        if !self.value.variants.is_empty() {
            formatter.field("variants", Lite(&self.value.variants));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Extern> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Extern");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.rename {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::As, proc_macro2::Ident));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("rename", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Fn> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Fn");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("sig", Lite(&self.value.sig));
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Foreign> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Foreign");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.unsafe_.is_some() {
            formatter.field("unsafe_", &Present);
        }
        formatter.field("abi", Lite(&self.value.abi));
        if !self.value.items.is_empty() {
            formatter.field("items", Lite(&self.value.items));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Impl> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Impl");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.default_.is_some() {
            formatter.field("defaultness", &Present);
        }
        if self.value.unsafe_.is_some() {
            formatter.field("unsafe_", &Present);
        }
        formatter.field("gens", Lite(&self.value.gens));
        if let Some(val) = &self.value.trait_ {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((Option<syn::tok::Not>, syn::Path, syn::tok::For));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(
                        &(
                            &super::Option {
                                present: self.0 .0.is_some(),
                            },
                            Lite(&self.0 .1),
                        ),
                        formatter,
                    )?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("trait_", Print::ref_cast(val));
        }
        formatter.field("self_ty", Lite(&self.value.typ));
        if !self.value.items.is_empty() {
            formatter.field("items", Lite(&self.value.items));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Mac> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Mac");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.ident {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(proc_macro2::Ident);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("ident", Print::ref_cast(val));
        }
        formatter.field("mac", Lite(&self.value.mac));
        if self.value.semi.is_some() {
            formatter.field("semi", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Mod> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Mod");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.unsafe_.is_some() {
            formatter.field("unsafe_", &Present);
        }
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.items {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Brace, Vec<syn::Item>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("content", Print::ref_cast(val));
        }
        if self.value.semi.is_some() {
            formatter.field("semi", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Static> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Static");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        match self.value.mut_ {
            syn::StaticMut::None => {},
            _ => {
                formatter.field("mut_", Lite(&self.value.mut_));
            },
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("ty", Lite(&self.value.typ));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Struct> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Struct");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        formatter.field("fields", Lite(&self.value.fields));
        if self.value.semi.is_some() {
            formatter.field("semi", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Trait> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Trait");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.unsafe_.is_some() {
            formatter.field("unsafe_", &Present);
        }
        if self.value.auto_.is_some() {
            formatter.field("auto_", &Present);
        }
        if let Some(val) = &self.value.restriction {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::item::Impl::Restriction);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("restriction", Print::ref_cast(val));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        if self.value.colon.is_some() {
            formatter.field("colon", &Present);
        }
        if !self.value.supers.is_empty() {
            formatter.field("supertraits", Lite(&self.value.supers));
        }
        if !self.value.items.is_empty() {
            formatter.field("items", Lite(&self.value.items));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::TraitAlias> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::TraitAlias");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Type> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Type");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        formatter.field("ty", Lite(&self.value.typ));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Union> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Union");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        formatter.field("fields", Lite(&self.value.fields));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Use> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Use");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.colon.is_some() {
            formatter.field("leading_colon", &Present);
        }
        formatter.field("tree", Lite(&self.value.tree));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Label> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Label");
        formatter.field("name", Lite(&self.value.name));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Life> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Life");
        formatter.field("ident", Lite(&self.value.ident));
        formatter.finish()
    }
}
impl Debug for Lite<syn::gen::param::Life> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::param::Life");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("life", Lite(&self.value.life));
        if self.value.colon.is_some() {
            formatter.field("colon", &Present);
        }
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Lit> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Lit::Str(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::ByteStr(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::Byte(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::Char(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::Int(_val) => write!(formatter, "{}", _val),
            syn::Lit::Float(_val) => write!(formatter, "{}", _val),
            syn::Lit::Bool(_val) => {
                let mut formatter = formatter.debug_struct("Lit::Bool");
                formatter.field("value", Lite(&_val.val));
                formatter.finish()
            },
            syn::Lit::Verbatim(_val) => {
                formatter.write_str("Lit::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::lit::Bool> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("lit::Bool");
        formatter.field("value", Lite(&self.value.val));
        formatter.finish()
    }
}
impl Debug for Lite<syn::lit::Byte> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::lit::ByteStr> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::lit::Char> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::lit::Float> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}", &self.value)
    }
}
impl Debug for Lite<syn::lit::Int> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}", &self.value)
    }
}
impl Debug for Lite<syn::lit::Str> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::stmt::Local> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("stmt::Local");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("pat", Lite(&self.value.pat));
        if let Some(val) = &self.value.init {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::stmt::Init);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("init", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::stmt::Init> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("stmt::Init");
        formatter.field("expr", Lite(&self.value.expr));
        if let Some(val) = &self.value.diverge {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Else, Box<syn::Expr>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("diverge", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Macro> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Macro");
        formatter.field("path", Lite(&self.value.path));
        formatter.field("delimiter", Lite(&self.value.delim));
        formatter.field("tokens", Lite(&self.value.toks));
        formatter.finish()
    }
}
impl Debug for Lite<syn::tok::Delim> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::tok::Delim::Paren(_val) => {
                formatter.write_str("MacroDelimiter::Paren")?;
                Ok(())
            },
            syn::tok::Delim::Brace(_val) => {
                formatter.write_str("MacroDelimiter::Brace")?;
                Ok(())
            },
            syn::tok::Delim::Bracket(_val) => {
                formatter.write_str("MacroDelimiter::Bracket")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::Member> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Member::Named(_val) => {
                formatter.write_str("Member::Named")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Member::Unnamed(_val) => {
                formatter.write_str("Member::Unnamed")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::meta::Meta> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::meta::Meta::Path(_val) => {
                let mut formatter = formatter.debug_struct("meta::Meta::Path");
                if _val.colon.is_some() {
                    formatter.field("leading_colon", &Present);
                }
                if !_val.segs.is_empty() {
                    formatter.field("segments", Lite(&_val.segs));
                }
                formatter.finish()
            },
            syn::meta::Meta::List(_val) => {
                let mut formatter = formatter.debug_struct("meta::Meta::List");
                formatter.field("path", Lite(&_val.path));
                formatter.field("delimiter", Lite(&_val.delim));
                formatter.field("tokens", Lite(&_val.toks));
                formatter.finish()
            },
            syn::meta::Meta::NameValue(_val) => {
                let mut formatter = formatter.debug_struct("meta::Meta::NameValue");
                formatter.field("path", Lite(&_val.path));
                formatter.field("value", Lite(&_val.expr));
                formatter.finish()
            },
        }
    }
}
impl Debug for Lite<syn::meta::List> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("meta::List");
        formatter.field("path", Lite(&self.value.path));
        formatter.field("delimiter", Lite(&self.value.delim));
        formatter.field("tokens", Lite(&self.value.toks));
        formatter.finish()
    }
}
impl Debug for Lite<syn::meta::NameValue> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("meta::NameValue");
        formatter.field("path", Lite(&self.value.path));
        formatter.field("value", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ParenthesizedArgs> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("path::ParenthesizedArgs");
        if !self.value.ins.is_empty() {
            formatter.field("inputs", Lite(&self.value.ins));
        }
        formatter.field("output", Lite(&self.value.out));
        formatter.finish()
    }
}
impl Debug for Lite<syn::patt::Patt> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::patt::Patt::Const(_val) => {
                formatter.write_str("patt::Patt::Const")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::patt::Patt::Ident(_val) => {
                let mut formatter = formatter.debug_struct("patt::Patt::Ident");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.ref_.is_some() {
                    formatter.field("by_ref", &Present);
                }
                if _val.mut_.is_some() {
                    formatter.field("mut_", &Present);
                }
                formatter.field("ident", Lite(&_val.ident));
                if let Some(val) = &_val.sub {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::At, Box<syn::patt::Patt>));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("subpat", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::patt::Patt::Lit(_val) => {
                formatter.write_str("patt::Patt::Lit")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::patt::Patt::Mac(_val) => {
                formatter.write_str("patt::Patt::Macro")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::patt::Patt::Or(_val) => {
                let mut formatter = formatter.debug_struct("patt::Patt::Or");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.vert.is_some() {
                    formatter.field("leading_vert", &Present);
                }
                if !_val.cases.is_empty() {
                    formatter.field("cases", Lite(&_val.cases));
                }
                formatter.finish()
            },
            syn::patt::Patt::Paren(_val) => {
                let mut formatter = formatter.debug_struct("patt::Patt::Paren");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("pat", Lite(&_val.patt));
                formatter.finish()
            },
            syn::patt::Patt::Path(_val) => {
                formatter.write_str("patt::Patt::Path")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::patt::Patt::Range(_val) => {
                formatter.write_str("patt::Patt::Range")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::patt::Patt::Ref(_val) => {
                let mut formatter = formatter.debug_struct("patt::Patt::Reference");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.mut_.is_some() {
                    formatter.field("mut_", &Present);
                }
                formatter.field("pat", Lite(&_val.patt));
                formatter.finish()
            },
            syn::patt::Patt::Rest(_val) => {
                let mut formatter = formatter.debug_struct("patt::Patt::Rest");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.finish()
            },
            syn::patt::Patt::Slice(_val) => {
                let mut formatter = formatter.debug_struct("patt::Patt::Slice");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::patt::Patt::Struct(_val) => {
                let mut formatter = formatter.debug_struct("patt::Patt::Struct");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("qself", Print::ref_cast(val));
                }
                formatter.field("path", Lite(&_val.path));
                if !_val.fields.is_empty() {
                    formatter.field("fields", Lite(&_val.fields));
                }
                if let Some(val) = &_val.rest {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::patt::Rest);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("rest", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::patt::Patt::Tuple(_val) => {
                let mut formatter = formatter.debug_struct("patt::Patt::Tuple");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::patt::Patt::TupleStruct(_val) => {
                let mut formatter = formatter.debug_struct("patt::Patt::TupleStruct");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("qself", Print::ref_cast(val));
                }
                formatter.field("path", Lite(&_val.path));
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::patt::Patt::Type(_val) => {
                let mut formatter = formatter.debug_struct("patt::Patt::Type");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("pat", Lite(&_val.patt));
                formatter.field("ty", Lite(&_val.typ));
                formatter.finish()
            },
            syn::patt::Patt::Verbatim(_val) => {
                formatter.write_str("patt::Patt::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            syn::patt::Patt::Wild(_val) => {
                let mut formatter = formatter.debug_struct("patt::Patt::Wild");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.finish()
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::patt::Ident> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::Ident");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.ref_.is_some() {
            formatter.field("by_ref", &Present);
        }
        if self.value.mut_.is_some() {
            formatter.field("mut_", &Present);
        }
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.sub {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::At, Box<syn::patt::Patt>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("subpat", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::patt::Or> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::Or");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.vert.is_some() {
            formatter.field("leading_vert", &Present);
        }
        if !self.value.cases.is_empty() {
            formatter.field("cases", Lite(&self.value.cases));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::patt::Paren> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::Paren");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("pat", Lite(&self.value.patt));
        formatter.finish()
    }
}
impl Debug for Lite<syn::patt::Ref> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::Ref");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.mut_.is_some() {
            formatter.field("mut_", &Present);
        }
        formatter.field("pat", Lite(&self.value.patt));
        formatter.finish()
    }
}
impl Debug for Lite<syn::patt::Rest> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::Rest");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::patt::Slice> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::Slice");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::patt::Struct> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::Struct");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("qself", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        if !self.value.fields.is_empty() {
            formatter.field("fields", Lite(&self.value.fields));
        }
        if let Some(val) = &self.value.rest {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::patt::Rest);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("rest", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::patt::Tuple> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::Tuple");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::patt::TupleStruct> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::TupleStruct");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("qself", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::patt::Type> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::Type");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("pat", Lite(&self.value.patt));
        formatter.field("ty", Lite(&self.value.typ));
        formatter.finish()
    }
}
impl Debug for Lite<syn::patt::Wild> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("patt::Wild");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Path> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Path");
        if self.value.colon.is_some() {
            formatter.field("leading_colon", &Present);
        }
        if !self.value.segs.is_empty() {
            formatter.field("segments", Lite(&self.value.segs));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Args> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Args::None => formatter.write_str("path::Args::None"),
            syn::Args::Angled(_val) => {
                let mut formatter = formatter.debug_struct("path::Args::AngleBracketed");
                if _val.colon2.is_some() {
                    formatter.field("colon2", &Present);
                }
                if !_val.args.is_empty() {
                    formatter.field("args", Lite(&_val.args));
                }
                formatter.finish()
            },
            syn::Args::Parenthesized(_val) => {
                let mut formatter = formatter.debug_struct("path::Args::Parenthesized");
                if !_val.ins.is_empty() {
                    formatter.field("inputs", Lite(&_val.ins));
                }
                formatter.field("output", Lite(&_val.out));
                formatter.finish()
            },
        }
    }
}
impl Debug for Lite<syn::Segment> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("path::Segment");
        formatter.field("ident", Lite(&self.value.ident));
        match self.value.args {
            syn::Args::None => {},
            _ => {
                formatter.field("arguments", Lite(&self.value.args));
            },
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::gen::Where::Life> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PredicateLifetime");
        formatter.field("life", Lite(&self.value.life));
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::gen::Where::Type> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PredicateType");
        if let Some(val) = &self.value.lifes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Bgen::bound::Lifes);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("lifetimes", Print::ref_cast(val));
        }
        formatter.field("bounded", Lite(&self.value.bounded));
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::QSelf> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("QSelf");
        formatter.field("ty", Lite(&self.value.ty));
        formatter.field("position", Lite(&self.value.pos));
        if self.value.as_.is_some() {
            formatter.field("as_", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::RangeLimits> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::RangeLimits::HalfOpen(_val) => {
                formatter.write_str("RangeLimits::HalfOpen")?;
                Ok(())
            },
            syn::RangeLimits::Closed(_val) => {
                formatter.write_str("RangeLimits::Closed")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::item::Receiver> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Receiver");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.ref_ {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::And, Option<syn::Life>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(
                        {
                            #[derive(RefCast)]
                            #[repr(transparent)]
                            struct Print(Option<syn::Life>);
                            impl Debug for Print {
                                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                                    match &self.0 {
                                        Some(_val) => {
                                            formatter.write_str("Some(")?;
                                            Debug::fmt(Lite(_val), formatter)?;
                                            formatter.write_str(")")?;
                                            Ok(())
                                        },
                                        None => formatter.write_str("None"),
                                    }
                                }
                            }
                            Print::ref_cast(&self.0 .1)
                        },
                        formatter,
                    )?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("reference", Print::ref_cast(val));
        }
        if self.value.mut_.is_some() {
            formatter.field("mut_", &Present);
        }
        if self.value.colon.is_some() {
            formatter.field("colon", &Present);
        }
        formatter.field("ty", Lite(&self.value.typ));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Ret> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::typ::Ret::Default => formatter.write_str("typ::Ret::Default"),
            syn::typ::Ret::Type(_v0, _v1) => {
                let mut formatter = formatter.debug_tuple("typ::Ret::Type");
                formatter.field(Lite(_v1));
                formatter.finish()
            },
        }
    }
}
impl Debug for Lite<syn::item::Sig> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Sig");
        if self.value.const_.is_some() {
            formatter.field("const_", &Present);
        }
        if self.value.async_.is_some() {
            formatter.field("asyncness", &Present);
        }
        if self.value.unsafe_.is_some() {
            formatter.field("unsafe_", &Present);
        }
        if let Some(val) = &self.value.abi {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Abi);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("abi", Print::ref_cast(val));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        if !self.value.args.is_empty() {
            formatter.field("inputs", Lite(&self.value.args));
        }
        if let Some(val) = &self.value.vari {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::item::Variadic);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("vari", Print::ref_cast(val));
        }
        formatter.field("output", Lite(&self.value.ret));
        formatter.finish()
    }
}
impl Debug for Lite<syn::StaticMut> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::StaticMut::Mut(_val) => {
                formatter.write_str("StaticMutability::Mut")?;
                Ok(())
            },
            syn::StaticMut::None => formatter.write_str("StaticMutability::None"),
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::stmt::Stmt> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::stmt::Stmt::stmt::Local(_val) => {
                let mut formatter = formatter.debug_struct("stmt::Stmt::stmt::Local");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("pat", Lite(&_val.pat));
                if let Some(val) = &_val.init {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::stmt::Init);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("init", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::stmt::Stmt::Item(_val) => {
                formatter.write_str("stmt::Stmt::Item")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::stmt::Stmt::Expr(_v0, _v1) => {
                let mut formatter = formatter.debug_tuple("stmt::Stmt::Expr");
                formatter.field(Lite(_v0));
                formatter.field(&super::Option { present: _v1.is_some() });
                formatter.finish()
            },
            syn::stmt::Stmt::Mac(_val) => {
                let mut formatter = formatter.debug_struct("stmt::Stmt::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("mac", Lite(&_val.mac));
                if _val.semi.is_some() {
                    formatter.field("semi", &Present);
                }
                formatter.finish()
            },
        }
    }
}
impl Debug for Lite<syn::stmt::Mac> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("stmt::Mac");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("mac", Lite(&self.value.mac));
        if self.value.semi.is_some() {
            formatter.field("semi", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::gen::bound::Trait> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::bound::Trait");
        if self.value.paren.is_some() {
            formatter.field("paren", &Present);
        }
        match self.value.modif {
            syn::gen::bound::Modifier::None => {},
            _ => {
                formatter.field("modifier", Lite(&self.value.modif));
            },
        }
        if let Some(val) = &self.value.lifes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Bgen::bound::Lifes);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("lifetimes", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        formatter.finish()
    }
}
impl Debug for Lite<syn::gen::bound::Modifier> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::gen::bound::Modifier::None => formatter.write_str("gen::bound::Modifier::None"),
            syn::gen::bound::Modifier::Maybe(_val) => {
                formatter.write_str("gen::bound::Modifier::Maybe")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::item::Trait::Item> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::item::Trait::Item::Const(_val) => {
                let mut formatter = formatter.debug_struct("item::Trait::Item::Const");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                formatter.field("ty", Lite(&_val.typ));
                if let Some(val) = &_val.default {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Eq, syn::Expr));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("default", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::item::Trait::Item::Fn(_val) => {
                let mut formatter = formatter.debug_struct("item::Trait::Item::Fn");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("sig", Lite(&_val.sig));
                if let Some(val) = &_val.default {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Block);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("default", Print::ref_cast(val));
                }
                if _val.semi.is_some() {
                    formatter.field("semi", &Present);
                }
                formatter.finish()
            },
            syn::item::Trait::Item::Type(_val) => {
                let mut formatter = formatter.debug_struct("item::Trait::Item::Type");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("gens", Lite(&_val.gens));
                if _val.colon.is_some() {
                    formatter.field("colon", &Present);
                }
                if !_val.bounds.is_empty() {
                    formatter.field("bounds", Lite(&_val.bounds));
                }
                if let Some(val) = &_val.default {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Eq, syn::typ::Type));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("default", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::item::Trait::Item::Macro(_val) => {
                let mut formatter = formatter.debug_struct("item::Trait::Item::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("mac", Lite(&_val.mac));
                if _val.semi.is_some() {
                    formatter.field("semi", &Present);
                }
                formatter.finish()
            },
            syn::item::Trait::Item::Verbatim(_val) => {
                formatter.write_str("item::Trait::Item::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::item::Trait::Const> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Trait::Const");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        formatter.field("ty", Lite(&self.value.typ));
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Eq, syn::Expr));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("default", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Trait::Fn> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Trait::Fn");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("sig", Lite(&self.value.sig));
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Block);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("default", Print::ref_cast(val));
        }
        if self.value.semi.is_some() {
            formatter.field("semi", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Trait::Mac> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Trait::Mac");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("mac", Lite(&self.value.mac));
        if self.value.semi.is_some() {
            formatter.field("semi", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Trait::Type> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Trait::Type");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("gens", Lite(&self.value.gens));
        if self.value.colon.is_some() {
            formatter.field("colon", &Present);
        }
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Eq, syn::typ::Type));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("default", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Type> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::typ::Type::Array(_val) => {
                let mut formatter = formatter.debug_struct("Type::Array");
                formatter.field("elem", Lite(&_val.elem));
                formatter.field("len", Lite(&_val.len));
                formatter.finish()
            },
            syn::typ::Type::Fn(_val) => {
                let mut formatter = formatter.debug_struct("Type::Fn");
                if let Some(val) = &_val.lifes {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Bgen::bound::Lifes);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("lifetimes", Print::ref_cast(val));
                }
                if _val.unsafe_.is_some() {
                    formatter.field("unsafe_", &Present);
                }
                if let Some(val) = &_val.abi {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Abi);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("abi", Print::ref_cast(val));
                }
                if !_val.args.is_empty() {
                    formatter.field("inputs", Lite(&_val.args));
                }
                if let Some(val) = &_val.vari {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::typ::Variadic);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("vari", Print::ref_cast(val));
                }
                formatter.field("output", Lite(&_val.ret));
                formatter.finish()
            },
            syn::typ::Type::Group(_val) => {
                let mut formatter = formatter.debug_struct("Type::Group");
                formatter.field("elem", Lite(&_val.elem));
                formatter.finish()
            },
            syn::typ::Type::Impl(_val) => {
                let mut formatter = formatter.debug_struct("Type::ImplTrait");
                if !_val.bounds.is_empty() {
                    formatter.field("bounds", Lite(&_val.bounds));
                }
                formatter.finish()
            },
            syn::typ::Type::Infer(_val) => {
                let mut formatter = formatter.debug_struct("Type::Infer");
                formatter.finish()
            },
            syn::typ::Type::Mac(_val) => {
                let mut formatter = formatter.debug_struct("Type::Macro");
                formatter.field("mac", Lite(&_val.mac));
                formatter.finish()
            },
            syn::typ::Type::Never(_val) => {
                let mut formatter = formatter.debug_struct("Type::Never");
                formatter.finish()
            },
            syn::typ::Type::Paren(_val) => {
                let mut formatter = formatter.debug_struct("Type::Paren");
                formatter.field("elem", Lite(&_val.elem));
                formatter.finish()
            },
            syn::typ::Type::Path(_val) => {
                let mut formatter = formatter.debug_struct("Type::Path");
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("qself", Print::ref_cast(val));
                }
                formatter.field("path", Lite(&_val.path));
                formatter.finish()
            },
            syn::typ::Type::Ptr(_val) => {
                let mut formatter = formatter.debug_struct("Type::Ptr");
                if _val.const_.is_some() {
                    formatter.field("const_", &Present);
                }
                if _val.mut_.is_some() {
                    formatter.field("mut_", &Present);
                }
                formatter.field("elem", Lite(&_val.elem));
                formatter.finish()
            },
            syn::typ::Type::Ref(_val) => {
                let mut formatter = formatter.debug_struct("Type::Reference");
                if let Some(val) = &_val.life {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Life);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("life", Print::ref_cast(val));
                }
                if _val.mut_.is_some() {
                    formatter.field("mut_", &Present);
                }
                formatter.field("elem", Lite(&_val.elem));
                formatter.finish()
            },
            syn::typ::Type::Slice(_val) => {
                let mut formatter = formatter.debug_struct("Type::Slice");
                formatter.field("elem", Lite(&_val.elem));
                formatter.finish()
            },
            syn::typ::Type::Trait(_val) => {
                let mut formatter = formatter.debug_struct("Type::TraitObject");
                if _val.dyn_.is_some() {
                    formatter.field("dyn_", &Present);
                }
                if !_val.bounds.is_empty() {
                    formatter.field("bounds", Lite(&_val.bounds));
                }
                formatter.finish()
            },
            syn::typ::Type::Tuple(_val) => {
                let mut formatter = formatter.debug_struct("Type::Tuple");
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::typ::Type::Verbatim(_val) => {
                formatter.write_str("Type::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::typ::Array> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Array");
        formatter.field("elem", Lite(&self.value.elem));
        formatter.field("len", Lite(&self.value.len));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Fn> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Fn");
        if let Some(val) = &self.value.lifes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Bgen::bound::Lifes);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("lifetimes", Print::ref_cast(val));
        }
        if self.value.unsafe_.is_some() {
            formatter.field("unsafe_", &Present);
        }
        if let Some(val) = &self.value.abi {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Abi);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("abi", Print::ref_cast(val));
        }
        if !self.value.args.is_empty() {
            formatter.field("inputs", Lite(&self.value.args));
        }
        if let Some(val) = &self.value.vari {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::typ::Variadic);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("vari", Print::ref_cast(val));
        }
        formatter.field("output", Lite(&self.value.ret));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Group> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Group");
        formatter.field("elem", Lite(&self.value.elem));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Impl> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Impl");
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Infer> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Infer");
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Mac> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Mac");
        formatter.field("mac", Lite(&self.value.mac));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Never> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Never");
        formatter.finish()
    }
}
impl Debug for Lite<syn::gen::param::Type> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::param::Type");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("ident", Lite(&self.value.ident));
        if self.value.colon.is_some() {
            formatter.field("colon", &Present);
        }
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        if self.value.eq.is_some() {
            formatter.field("eq", &Present);
        }
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::typ::Type);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("default", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::gen::bound::Type> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::gen::bound::Type::Trait(_val) => {
                formatter.write_str("gen::bound::Type::Trait")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::gen::bound::Type::Life(_val) => {
                let mut formatter = formatter.debug_struct("gen::bound::Type::Life");
                formatter.field("ident", Lite(&_val.ident));
                formatter.finish()
            },
            syn::gen::bound::Type::Stream(_val) => {
                formatter.write_str("gen::bound::Type::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::typ::Paren> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Paren");
        formatter.field("elem", Lite(&self.value.elem));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Path> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Path");
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("qself", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Ptr> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Ptr");
        if self.value.const_.is_some() {
            formatter.field("const_", &Present);
        }
        if self.value.mut_.is_some() {
            formatter.field("mut_", &Present);
        }
        formatter.field("elem", Lite(&self.value.elem));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Ref> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Ref");
        if let Some(val) = &self.value.life {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Life);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("life", Print::ref_cast(val));
        }
        if self.value.mut_.is_some() {
            formatter.field("mut_", &Present);
        }
        formatter.field("elem", Lite(&self.value.elem));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Slice> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Slice");
        formatter.field("elem", Lite(&self.value.elem));
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Trait> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Trait");
        if self.value.dyn_.is_some() {
            formatter.field("dyn_", &Present);
        }
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::typ::Tuple> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("typ::Tuple");
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::UnOp> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::UnOp::Deref(_val) => {
                formatter.write_str("UnOp::Deref")?;
                Ok(())
            },
            syn::UnOp::Not(_val) => {
                formatter.write_str("UnOp::Not")?;
                Ok(())
            },
            syn::UnOp::Neg(_val) => {
                formatter.write_str("UnOp::Neg")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::item::Use::Glob> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Use::Glob");
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Use::Group> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Use::Group");
        if !self.value.elems.is_empty() {
            formatter.field("items", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Use::Name> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Use::Name");
        formatter.field("ident", Lite(&self.value.ident));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Use::Path> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Use::Path");
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("tree", Lite(&self.value.tree));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Use::Rename> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Use::Rename");
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("rename", Lite(&self.value.rename));
        formatter.finish()
    }
}
impl Debug for Lite<syn::item::Use::Tree> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::item::Use::Tree::Path(_val) => {
                formatter.write_str("item::Use::Tree::Path")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::item::Use::Tree::Name(_val) => {
                formatter.write_str("item::Use::Tree::Name")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::item::Use::Tree::Rename(_val) => {
                formatter.write_str("item::Use::Tree::Rename")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::item::Use::Tree::Glob(_val) => {
                formatter.write_str("item::Use::Tree::Glob")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::item::Use::Tree::Group(_val) => {
                formatter.write_str("item::Use::Tree::Group")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::item::Variadic> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("item::Variadic");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.pat {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((Box<syn::patt::Patt>, syn::tok::Colon));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("pat", Print::ref_cast(val));
        }
        if self.value.comma.is_some() {
            formatter.field("comma", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Variant> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Variant");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("fields", Lite(&self.value.fields));
        if let Some(val) = &self.value.discriminant {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Eq, syn::Expr));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("discriminant", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::data::Restricted> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("data::Restricted");
        if self.value.in_.is_some() {
            formatter.field("in_", &Present);
        }
        formatter.field("path", Lite(&self.value.path));
        formatter.finish()
    }
}
impl Debug for Lite<syn::data::Visibility> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::data::Visibility::Public(_val) => {
                formatter.write_str("data::Visibility::Public")?;
                Ok(())
            },
            syn::data::Visibility::Restricted(_val) => {
                let mut formatter = formatter.debug_struct("data::Visibility::Restricted");
                if _val.in_.is_some() {
                    formatter.field("in_", &Present);
                }
                formatter.field("path", Lite(&_val.path));
                formatter.finish()
            },
            syn::data::Visibility::Inherited => formatter.write_str("data::Visibility::Inherited"),
        }
    }
}
impl Debug for Lite<syn::gen::Where> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("gen::Where");
        if !self.value.preds.is_empty() {
            formatter.field("predicates", Lite(&self.value.preds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::gen::Where::Pred> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::gen::Where::Pred::Life(_val) => {
                formatter.write_str("WherePredicate::Life")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::gen::Where::Pred::Type(_val) => {
                formatter.write_str("WherePredicate::Type")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
