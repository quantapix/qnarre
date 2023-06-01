use super::GenError;
use crate::callbacks::IntKind;
use crate::ir::comp::CompKind;
use crate::ir::func::{FnKind, Func};
use crate::ir::item::CanonName;
use crate::ir::item::Item;
use crate::ir::item_kind::ItemKind;
use crate::ir::typ::{FloatKind, Type, TypeKind};
use crate::ir::{Context, TypeId};
use std::io::Write;
fn get_loc(it: &Item) -> String {
    it.loc().map(|x| x.to_string()).unwrap_or_else(|| "unknown".to_owned())
}
pub trait CSerialize<'a> {
    type Extra;
    fn serialize<W: Write>(
        &self,
        ctx: &Context,
        extra: Self::Extra,
        stack: &mut Vec<String>,
        writer: &mut W,
    ) -> Result<(), GenError>;
}
impl<'a> CSerialize<'a> for Item {
    type Extra = ();
    fn serialize<W: Write>(
        &self,
        ctx: &Context,
        (): Self::Extra,
        stack: &mut Vec<String>,
        writer: &mut W,
    ) -> Result<(), GenError> {
        match self.kind() {
            ItemKind::Func(x) => x.serialize(ctx, self, stack, writer),
            kind => Err(GenError::Serialize {
                msg: format!("Cannot serialize item kind {:?}", kind),
                loc: get_loc(self),
            }),
        }
    }
}
impl<'a> CSerialize<'a> for Func {
    type Extra = &'a Item;
    fn serialize<W: Write>(
        &self,
        ctx: &Context,
        item: Self::Extra,
        stack: &mut Vec<String>,
        writer: &mut W,
    ) -> Result<(), GenError> {
        if self.kind() != FnKind::Func {
            return Err(GenError::Serialize {
                msg: format!("Cannot serialize function kind {:?}", self.kind(),),
                loc: get_loc(item),
            });
        }
        let signature = match ctx.resolve_type(self.sig()).kind() {
            TypeKind::Func(signature) => signature,
            _ => unreachable!(),
        };
        assert!(!signature.is_variadic());
        let name = self.name();
        let args = {
            let mut count = 0;
            signature
                .arg_types()
                .iter()
                .cloned()
                .map(|(opt_name, type_id)| {
                    (
                        opt_name.unwrap_or_else(|| {
                            let name = format!("arg_{}", count);
                            count += 1;
                            name
                        }),
                        type_id,
                    )
                })
                .collect::<Vec<_>>()
        };
        let wrap_name = format!("{}{}", name, ctx.wrap_static_fns_suffix());
        let ret_ty = {
            let type_id = signature.ret_type();
            let item = ctx.resolve_item(type_id);
            let ret_ty = item.expect_type();
            ret_ty.serialize(ctx, item, stack, writer)?;
            ret_ty
        };
        write!(writer, " {}(", wrap_name)?;
        serialize_args(&args, ctx, writer)?;
        if ret_ty.is_void() {
            write!(writer, ") {{ {}(", name)?;
        } else {
            write!(writer, ") {{ return {}(", name)?;
        }
        serialize_sep(", ", args.iter(), ctx, writer, |(name, _), _, buf| {
            write!(buf, "{}", name).map_err(From::from)
        })?;
        writeln!(writer, "); }}")?;
        Ok(())
    }
}
impl<'a> CSerialize<'a> for TypeId {
    type Extra = ();
    fn serialize<W: Write>(
        &self,
        ctx: &Context,
        (): Self::Extra,
        stack: &mut Vec<String>,
        writer: &mut W,
    ) -> Result<(), GenError> {
        let item = ctx.resolve_item(*self);
        item.expect_type().serialize(ctx, item, stack, writer)
    }
}
impl<'a> CSerialize<'a> for Type {
    type Extra = &'a Item;
    fn serialize<W: Write>(
        &self,
        ctx: &Context,
        item: Self::Extra,
        stack: &mut Vec<String>,
        writer: &mut W,
    ) -> Result<(), GenError> {
        match self.kind() {
            TypeKind::Void => {
                if self.is_const() {
                    write!(writer, "const ")?;
                }
                write!(writer, "void")?
            },
            TypeKind::NullPtr => {
                if self.is_const() {
                    write!(writer, "const ")?;
                }
                write!(writer, "nullptr_t")?
            },
            TypeKind::Int(int_kind) => {
                if self.is_const() {
                    write!(writer, "const ")?;
                }
                match int_kind {
                    IntKind::Bool => write!(writer, "bool")?,
                    IntKind::SChar => write!(writer, "signed char")?,
                    IntKind::UChar => write!(writer, "unsigned char")?,
                    IntKind::WChar => write!(writer, "wchar_t")?,
                    IntKind::Short => write!(writer, "short")?,
                    IntKind::UShort => write!(writer, "unsigned short")?,
                    IntKind::Int => write!(writer, "int")?,
                    IntKind::UInt => write!(writer, "unsigned int")?,
                    IntKind::Long => write!(writer, "long")?,
                    IntKind::ULong => write!(writer, "unsigned long")?,
                    IntKind::LongLong => write!(writer, "long long")?,
                    IntKind::ULongLong => write!(writer, "unsigned long long")?,
                    IntKind::Char { .. } => write!(writer, "char")?,
                    int_kind => {
                        return Err(GenError::Serialize {
                            msg: format!("Cannot serialize integer kind {:?}", int_kind),
                            loc: get_loc(item),
                        })
                    },
                }
            },
            TypeKind::Float(float_kind) => {
                if self.is_const() {
                    write!(writer, "const ")?;
                }
                match float_kind {
                    FloatKind::Float => write!(writer, "float")?,
                    FloatKind::Double => write!(writer, "double")?,
                    FloatKind::LongDouble => write!(writer, "long double")?,
                    FloatKind::Float128 => write!(writer, "__float128")?,
                }
            },
            TypeKind::Complex(float_kind) => {
                if self.is_const() {
                    write!(writer, "const ")?;
                }
                match float_kind {
                    FloatKind::Float => write!(writer, "float complex")?,
                    FloatKind::Double => write!(writer, "double complex")?,
                    FloatKind::LongDouble => write!(writer, "long double complex")?,
                    FloatKind::Float128 => write!(writer, "__complex128")?,
                }
            },
            TypeKind::Alias(type_id) => {
                if let Some(name) = self.name() {
                    if self.is_const() {
                        write!(writer, "const {}", name)?;
                    } else {
                        write!(writer, "{}", name)?;
                    }
                } else {
                    type_id.serialize(ctx, (), stack, writer)?;
                }
            },
            TypeKind::Array(type_id, length) => {
                type_id.serialize(ctx, (), stack, writer)?;
                write!(writer, " [{}]", length)?
            },
            TypeKind::Func(signature) => {
                if self.is_const() {
                    stack.push("const ".to_string());
                }
                signature.ret_type().serialize(ctx, (), &mut vec![], writer)?;
                write!(writer, " (")?;
                while let Some(item) = stack.pop() {
                    write!(writer, "{}", item)?;
                }
                write!(writer, ")")?;
                write!(writer, " (")?;
                serialize_sep(
                    ", ",
                    signature.arg_types().iter(),
                    ctx,
                    writer,
                    |(name, type_id), ctx, buf| {
                        let mut stack = vec![];
                        if let Some(name) = name {
                            stack.push(name.clone());
                        }
                        type_id.serialize(ctx, (), &mut stack, buf)
                    },
                )?;
                write!(writer, ")")?
            },
            TypeKind::ResolvedTypeRef(type_id) => {
                if self.is_const() {
                    write!(writer, "const ")?;
                }
                type_id.serialize(ctx, (), stack, writer)?
            },
            TypeKind::Pointer(type_id) => {
                if self.is_const() {
                    stack.push("*const ".to_owned());
                } else {
                    stack.push("*".to_owned());
                }
                type_id.serialize(ctx, (), stack, writer)?
            },
            TypeKind::Comp(comp_info) => {
                if self.is_const() {
                    write!(writer, "const ")?;
                }
                let name = item.canon_name(ctx);
                match comp_info.kind() {
                    CompKind::Struct => write!(writer, "struct {}", name)?,
                    CompKind::Union => write!(writer, "union {}", name)?,
                };
            },
            TypeKind::Enum(_enum_ty) => {
                if self.is_const() {
                    write!(writer, "const ")?;
                }
                let name = item.canon_name(ctx);
                write!(writer, "enum {}", name)?;
            },
            ty => {
                return Err(GenError::Serialize {
                    msg: format!("Cannot serialize type kind {:?}", ty),
                    loc: get_loc(item),
                })
            },
        };
        if !stack.is_empty() {
            write!(writer, " ")?;
            while let Some(item) = stack.pop() {
                write!(writer, "{}", item)?;
            }
        }
        Ok(())
    }
}
fn serialize_args<W: Write>(args: &[(String, TypeId)], ctx: &Context, writer: &mut W) -> Result<(), GenError> {
    if args.is_empty() {
        write!(writer, "void")?;
    } else {
        serialize_sep(", ", args.iter(), ctx, writer, |(name, type_id), ctx, buf| {
            type_id.serialize(ctx, (), &mut vec![name.clone()], buf)
        })?;
    }
    Ok(())
}
fn serialize_sep<W: Write, F: FnMut(I::Item, &Context, &mut W) -> Result<(), GenError>, I: Iterator>(
    sep: &str,
    mut iter: I,
    ctx: &Context,
    buf: &mut W,
    mut f: F,
) -> Result<(), GenError> {
    if let Some(item) = iter.next() {
        f(item, ctx, buf)?;
        let sep = sep.as_bytes();
        for item in iter {
            buf.write_all(sep)?;
            f(item, ctx, buf)?;
        }
    }
    Ok(())
}
