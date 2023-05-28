use super::super::codegen::EnumVariation;
use super::context::{BindgenContext, TypeId};
use super::item::Item;
use super::ty::{Type, TypeKind};
use crate::clang;
use crate::ir::annotations::Annotations;
use crate::parse::ParseError;
use crate::regex_set::RegexSet;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EnumVariantCustomBehavior {
    ModuleConstify,
    Constify,
    Hide,
}

#[derive(Debug)]
pub(crate) struct Enum {
    repr: Option<TypeId>,
    variants: Vec<EnumVariant>,
}

impl Enum {
    pub(crate) fn new(repr: Option<TypeId>, variants: Vec<EnumVariant>) -> Self {
        Enum { repr, variants }
    }

    pub(crate) fn repr(&self) -> Option<TypeId> {
        self.repr
    }

    pub(crate) fn variants(&self) -> &[EnumVariant] {
        &self.variants
    }

    pub(crate) fn from_ty(ty: &clang::Type, ctx: &mut BindgenContext) -> Result<Self, ParseError> {
        use clang_lib::*;
        debug!("Enum::from_ty {:?}", ty);

        if ty.kind() != CXType_Enum {
            return Err(ParseError::Continue);
        }

        let declaration = ty.declaration().canonical();
        let repr = declaration
            .enum_type()
            .and_then(|et| Item::from_ty(&et, declaration, None, ctx).ok());
        let mut variants = vec![];

        let variant_ty = repr.and_then(|r| ctx.resolve_type(r).safe_canonical_type(ctx));
        let is_bool = variant_ty.map_or(false, Type::is_bool);

        let is_signed = variant_ty.map_or(true, |ty| match *ty.kind() {
            TypeKind::Int(ref int_kind) => int_kind.is_signed(),
            ref other => {
                panic!("Since when enums can be non-integers? {:?}", other)
            },
        });

        let type_name = ty.spelling();
        let type_name = if type_name.is_empty() { None } else { Some(type_name) };
        let type_name = type_name.as_deref();

        let definition = declaration.definition().unwrap_or(declaration);
        definition.visit(|cursor| {
            if cursor.kind() == CXCursor_EnumConstantDecl {
                let value = if is_bool {
                    cursor.enum_val_boolean().map(EnumVariantValue::Boolean)
                } else if is_signed {
                    cursor.enum_val_signed().map(EnumVariantValue::Signed)
                } else {
                    cursor.enum_val_unsigned().map(EnumVariantValue::Unsigned)
                };
                if let Some(val) = value {
                    let name = cursor.spelling();
                    let annotations = Annotations::new(&cursor);
                    let custom_behavior = ctx
                        .opts()
                        .last_callback(|callbacks| callbacks.enum_variant_behavior(type_name, &name, val))
                        .or_else(|| {
                            let annotations = annotations.as_ref()?;
                            if annotations.hide() {
                                Some(EnumVariantCustomBehavior::Hide)
                            } else if annotations.constify_enum_variant() {
                                Some(EnumVariantCustomBehavior::Constify)
                            } else {
                                None
                            }
                        });

                    let new_name = ctx
                        .opts()
                        .last_callback(|callbacks| callbacks.enum_variant_name(type_name, &name, val))
                        .or_else(|| annotations.as_ref()?.use_instead_of()?.last().cloned())
                        .unwrap_or_else(|| name.clone());

                    let comment = cursor.raw_comment();
                    variants.push(EnumVariant::new(new_name, name, comment, val, custom_behavior));
                }
            }
            CXChildVisit_Continue
        });
        Ok(Enum::new(repr, variants))
    }

    fn is_matching_enum(&self, ctx: &BindgenContext, enums: &RegexSet, item: &Item) -> bool {
        let path = item.path_for_allowlisting(ctx);
        let enum_ty = item.expect_type();

        if enums.matches(path[1..].join("::")) {
            return true;
        }

        if enum_ty.name().is_some() {
            return false;
        }

        self.variants().iter().any(|v| enums.matches(v.name()))
    }

    pub(crate) fn computed_enum_variation(&self, ctx: &BindgenContext, item: &Item) -> EnumVariation {
        if self.is_matching_enum(ctx, &ctx.opts().constified_enum_modules, item) {
            EnumVariation::ModuleConsts
        } else if self.is_matching_enum(ctx, &ctx.opts().bitfield_enums, item) {
            EnumVariation::NewType {
                is_bitfield: true,
                is_global: false,
            }
        } else if self.is_matching_enum(ctx, &ctx.opts().newtype_enums, item) {
            EnumVariation::NewType {
                is_bitfield: false,
                is_global: false,
            }
        } else if self.is_matching_enum(ctx, &ctx.opts().newtype_global_enums, item) {
            EnumVariation::NewType {
                is_bitfield: false,
                is_global: true,
            }
        } else if self.is_matching_enum(ctx, &ctx.opts().rustified_enums, item) {
            EnumVariation::Rust { non_exhaustive: false }
        } else if self.is_matching_enum(ctx, &ctx.opts().rustified_non_exhaustive_enums, item) {
            EnumVariation::Rust { non_exhaustive: true }
        } else if self.is_matching_enum(ctx, &ctx.opts().constified_enums, item) {
            EnumVariation::Consts
        } else {
            ctx.opts().default_enum_style
        }
    }
}

#[derive(Debug)]
pub(crate) struct EnumVariant {
    name: String,
    name_for_allowlisting: String,
    comment: Option<String>,
    val: EnumVariantValue,
    custom_behavior: Option<EnumVariantCustomBehavior>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EnumVariantValue {
    Boolean(bool),

    Signed(i64),

    Unsigned(u64),
}

impl EnumVariant {
    pub(crate) fn new(
        name: String,
        name_for_allowlisting: String,
        comment: Option<String>,
        val: EnumVariantValue,
        custom_behavior: Option<EnumVariantCustomBehavior>,
    ) -> Self {
        EnumVariant {
            name,
            name_for_allowlisting,
            comment,
            val,
            custom_behavior,
        }
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn name_for_allowlisting(&self) -> &str {
        &self.name_for_allowlisting
    }

    pub(crate) fn val(&self) -> EnumVariantValue {
        self.val
    }

    pub(crate) fn comment(&self) -> Option<&str> {
        self.comment.as_deref()
    }

    pub(crate) fn force_constification(&self) -> bool {
        self.custom_behavior
            .map_or(false, |b| b == EnumVariantCustomBehavior::Constify)
    }

    pub(crate) fn hidden(&self) -> bool {
        self.custom_behavior
            .map_or(false, |b| b == EnumVariantCustomBehavior::Hide)
    }
}
