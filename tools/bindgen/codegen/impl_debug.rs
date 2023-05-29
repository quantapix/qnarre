use crate::ir::comp::{BitfieldUnit, CompKind, Field, FieldData, FieldMethods};
use crate::ir::item::{CanonicalName, HasTypeParamInArray, IsOpaque, Item};
use crate::ir::ty::TyKind;
use crate::ir::Context;

pub(crate) fn gen_debug_impl(ctx: &Context, fields: &[Field], item: &Item, kind: CompKind) -> proc_macro2::TokenStream {
    let struct_name = item.canonical_name(ctx);
    let mut format_string = format!("{} {{{{ ", struct_name);
    let mut tokens = vec![];

    if item.is_opaque(ctx, &()) {
        format_string.push_str("opaque");
    } else {
        match kind {
            CompKind::Union => {
                format_string.push_str("union");
            },
            CompKind::Struct => {
                let processed_fields = fields.iter().filter_map(|f| match f {
                    Field::DataMember(ref fd) => fd.impl_debug(ctx, ()),
                    Field::Bitfields(ref bu) => bu.impl_debug(ctx, ()),
                });

                for (i, (fstring, toks)) in processed_fields.enumerate() {
                    if i > 0 {
                        format_string.push_str(", ");
                    }
                    tokens.extend(toks);
                    format_string.push_str(&fstring);
                }
            },
        }
    }

    format_string.push_str(" }}");
    tokens.insert(0, quote! { #format_string });

    let prefix = ctx.trait_prefix();

    quote! {
        fn fmt(&self, f: &mut ::#prefix::fmt::Formatter<'_>) -> ::#prefix ::fmt::Result {
            write!(f, #( #tokens ),*)
        }
    }
}

pub(crate) trait ImplDebug<'a> {
    type Extra;

    fn impl_debug(&self, ctx: &Context, extra: Self::Extra) -> Option<(String, Vec<proc_macro2::TokenStream>)>;
}

impl<'a> ImplDebug<'a> for FieldData {
    type Extra = ();

    fn impl_debug(&self, ctx: &Context, _: Self::Extra) -> Option<(String, Vec<proc_macro2::TokenStream>)> {
        if let Some(name) = self.name() {
            ctx.resolve_item(self.ty()).impl_debug(ctx, name)
        } else {
            None
        }
    }
}

impl<'a> ImplDebug<'a> for BitfieldUnit {
    type Extra = ();

    fn impl_debug(&self, ctx: &Context, _: Self::Extra) -> Option<(String, Vec<proc_macro2::TokenStream>)> {
        let mut format_string = String::new();
        let mut tokens = vec![];
        for (i, bitfield) in self.bitfields().iter().enumerate() {
            if i > 0 {
                format_string.push_str(", ");
            }

            if let Some(bitfield_name) = bitfield.name() {
                format_string.push_str(&format!("{} : {{:?}}", bitfield_name));
                let getter_name = bitfield.getter();
                let name_ident = ctx.rust_ident_raw(getter_name);
                tokens.push(quote! {
                    self.#name_ident ()
                });
            }
        }

        Some((format_string, tokens))
    }
}

impl<'a> ImplDebug<'a> for Item {
    type Extra = &'a str;

    fn impl_debug(&self, ctx: &Context, name: &str) -> Option<(String, Vec<proc_macro2::TokenStream>)> {
        let name_ident = ctx.rust_ident(name);

        if !ctx.allowed_items().contains(&self.id()) {
            return None;
        }

        let ty = match self.as_type() {
            Some(ty) => ty,
            None => {
                return None;
            },
        };

        fn debug_print(
            name: &str,
            name_ident: proc_macro2::TokenStream,
        ) -> Option<(String, Vec<proc_macro2::TokenStream>)> {
            Some((
                format!("{}: {{:?}}", name),
                vec![quote! {
                    self.#name_ident
                }],
            ))
        }
        match *ty.kind() {
            TyKind::Void
            | TyKind::NullPtr
            | TyKind::Int(..)
            | TyKind::Float(..)
            | TyKind::Complex(..)
            | TyKind::Function(..)
            | TyKind::Enum(..)
            | TyKind::Reference(..)
            | TyKind::UnresolvedTypeRef(..)
            | TyKind::Comp(..) => debug_print(name, quote! { #name_ident }),

            TyKind::TemplateInstantiation(ref x) => {
                if x.is_opaque(ctx, self) {
                    Some((format!("{}: opaque", name), vec![]))
                } else {
                    debug_print(name, quote! { #name_ident })
                }
            },

            TyKind::TypeParam => Some((format!("{}: Non-debuggable generic", name), vec![])),

            TyKind::Array(_, len) => {
                if self.has_type_param_in_array(ctx) {
                    Some((format!("{}: Array with length {}", name, len), vec![]))
                } else {
                    debug_print(name, quote! { #name_ident })
                }
            },
            TyKind::Vector(_, len) => {
                if ctx.opts().use_core {
                    Some((format!("{}(...)", name), vec![]))
                } else {
                    let self_ids = 0..len;
                    Some((
                        format!("{}({{}})", name),
                        vec![quote! {
                            #(format!("{:?}", self.#self_ids)),*
                        }],
                    ))
                }
            },

            TyKind::ResolvedTypeRef(t) | TyKind::TemplateAlias(t, _) | TyKind::Alias(t) | TyKind::BlockPointer(t) => {
                ctx.resolve_item(t).impl_debug(ctx, name)
            },

            TyKind::Pointer(inner) => {
                let inner_type = ctx.resolve_type(inner).canonical_type(ctx);
                match *inner_type.kind() {
                    TyKind::Function(ref sig) if !sig.function_pointers_can_derive() => {
                        Some((format!("{}: FunctionPointer", name), vec![]))
                    },
                    _ => debug_print(name, quote! { #name_ident }),
                }
            },

            TyKind::Opaque => None,
        }
    }
}
