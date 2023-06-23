use crate::ir::{
    comp::{CompKind, Data, Field, FieldMeths},
    item::{CanonName, HasTypeParam, IsOpaque, Item},
    typ::TypeKind,
    Context,
};

pub fn gen_debug_impl(ctx: &Context, fields: &[Field], it: &Item, kind: CompKind) -> proc_macro2::TokenStream {
    let struct_name = it.canon_name(ctx);
    let mut format_string = format!("{} {{{{ ", struct_name);
    let mut tokens = vec![];
    if it.is_opaque(ctx, &()) {
        format_string.push_str("opaque");
    } else {
        match kind {
            CompKind::Union => {
                format_string.push_str("union");
            },
            CompKind::Struct => {
                let processed_fields = fields.iter().filter_map(|f| match f {
                    Field::Data(ref fd) => fd.impl_debug(ctx, ()),
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
        fn fmt(&self, x: &mut ::#prefix::fmt::Formatter<'_>) -> ::#prefix ::fmt::Result {
            write!(x, #( #tokens ),*)
        }
    }
}

pub trait ImplDebug<'a> {
    type Extra;
    fn impl_debug(&self, ctx: &Context, extra: Self::Extra) -> Option<(String, Vec<proc_macro2::TokenStream>)>;
}
impl<'a> ImplDebug<'a> for Data {
    type Extra = ();
    fn impl_debug(&self, ctx: &Context, _: Self::Extra) -> Option<(String, Vec<proc_macro2::TokenStream>)> {
        if let Some(name) = self.name() {
            ctx.resolve_item(self.ty()).impl_debug(ctx, name)
        } else {
            None
        }
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
            TypeKind::Void
            | TypeKind::NullPtr
            | TypeKind::Int(..)
            | TypeKind::Float(..)
            | TypeKind::Complex(..)
            | TypeKind::Func(..)
            | TypeKind::Enum(..)
            | TypeKind::Reference(..)
            | TypeKind::UnresolvedRef(..)
            | TypeKind::Comp(..) => debug_print(name, quote! { #name_ident }),
            TypeKind::TemplInst(ref x) => {
                if x.is_opaque(ctx, self) {
                    Some((format!("{}: opaque", name), vec![]))
                } else {
                    debug_print(name, quote! { #name_ident })
                }
            },
            TypeKind::Param => Some((format!("{}: Non-debuggable generic", name), vec![])),
            TypeKind::Array(_, len) => {
                if self.has_ty_param_in_array(ctx) {
                    Some((format!("{}: Array with length {}", name, len), vec![]))
                } else {
                    debug_print(name, quote! { #name_ident })
                }
            },
            TypeKind::Vector(_, len) => {
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
            TypeKind::ResolvedRef(t) | TypeKind::TemplAlias(t, _) | TypeKind::Alias(t) | TypeKind::BlockPtr(t) => {
                ctx.resolve_item(t).impl_debug(ctx, name)
            },
            TypeKind::Pointer(x) => {
                let inner_type = ctx.resolve_type(x).canon_type(ctx);
                match *inner_type.kind() {
                    TypeKind::Func(ref sig) if !sig.fn_ptrs_can_derive() => {
                        Some((format!("{}: FnPointer", name), vec![]))
                    },
                    _ => debug_print(name, quote! { #name_ident }),
                }
            },
            TypeKind::Opaque => None,
        }
    }
}