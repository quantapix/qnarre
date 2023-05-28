use super::context::{BindgenContext, ItemId, TypeId};
use super::item::{Ancestors, IsOpaque, Item};
use super::traversal::{EdgeKind, Trace, Tracer};
use crate::clang;

pub(crate) trait TemplParams: Sized {
    fn self_template_params(&self, ctx: &BindgenContext) -> Vec<TypeId>;

    fn num_self_template_params(&self, ctx: &BindgenContext) -> usize {
        self.self_template_params(ctx).len()
    }

    fn all_template_params(&self, ctx: &BindgenContext) -> Vec<TypeId>
    where
        Self: Ancestors,
    {
        let mut ancestors: Vec<_> = self.ancestors(ctx).collect();
        ancestors.reverse();
        ancestors
            .into_iter()
            .flat_map(|id| id.self_template_params(ctx).into_iter())
            .collect()
    }

    fn used_template_params(&self, ctx: &BindgenContext) -> Vec<TypeId>
    where
        Self: AsRef<ItemId>,
    {
        assert!(
            ctx.in_codegen_phase(),
            "template parameter usage is not computed until codegen"
        );

        let id = *self.as_ref();
        ctx.resolve_item(id)
            .all_template_params(ctx)
            .into_iter()
            .filter(|p| ctx.uses_template_parameter(id, *p))
            .collect()
    }
}

pub(crate) trait AsTemplParam {
    type Extra;

    fn as_template_param(&self, ctx: &BindgenContext, extra: &Self::Extra) -> Option<TypeId>;

    fn is_template_param(&self, ctx: &BindgenContext, extra: &Self::Extra) -> bool {
        self.as_template_param(ctx, extra).is_some()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TemplInstantiation {
    definition: TypeId,
    args: Vec<TypeId>,
}

impl TemplInstantiation {
    pub(crate) fn new<I>(definition: TypeId, args: I) -> TemplInstantiation
    where
        I: IntoIterator<Item = TypeId>,
    {
        TemplInstantiation {
            definition,
            args: args.into_iter().collect(),
        }
    }

    pub(crate) fn template_definition(&self) -> TypeId {
        self.definition
    }

    pub(crate) fn template_arguments(&self) -> &[TypeId] {
        &self.args[..]
    }

    pub(crate) fn from_ty(ty: &clang::Type, ctx: &mut BindgenContext) -> Option<TemplInstantiation> {
        use clang::*;

        let template_args = ty
            .template_args()
            .map_or(vec![], |args| match ty.canonical_type().template_args() {
                Some(canonical_args) => {
                    let arg_count = args.len();
                    args.chain(canonical_args.skip(arg_count))
                        .filter(|t| t.kind() != CXType_Invalid)
                        .map(|t| Item::from_ty_or_ref(t, t.declaration(), None, ctx))
                        .collect()
                },
                None => args
                    .filter(|t| t.kind() != CXType_Invalid)
                    .map(|t| Item::from_ty_or_ref(t, t.declaration(), None, ctx))
                    .collect(),
            });

        let declaration = ty.declaration();
        let definition = if declaration.kind() == CXCursor_TypeAliasTemplateDecl {
            Some(declaration)
        } else {
            declaration.specialized().or_else(|| {
                let mut template_ref = None;
                ty.declaration().visit(|child| {
                    if child.kind() == CXCursor_TemplateRef {
                        template_ref = Some(child);
                        return CXVisit_Break;
                    }

                    CXChildVisit_Recurse
                });

                template_ref.and_then(|cur| cur.referenced())
            })
        };

        let definition = match definition {
            Some(def) => def,
            None => {
                if !ty.declaration().is_builtin() {
                    warn!(
                        "Could not find template definition for template \
                         instantiation"
                    );
                }
                return None;
            },
        };

        let template_definition = Item::from_ty_or_ref(definition.cur_type(), definition, None, ctx);

        Some(TemplInstantiation::new(template_definition, template_args))
    }
}

impl IsOpaque for TemplInstantiation {
    type Extra = Item;

    fn is_opaque(&self, ctx: &BindgenContext, item: &Item) -> bool {
        if self.template_definition().is_opaque(ctx, &()) {
            return true;
        }

        let mut path = item.path_for_allowlisting(ctx).clone();
        let args: Vec<_> = self
            .template_arguments()
            .iter()
            .map(|arg| {
                let arg_path = ctx.resolve_item(*arg).path_for_allowlisting(ctx);
                arg_path[1..].join("::")
            })
            .collect();
        {
            let last = path.last_mut().unwrap();
            last.push('<');
            last.push_str(&args.join(", "));
            last.push('>');
        }

        ctx.opaque_by_name(&path)
    }
}

impl Trace for TemplInstantiation {
    type Extra = ();

    fn trace<T>(&self, _ctx: &BindgenContext, tracer: &mut T, _: &())
    where
        T: Tracer,
    {
        tracer.visit_kind(self.definition.into(), EdgeKind::TemplateDeclaration);
        for arg in self.template_arguments() {
            tracer.visit_kind(arg.into(), EdgeKind::TemplateArgument);
        }
    }
}
