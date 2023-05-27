use super::{Monotone, YConstrain};
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::item::{Item, ItemSet};
use crate::ir::template::{TemplateInstantiation, TemplateParameters};
use crate::ir::traversal::{EdgeKind, Trace};
use crate::ir::ty::TypeKind;
use crate::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct UsedTemplateParams<'ctx> {
    ctx: &'ctx BindgenContext,
    used: HashMap<ItemId, Option<ItemSet>>,
    deps: HashMap<ItemId, Vec<ItemId>>,
    allowed_items: HashSet<ItemId>,
}

impl<'ctx> UsedTemplateParams<'ctx> {
    fn check_edge(k: EdgeKind) -> bool {
        match k {
            EdgeKind::TemplateArgument
            | EdgeKind::BaseMember
            | EdgeKind::Field
            | EdgeKind::Constructor
            | EdgeKind::Destructor
            | EdgeKind::VarType
            | EdgeKind::FunctionReturn
            | EdgeKind::FunctionParameter
            | EdgeKind::TypeReference => true,
            EdgeKind::InnerVar | EdgeKind::InnerType => false,
            EdgeKind::Method => false,
            EdgeKind::TemplateDeclaration | EdgeKind::TemplateParameterDefinition => false,
            EdgeKind::Generic => false,
        }
    }

    fn take_this_id_usage_set<Id: Into<ItemId>>(&mut self, this_id: Id) -> ItemSet {
        let this_id = this_id.into();
        self.used
            .get_mut(&this_id)
            .expect(
                "Should have a set of used template params for every item \
                 id",
            )
            .take()
            .expect(
                "Should maintain the invariant that all used template param \
                 sets are `Some` upon entry of `constrain`",
            )
    }

    fn constrain_instantiation_of_blocklisted_template(
        &self,
        this_id: ItemId,
        used_by_this_id: &mut ItemSet,
        instantiation: &TemplateInstantiation,
    ) {
        trace!(
            "    instantiation of blocklisted template, uses all template \
             arguments"
        );
        let args = instantiation
            .template_arguments()
            .iter()
            .map(|x| {
                x.into_resolver()
                    .through_type_refs()
                    .through_type_aliases()
                    .resolve(self.ctx)
                    .id()
            })
            .filter(|x| *x != this_id)
            .flat_map(|x| {
                self.used
                    .get(&x)
                    .expect("Should have a used entry for the template arg")
                    .as_ref()
                    .expect(
                        "Because a != this_id, and all used template \
                         param sets other than this_id's are `Some`, \
                         a's used template param set should be `Some`",
                    )
                    .iter()
                    .cloned()
            });
        used_by_this_id.extend(args);
    }

    fn constrain_instantiation(
        &self,
        this_id: ItemId,
        used_by_this_id: &mut ItemSet,
        instantiation: &TemplateInstantiation,
    ) {
        trace!("    template instantiation");
        let decl = self.ctx.resolve_type(instantiation.template_definition());
        let args = instantiation.template_arguments();
        let params = decl.self_template_params(self.ctx);
        debug_assert!(this_id != instantiation.template_definition());
        let used_by_def = self
            .used
            .get(&instantiation.template_definition().into())
            .expect("Should have a used entry for instantiation's template definition")
            .as_ref()
            .expect(
                "And it should be Some because only this_id's set is None, and an \
                     instantiation's template definition should never be the \
                     instantiation itself",
            );

        for (arg, param) in args.iter().zip(params.iter()) {
            trace!(
                "      instantiation's argument {:?} is used if definition's \
                 parameter {:?} is used",
                arg,
                param
            );

            if used_by_def.contains(&param.into()) {
                trace!("        param is used by template definition");
                let arg = arg
                    .into_resolver()
                    .through_type_refs()
                    .through_type_aliases()
                    .resolve(self.ctx)
                    .id();
                if arg == this_id {
                    continue;
                }
                let used_by_arg = self
                    .used
                    .get(&arg)
                    .expect("Should have a used entry for the template arg")
                    .as_ref()
                    .expect(
                        "Because arg != this_id, and all used template \
                         param sets other than this_id's are `Some`, \
                         arg's used template param set should be \
                         `Some`",
                    )
                    .iter()
                    .cloned();
                used_by_this_id.extend(used_by_arg);
            }
        }
    }

    fn constrain_join(&self, used_by_this_id: &mut ItemSet, item: &Item) {
        trace!("    other item: join with successors' usage");
        item.trace(
            self.ctx,
            &mut |sub_id, edge_kind| {
                if sub_id == item.id() || !Self::check_edge(edge_kind) {
                    return;
                }
                let used_by_sub_id = self
                    .used
                    .get(&sub_id)
                    .expect("Should have a used set for the sub_id successor")
                    .as_ref()
                    .expect(
                        "Because sub_id != id, and all used template \
                         param sets other than id's are `Some`, \
                         sub_id's used template param set should be \
                         `Some`",
                    )
                    .iter()
                    .cloned();
                trace!(
                    "      union with {:?}'s usage: {:?}",
                    sub_id,
                    used_by_sub_id.clone().collect::<Vec<_>>()
                );
                used_by_this_id.extend(used_by_sub_id);
            },
            &(),
        );
    }
}

impl<'ctx> Monotone for UsedTemplateParams<'ctx> {
    type Node = ItemId;
    type Extra = &'ctx BindgenContext;
    type Output = HashMap<ItemId, ItemSet>;

    fn new(ctx: &'ctx BindgenContext) -> UsedTemplateParams<'ctx> {
        let mut used = HashMap::default();
        let mut deps = HashMap::default();
        let allowed_items: HashSet<_> = ctx.allowed_items().iter().cloned().collect();
        let allowed_and_blocklisted_items: ItemSet = allowed_items
            .iter()
            .cloned()
            .flat_map(|i| {
                let mut ys = vec![i];
                i.trace(
                    ctx,
                    &mut |i2, _| {
                        ys.push(i2);
                    },
                    &(),
                );
                ys
            })
            .collect();
        for i in allowed_and_blocklisted_items {
            deps.entry(i).or_insert_with(Vec::new);
            used.entry(i).or_insert_with(|| Some(ItemSet::new()));
            {
                i.trace(
                    ctx,
                    &mut |i2: ItemId, _| {
                        used.entry(i2).or_insert_with(|| Some(ItemSet::new()));
                        deps.entry(i2).or_insert_with(Vec::new).push(i);
                    },
                    &(),
                );
            }
            let k = ctx.resolve_item(i).as_type().map(|ty| ty.kind());
            if let Some(TypeKind::TemplateInstantiation(inst)) = k {
                let decl = ctx.resolve_type(inst.template_definition());
                let args = inst.template_arguments();
                let ps = decl.self_template_params(ctx);
                for (arg, p) in args.iter().zip(ps.iter()) {
                    let arg = arg
                        .into_resolver()
                        .through_type_aliases()
                        .through_type_refs()
                        .resolve(ctx)
                        .id();
                    let p = p
                        .into_resolver()
                        .through_type_aliases()
                        .through_type_refs()
                        .resolve(ctx)
                        .id();
                    used.entry(arg).or_insert_with(|| Some(ItemSet::new()));
                    used.entry(p).or_insert_with(|| Some(ItemSet::new()));
                    deps.entry(arg).or_insert_with(Vec::new).push(p);
                }
            }
        }
        if cfg!(feature = "__testing_only_extra_assertions") {
            for i in allowed_items.iter() {
                extra_assert!(used.contains_key(i));
                extra_assert!(deps.contains_key(i));
                i.trace(
                    ctx,
                    &mut |i2, _| {
                        extra_assert!(used.contains_key(&i2));
                        extra_assert!(deps.contains_key(&i2));
                    },
                    &(),
                )
            }
        }
        UsedTemplateParams {
            ctx,
            used,
            deps,
            allowed_items,
        }
    }

    fn initial_worklist(&self) -> Vec<ItemId> {
        self.ctx
            .allowed_items()
            .iter()
            .cloned()
            .flat_map(|i| {
                let mut ys = vec![i];
                i.trace(
                    self.ctx,
                    &mut |i2, _| {
                        ys.push(i2);
                    },
                    &(),
                );
                ys
            })
            .collect()
    }

    fn constrain(&mut self, id: ItemId) -> YConstrain {
        extra_assert!(self.used.values().all(|v| v.is_some()));
        let mut used_by_this_id = self.take_this_id_usage_set(id);
        trace!("constrain {:?}", id);
        trace!("  initially, used set is {:?}", used_by_this_id);
        let original_len = used_by_this_id.len();
        let item = self.ctx.resolve_item(id);
        let ty_kind = item.as_type().map(|ty| ty.kind());
        match ty_kind {
            Some(&TypeKind::TypeParam) => {
                trace!("    named type, trivially uses itself");
                used_by_this_id.insert(id);
            },
            Some(TypeKind::TemplateInstantiation(inst)) => {
                if self.allowed_items.contains(&inst.template_definition().into()) {
                    self.constrain_instantiation(id, &mut used_by_this_id, inst);
                } else {
                    self.constrain_instantiation_of_blocklisted_template(id, &mut used_by_this_id, inst);
                }
            },
            _ => self.constrain_join(&mut used_by_this_id, item),
        }
        trace!("  finally, used set is {:?}", used_by_this_id);
        let new_len = used_by_this_id.len();
        assert!(
            new_len >= original_len,
            "This is the property that ensures this function is monotone -- \
             if it doesn't hold, the analysis might never terminate!"
        );
        debug_assert!(self.used[&id].is_none());
        self.used.insert(id, Some(used_by_this_id));
        extra_assert!(self.used.values().all(|v| v.is_some()));
        if new_len != original_len {
            YConstrain::Changed
        } else {
            YConstrain::Same
        }
    }

    fn each_depending_on<F>(&self, id: ItemId, mut f: F)
    where
        F: FnMut(ItemId),
    {
        if let Some(es) = self.deps.get(&id) {
            for e in es {
                trace!("enqueue {:?} into worklist", e);
                f(*e);
            }
        }
    }
}

impl<'ctx> From<UsedTemplateParams<'ctx>> for HashMap<ItemId, ItemSet> {
    fn from(x: UsedTemplateParams<'ctx>) -> Self {
        x.used.into_iter().map(|(k, v)| (k, v.unwrap())).collect()
    }
}
