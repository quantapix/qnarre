use super::{ConstrainResult, MonotoneFramework};
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::item::{Item, ItemSet};
use crate::ir::template::{TemplateInstantiation, TemplateParameters};
use crate::ir::traversal::{EdgeKind, Trace};
use crate::ir::ty::TypeKind;
use crate::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct UsedTemplateParameters<'ctx> {
    ctx: &'ctx BindgenContext,

    used: HashMap<ItemId, Option<ItemSet>>,

    dependencies: HashMap<ItemId, Vec<ItemId>>,

    allowlisted_items: HashSet<ItemId>,
}

impl<'ctx> UsedTemplateParameters<'ctx> {
    fn consider_edge(kind: EdgeKind) -> bool {
        match kind {
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
            .map(|a| {
                a.into_resolver()
                    .through_type_refs()
                    .through_type_aliases()
                    .resolve(self.ctx)
                    .id()
            })
            .filter(|a| *a != this_id)
            .flat_map(|a| {
                self.used
                    .get(&a)
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
                // Ignore ourselves, since union with ourself is a
                // no-op. Ignore edges that aren't relevant to the
                // analysis.
                if sub_id == item.id() || !Self::consider_edge(edge_kind) {
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

impl<'ctx> MonotoneFramework for UsedTemplateParameters<'ctx> {
    type Node = ItemId;
    type Extra = &'ctx BindgenContext;
    type Output = HashMap<ItemId, ItemSet>;

    fn new(ctx: &'ctx BindgenContext) -> UsedTemplateParameters<'ctx> {
        let mut used = HashMap::default();
        let mut dependencies = HashMap::default();
        let allowlisted_items: HashSet<_> = ctx.allowlisted_items().iter().cloned().collect();

        let allowlisted_and_blocklisted_items: ItemSet = allowlisted_items
            .iter()
            .cloned()
            .flat_map(|i| {
                let mut reachable = vec![i];
                i.trace(
                    ctx,
                    &mut |s, _| {
                        reachable.push(s);
                    },
                    &(),
                );
                reachable
            })
            .collect();

        for item in allowlisted_and_blocklisted_items {
            dependencies.entry(item).or_insert_with(Vec::new);
            used.entry(item).or_insert_with(|| Some(ItemSet::new()));

            {
                // We reverse our natural IR graph edges to find dependencies
                // between nodes.
                item.trace(
                    ctx,
                    &mut |sub_item: ItemId, _| {
                        used.entry(sub_item).or_insert_with(|| Some(ItemSet::new()));
                        dependencies.entry(sub_item).or_insert_with(Vec::new).push(item);
                    },
                    &(),
                );
            }

            let item_kind = ctx.resolve_item(item).as_type().map(|ty| ty.kind());
            if let Some(TypeKind::TemplateInstantiation(inst)) = item_kind {
                let decl = ctx.resolve_type(inst.template_definition());
                let args = inst.template_arguments();

                // Although template definitions should always have
                // template parameters, there is a single exception:
                // opaque templates. Hence the unwrap_or.
                let params = decl.self_template_params(ctx);

                for (arg, param) in args.iter().zip(params.iter()) {
                    let arg = arg
                        .into_resolver()
                        .through_type_aliases()
                        .through_type_refs()
                        .resolve(ctx)
                        .id();

                    let param = param
                        .into_resolver()
                        .through_type_aliases()
                        .through_type_refs()
                        .resolve(ctx)
                        .id();

                    used.entry(arg).or_insert_with(|| Some(ItemSet::new()));
                    used.entry(param).or_insert_with(|| Some(ItemSet::new()));

                    dependencies.entry(arg).or_insert_with(Vec::new).push(param);
                }
            }
        }

        if cfg!(feature = "__testing_only_extra_assertions") {
            for item in allowlisted_items.iter() {
                extra_assert!(used.contains_key(item));
                extra_assert!(dependencies.contains_key(item));
                item.trace(
                    ctx,
                    &mut |sub_item, _| {
                        extra_assert!(used.contains_key(&sub_item));
                        extra_assert!(dependencies.contains_key(&sub_item));
                    },
                    &(),
                )
            }
        }

        UsedTemplateParameters {
            ctx,
            used,
            dependencies,
            allowlisted_items,
        }
    }

    fn initial_worklist(&self) -> Vec<ItemId> {
        self.ctx
            .allowlisted_items()
            .iter()
            .cloned()
            .flat_map(|i| {
                let mut reachable = vec![i];
                i.trace(
                    self.ctx,
                    &mut |s, _| {
                        reachable.push(s);
                    },
                    &(),
                );
                reachable
            })
            .collect()
    }

    fn constrain(&mut self, id: ItemId) -> ConstrainResult {
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
                if self.allowlisted_items.contains(&inst.template_definition().into()) {
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
            ConstrainResult::Changed
        } else {
            ConstrainResult::Same
        }
    }

    fn each_depending_on<F>(&self, item: ItemId, mut f: F)
    where
        F: FnMut(ItemId),
    {
        if let Some(edges) = self.dependencies.get(&item) {
            for item in edges {
                trace!("enqueue {:?} into worklist", item);
                f(*item);
            }
        }
    }
}

impl<'ctx> From<UsedTemplateParameters<'ctx>> for HashMap<ItemId, ItemSet> {
    fn from(used_templ_params: UsedTemplateParameters<'ctx>) -> Self {
        used_templ_params
            .used
            .into_iter()
            .map(|(k, v)| (k, v.unwrap()))
            .collect()
    }
}
