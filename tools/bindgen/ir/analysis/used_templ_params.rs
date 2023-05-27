use super::{Monotone, YConstrain};
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::item::{Item, ItemSet};
use crate::ir::template::{TemplateInstantiation, TemplateParameters};
use crate::ir::traversal::{EdgeKind, Trace};
use crate::ir::ty::TypeKind;
use crate::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct UsedTemplParamsAnalysis<'ctx> {
    ctx: &'ctx BindgenContext,
    ys: HashMap<ItemId, Option<ItemSet>>,
    deps: HashMap<ItemId, Vec<ItemId>>,
    alloweds: HashSet<ItemId>,
}

impl<'ctx> UsedTemplParamsAnalysis<'ctx> {
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

    fn take_this_id_usage_set<Id: Into<ItemId>>(&mut self, id: Id) -> ItemSet {
        let id = id.into();
        self.ys
            .get_mut(&id)
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
        id: ItemId,
        y: &mut ItemSet,
        inst: &TemplateInstantiation,
    ) {
        let args = inst
            .template_arguments()
            .iter()
            .map(|x| {
                x.into_resolver()
                    .through_type_refs()
                    .through_type_aliases()
                    .resolve(self.ctx)
                    .id()
            })
            .filter(|x| *x != id)
            .flat_map(|x| {
                self.ys
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
        y.extend(args);
    }

    fn constrain_instantiation(&self, id: ItemId, y: &mut ItemSet, inst: &TemplateInstantiation) {
        let decl = self.ctx.resolve_type(inst.template_definition());
        let args = inst.template_arguments();
        let ps = decl.self_template_params(self.ctx);
        debug_assert!(id != inst.template_definition());
        let used_by_def = self
            .ys
            .get(&inst.template_definition().into())
            .expect("Should have a used entry for instantiation's template definition")
            .as_ref()
            .expect(
                "And it should be Some because only this_id's set is None, and an \
                     instantiation's template definition should never be the \
                     instantiation itself",
            );
        for (arg, p) in args.iter().zip(ps.iter()) {
            if used_by_def.contains(&p.into()) {
                let arg = arg
                    .into_resolver()
                    .through_type_refs()
                    .through_type_aliases()
                    .resolve(self.ctx)
                    .id();
                if arg == id {
                    continue;
                }
                let used_by_arg = self
                    .ys
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
                y.extend(used_by_arg);
            }
        }
    }

    fn constrain_join(&self, y: &mut ItemSet, i: &Item) {
        i.trace(
            self.ctx,
            &mut |i2, kind| {
                if i2 == i.id() || !Self::check_edge(kind) {
                    return;
                }
                let y2 = self
                    .ys
                    .get(&i2)
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
                y.extend(y2);
            },
            &(),
        );
    }
}

impl<'ctx> Monotone for UsedTemplParamsAnalysis<'ctx> {
    type Node = ItemId;
    type Extra = &'ctx BindgenContext;
    type Output = HashMap<ItemId, ItemSet>;

    fn new(ctx: &'ctx BindgenContext) -> UsedTemplParamsAnalysis<'ctx> {
        let mut ys = HashMap::default();
        let mut deps = HashMap::default();
        let alloweds: HashSet<_> = ctx.allowed_items().iter().cloned().collect();
        let allowed_and_blocklisted_items: ItemSet = alloweds
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
            ys.entry(i).or_insert_with(|| Some(ItemSet::new()));
            {
                i.trace(
                    ctx,
                    &mut |i2: ItemId, _| {
                        ys.entry(i2).or_insert_with(|| Some(ItemSet::new()));
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
                    ys.entry(arg).or_insert_with(|| Some(ItemSet::new()));
                    ys.entry(p).or_insert_with(|| Some(ItemSet::new()));
                    deps.entry(arg).or_insert_with(Vec::new).push(p);
                }
            }
        }
        if cfg!(feature = "__testing_only_extra_assertions") {
            for i in alloweds.iter() {
                extra_assert!(ys.contains_key(i));
                extra_assert!(deps.contains_key(i));
                i.trace(
                    ctx,
                    &mut |i2, _| {
                        extra_assert!(ys.contains_key(&i2));
                        extra_assert!(deps.contains_key(&i2));
                    },
                    &(),
                )
            }
        }
        UsedTemplParamsAnalysis {
            ctx,
            ys,
            deps,
            alloweds,
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
        extra_assert!(self.ys.values().all(|v| v.is_some()));
        let mut y = self.take_this_id_usage_set(id);
        let len = y.len();
        let i = self.ctx.resolve_item(id);
        let ty_kind = i.as_type().map(|x| x.kind());
        match ty_kind {
            Some(&TypeKind::TypeParam) => {
                y.insert(id);
            },
            Some(TypeKind::TemplateInstantiation(x)) => {
                if self.alloweds.contains(&x.template_definition().into()) {
                    self.constrain_instantiation(id, &mut y, x);
                } else {
                    self.constrain_instantiation_of_blocklisted_template(id, &mut y, x);
                }
            },
            _ => self.constrain_join(&mut y, i),
        }
        let len2 = y.len();
        assert!(len2 >= len);
        debug_assert!(self.ys[&id].is_none());
        self.ys.insert(id, Some(y));
        extra_assert!(self.ys.values().all(|v| v.is_some()));
        if len2 != len {
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
                f(*e);
            }
        }
    }
}

impl<'ctx> From<UsedTemplParamsAnalysis<'ctx>> for HashMap<ItemId, ItemSet> {
    fn from(x: UsedTemplParamsAnalysis<'ctx>) -> Self {
        x.ys.into_iter().map(|(k, v)| (k, v.unwrap())).collect()
    }
}
