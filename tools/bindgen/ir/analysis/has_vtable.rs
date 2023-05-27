use super::{gen_deps, Monotone, YConstrain};
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::EdgeKind;
use crate::ir::ty::TypeKind;
use crate::{Entry, HashMap};
use std::cmp;
use std::ops;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum YHasVtable {
    No,
    SelfHasVtable,
    BaseHasVtable,
}

impl Default for YHasVtable {
    fn default() -> Self {
        YHasVtable::No
    }
}

impl YHasVtable {
    pub(crate) fn join(self, rhs: Self) -> Self {
        cmp::max(self, rhs)
    }
}

impl ops::BitOr for YHasVtable {
    type Output = Self;
    fn bitor(self, rhs: YHasVtable) -> Self::Output {
        self.join(rhs)
    }
}

impl ops::BitOrAssign for YHasVtable {
    fn bitor_assign(&mut self, rhs: YHasVtable) {
        *self = self.join(rhs)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HasVtableAnalysis<'ctx> {
    ctx: &'ctx BindgenContext,
    have_vtable: HashMap<ItemId, YHasVtable>,
    deps: HashMap<ItemId, Vec<ItemId>>,
}

impl<'ctx> HasVtableAnalysis<'ctx> {
    fn consider_edge(kind: EdgeKind) -> bool {
        matches!(
            kind,
            EdgeKind::TypeReference | EdgeKind::BaseMember | EdgeKind::TemplateDeclaration
        )
    }

    fn insert<Id: Into<ItemId>>(&mut self, id: Id, result: YHasVtable) -> YConstrain {
        if let YHasVtable::No = result {
            return YConstrain::Same;
        }
        let id = id.into();
        match self.have_vtable.entry(id) {
            Entry::Occupied(mut entry) => {
                if *entry.get() < result {
                    entry.insert(result);
                    YConstrain::Changed
                } else {
                    YConstrain::Same
                }
            },
            Entry::Vacant(entry) => {
                entry.insert(result);
                YConstrain::Changed
            },
        }
    }

    fn forward<Id1, Id2>(&mut self, from: Id1, to: Id2) -> YConstrain
    where
        Id1: Into<ItemId>,
        Id2: Into<ItemId>,
    {
        let from = from.into();
        let to = to.into();
        match self.have_vtable.get(&from).cloned() {
            None => YConstrain::Same,
            Some(r) => self.insert(to, r),
        }
    }
}

impl<'ctx> Monotone for HasVtableAnalysis<'ctx> {
    type Node = ItemId;
    type Extra = &'ctx BindgenContext;
    type Output = HashMap<ItemId, YHasVtable>;

    fn new(ctx: &'ctx BindgenContext) -> HasVtableAnalysis<'ctx> {
        let have_vtable = HashMap::default();
        let dependencies = gen_deps(ctx, Self::consider_edge);

        HasVtableAnalysis {
            ctx,
            have_vtable,
            deps: dependencies,
        }
    }

    fn initial_worklist(&self) -> Vec<ItemId> {
        self.ctx.allowed_items().iter().cloned().collect()
    }

    fn constrain(&mut self, id: ItemId) -> YConstrain {
        trace!("constrain {:?}", id);

        let item = self.ctx.resolve_item(id);
        let ty = match item.as_type() {
            None => return YConstrain::Same,
            Some(ty) => ty,
        };

        match *ty.kind() {
            TypeKind::TemplateAlias(t, _)
            | TypeKind::Alias(t)
            | TypeKind::ResolvedTypeRef(t)
            | TypeKind::Reference(t) => {
                trace!("    aliases and references forward to their inner type");
                self.forward(t, id)
            },

            TypeKind::Comp(ref info) => {
                trace!("    comp considers its own methods and bases");
                let mut result = YHasVtable::No;

                if info.has_own_virtual_method() {
                    trace!("    comp has its own virtual method");
                    result |= YHasVtable::SelfHasVtable;
                }

                let bases_has_vtable = info.base_members().iter().any(|base| {
                    trace!("    comp has a base with a vtable: {:?}", base);
                    self.have_vtable.contains_key(&base.ty.into())
                });
                if bases_has_vtable {
                    result |= YHasVtable::BaseHasVtable;
                }

                self.insert(id, result)
            },

            TypeKind::TemplateInstantiation(ref inst) => self.forward(inst.template_definition(), id),

            _ => YConstrain::Same,
        }
    }

    fn each_depending_on<F>(&self, id: ItemId, mut f: F)
    where
        F: FnMut(ItemId),
    {
        if let Some(edges) = self.deps.get(&id) {
            for item in edges {
                trace!("enqueue {:?} into worklist", item);
                f(*item);
            }
        }
    }
}

impl<'ctx> From<HasVtableAnalysis<'ctx>> for HashMap<ItemId, YHasVtable> {
    fn from(analysis: HasVtableAnalysis<'ctx>) -> Self {
        extra_assert!(analysis.have_vtable.values().all(|v| { *v != YHasVtable::No }));

        analysis.have_vtable
    }
}

pub(crate) trait HasVtable {
    fn has_vtable(&self, ctx: &BindgenContext) -> bool;
    fn has_vtable_ptr(&self, ctx: &BindgenContext) -> bool;
}
