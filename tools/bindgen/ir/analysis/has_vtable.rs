use super::{gen_deps, Monotone, YConstrain};
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::EdgeKind;
use crate::ir::ty::TypeKind;
use crate::{Entry, HashMap};
use std::cmp;
use std::ops;

pub(crate) trait HasVtable {
    fn has_vtable(&self, ctx: &BindgenContext) -> bool;
    fn has_vtable_ptr(&self, ctx: &BindgenContext) -> bool;
}

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
    ys: HashMap<ItemId, YHasVtable>,
    deps: HashMap<ItemId, Vec<ItemId>>,
}

impl<'ctx> HasVtableAnalysis<'ctx> {
    fn check_edge(k: EdgeKind) -> bool {
        matches!(
            k,
            EdgeKind::TypeReference | EdgeKind::BaseMember | EdgeKind::TemplateDeclaration
        )
    }

    fn insert<Id: Into<ItemId>>(&mut self, id: Id, y: YHasVtable) -> YConstrain {
        if let YHasVtable::No = y {
            return YConstrain::Same;
        }
        let id = id.into();
        match self.ys.entry(id) {
            Entry::Occupied(mut x) => {
                if *x.get() < y {
                    x.insert(y);
                    YConstrain::Changed
                } else {
                    YConstrain::Same
                }
            },
            Entry::Vacant(x) => {
                x.insert(y);
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
        match self.ys.get(&from).cloned() {
            None => YConstrain::Same,
            Some(x) => self.insert(to, x),
        }
    }
}

impl<'ctx> Monotone for HasVtableAnalysis<'ctx> {
    type Node = ItemId;
    type Extra = &'ctx BindgenContext;
    type Output = HashMap<ItemId, YHasVtable>;

    fn new(ctx: &'ctx BindgenContext) -> HasVtableAnalysis<'ctx> {
        let ys = HashMap::default();
        let deps = gen_deps(ctx, Self::check_edge);
        HasVtableAnalysis { ctx, ys, deps }
    }

    fn initial_worklist(&self) -> Vec<ItemId> {
        self.ctx.allowed_items().iter().cloned().collect()
    }

    fn constrain(&mut self, id: ItemId) -> YConstrain {
        let i = self.ctx.resolve_item(id);
        let ty = match i.as_type() {
            None => return YConstrain::Same,
            Some(ty) => ty,
        };
        match *ty.kind() {
            TypeKind::TemplateAlias(t, _)
            | TypeKind::Alias(t)
            | TypeKind::ResolvedTypeRef(t)
            | TypeKind::Reference(t) => self.forward(t, id),
            TypeKind::Comp(ref info) => {
                let mut y = YHasVtable::No;
                if info.has_own_virtual_method() {
                    y |= YHasVtable::SelfHasVtable;
                }
                let has_vtable = info.base_members().iter().any(|x| self.ys.contains_key(&x.ty.into()));
                if has_vtable {
                    y |= YHasVtable::BaseHasVtable;
                }
                self.insert(id, y)
            },
            TypeKind::TemplateInstantiation(ref x) => self.forward(x.template_definition(), id),
            _ => YConstrain::Same,
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

impl<'ctx> From<HasVtableAnalysis<'ctx>> for HashMap<ItemId, YHasVtable> {
    fn from(x: HasVtableAnalysis<'ctx>) -> Self {
        extra_assert!(x.ys.values().all(|x| { *x != YHasVtable::No }));
        x.ys
    }
}
