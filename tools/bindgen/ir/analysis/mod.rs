mod used_templ_params;
pub(crate) use self::used_templ_params::UsedTemplParamsAnalysis;
mod derive;
pub use self::derive::DeriveTrait;
pub(crate) use self::derive::{as_cannot_derive_set, DeriveAnalysis};
mod has_destructor;
pub(crate) use self::has_destructor::HasDestructorAnalysis;
mod has_type_param;
pub(crate) use self::has_type_param::HasTyParamInArrayAnalysis;

mod sizedness;
pub(crate) use self::sizedness::{Sizedness, SizednessAnalysis, YSizedness};

use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::{EdgeKind, Trace};
use crate::HashMap;
use std::fmt;
use std::ops;

pub(crate) trait Monotone: Sized + fmt::Debug {
    type Node: Copy;
    type Extra: Sized;
    type Output: From<Self> + fmt::Debug;
    fn new(x: Self::Extra) -> Self;
    fn initial_worklist(&self) -> Vec<Self::Node>;
    fn constrain(&mut self, n: Self::Node) -> YConstrain;
    fn each_depending_on<F>(&self, n: Self::Node, f: F)
    where
        F: FnMut(Self::Node);
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum YConstrain {
    Changed,
    Same,
}

impl Default for YConstrain {
    fn default() -> Self {
        YConstrain::Same
    }
}

impl ops::BitOr for YConstrain {
    type Output = Self;
    fn bitor(self, rhs: YConstrain) -> Self::Output {
        if self == YConstrain::Changed || rhs == YConstrain::Changed {
            YConstrain::Changed
        } else {
            YConstrain::Same
        }
    }
}

impl ops::BitOrAssign for YConstrain {
    fn bitor_assign(&mut self, rhs: YConstrain) {
        *self = *self | rhs;
    }
}

pub(crate) fn analyze<T>(x: T::Extra) -> T::Output
where
    T: Monotone,
{
    let mut y = T::new(x);
    let mut ns = y.initial_worklist();
    while let Some(n) = ns.pop() {
        if let YConstrain::Changed = y.constrain(n) {
            y.each_depending_on(n, |x| {
                ns.push(x);
            });
        }
    }
    y.into()
}

pub(crate) fn gen_deps<F>(ctx: &BindgenContext, f: F) -> HashMap<ItemId, Vec<ItemId>>
where
    F: Fn(EdgeKind) -> bool,
{
    let mut ys = HashMap::default();
    for &i in ctx.allowed_items() {
        ys.entry(i).or_insert_with(Vec::new);
        {
            i.trace(
                ctx,
                &mut |i2: ItemId, kind| {
                    if ctx.allowed_items().contains(&i2) && f(kind) {
                        ys.entry(i2).or_insert_with(Vec::new).push(i);
                    }
                },
                &(),
            );
        }
    }
    ys
}

pub(crate) mod has_float {
    use super::{gen_deps, Monotone, YConstrain};
    use crate::ir::comp::Field;
    use crate::ir::comp::FieldMethods;
    use crate::ir::context::{BindgenContext, ItemId};
    use crate::ir::traversal::EdgeKind;
    use crate::ir::ty::TyKind;
    use crate::{HashMap, HashSet};

    #[derive(Debug, Clone)]
    pub struct Analysis<'ctx> {
        ctx: &'ctx BindgenContext,
        ys: HashSet<ItemId>,
        deps: HashMap<ItemId, Vec<ItemId>>,
    }

    impl<'ctx> Analysis<'ctx> {
        fn check_edge(k: EdgeKind) -> bool {
            match k {
                EdgeKind::BaseMember
                | EdgeKind::Field
                | EdgeKind::TypeReference
                | EdgeKind::VarType
                | EdgeKind::TemplateArgument
                | EdgeKind::TemplateDeclaration
                | EdgeKind::TemplateParameterDefinition => true,

                EdgeKind::Constructor
                | EdgeKind::Destructor
                | EdgeKind::FunctionReturn
                | EdgeKind::FunctionParameter
                | EdgeKind::InnerType
                | EdgeKind::InnerVar
                | EdgeKind::Method => false,
                EdgeKind::Generic => false,
            }
        }

        fn insert<Id: Into<ItemId>>(&mut self, id: Id) -> YConstrain {
            let id = id.into();
            let newly = self.ys.insert(id);
            assert!(newly);
            YConstrain::Changed
        }
    }

    impl<'ctx> Monotone for Analysis<'ctx> {
        type Node = ItemId;
        type Extra = &'ctx BindgenContext;
        type Output = HashSet<ItemId>;

        fn new(ctx: &'ctx BindgenContext) -> Analysis<'ctx> {
            let ys = HashSet::default();
            let deps = gen_deps(ctx, Self::check_edge);
            Analysis { ctx, ys, deps }
        }

        fn initial_worklist(&self) -> Vec<ItemId> {
            self.ctx.allowed_items().iter().cloned().collect()
        }

        fn constrain(&mut self, id: ItemId) -> YConstrain {
            if self.ys.contains(&id) {
                return YConstrain::Same;
            }
            let i = self.ctx.resolve_item(id);
            let ty = match i.as_type() {
                Some(ty) => ty,
                None => {
                    return YConstrain::Same;
                },
            };
            match *ty.kind() {
                TyKind::Void
                | TyKind::NullPtr
                | TyKind::Int(..)
                | TyKind::Function(..)
                | TyKind::Enum(..)
                | TyKind::Reference(..)
                | TyKind::TypeParam
                | TyKind::Opaque
                | TyKind::Pointer(..)
                | TyKind::UnresolvedTypeRef(..) => YConstrain::Same,
                TyKind::Float(..) | TyKind::Complex(..) => self.insert(id),
                TyKind::Array(t, _) => {
                    if self.ys.contains(&t.into()) {
                        return self.insert(id);
                    }
                    YConstrain::Same
                },
                TyKind::Vector(t, _) => {
                    if self.ys.contains(&t.into()) {
                        return self.insert(id);
                    }
                    YConstrain::Same
                },
                TyKind::ResolvedTypeRef(t)
                | TyKind::TemplateAlias(t, _)
                | TyKind::Alias(t)
                | TyKind::BlockPointer(t) => {
                    if self.ys.contains(&t.into()) {
                        self.insert(id)
                    } else {
                        YConstrain::Same
                    }
                },
                TyKind::Comp(ref x) => {
                    let bases = x.base_members().iter().any(|x| self.ys.contains(&x.ty.into()));
                    if bases {
                        return self.insert(id);
                    }
                    let fields = x.fields().iter().any(|x| match *x {
                        Field::DataMember(ref x) => self.ys.contains(&x.ty().into()),
                        Field::Bitfields(ref x) => x.bitfields().iter().any(|x| self.ys.contains(&x.ty().into())),
                    });
                    if fields {
                        return self.insert(id);
                    }
                    YConstrain::Same
                },
                TyKind::TemplateInstantiation(ref t) => {
                    let args = t.template_arguments().iter().any(|x| self.ys.contains(&x.into()));
                    if args {
                        return self.insert(id);
                    }
                    let def = self.ys.contains(&t.template_definition().into());
                    if def {
                        return self.insert(id);
                    }
                    YConstrain::Same
                },
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

    impl<'ctx> From<Analysis<'ctx>> for HashSet<ItemId> {
        fn from(x: Analysis<'ctx>) -> Self {
            x.ys
        }
    }
}

pub(crate) trait HasVtable {
    fn has_vtable(&self, ctx: &BindgenContext) -> bool;
    fn has_vtable_ptr(&self, ctx: &BindgenContext) -> bool;
}

pub(crate) mod has_vtable {
    use super::{gen_deps, Monotone, YConstrain};
    use crate::ir::context::{BindgenContext, ItemId};
    use crate::ir::traversal::EdgeKind;
    use crate::ir::ty::TyKind;
    use crate::{Entry, HashMap};
    use std::cmp;
    use std::ops;

    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Result {
        No,
        SelfHasVtable,
        BaseHasVtable,
    }

    impl Default for Result {
        fn default() -> Self {
            Result::No
        }
    }

    impl Result {
        pub(crate) fn join(self, rhs: Self) -> Self {
            cmp::max(self, rhs)
        }
    }

    impl ops::BitOr for Result {
        type Output = Self;
        fn bitor(self, rhs: Result) -> Self::Output {
            self.join(rhs)
        }
    }

    impl ops::BitOrAssign for Result {
        fn bitor_assign(&mut self, rhs: Result) {
            *self = self.join(rhs)
        }
    }

    #[derive(Debug, Clone)]
    pub struct Analysis<'ctx> {
        ctx: &'ctx BindgenContext,
        ys: HashMap<ItemId, Result>,
        deps: HashMap<ItemId, Vec<ItemId>>,
    }

    impl<'ctx> Analysis<'ctx> {
        fn check_edge(k: EdgeKind) -> bool {
            matches!(
                k,
                EdgeKind::TypeReference | EdgeKind::BaseMember | EdgeKind::TemplateDeclaration
            )
        }

        fn insert<Id: Into<ItemId>>(&mut self, id: Id, y: Result) -> YConstrain {
            if let Result::No = y {
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

    impl<'ctx> Monotone for Analysis<'ctx> {
        type Node = ItemId;
        type Extra = &'ctx BindgenContext;
        type Output = HashMap<ItemId, Result>;

        fn new(ctx: &'ctx BindgenContext) -> Analysis<'ctx> {
            let ys = HashMap::default();
            let deps = gen_deps(ctx, Self::check_edge);
            Analysis { ctx, ys, deps }
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
                TyKind::TemplateAlias(t, _) | TyKind::Alias(t) | TyKind::ResolvedTypeRef(t) | TyKind::Reference(t) => {
                    self.forward(t, id)
                },
                TyKind::Comp(ref info) => {
                    let mut y = Result::No;
                    if info.has_own_virtual_method() {
                        y |= Result::SelfHasVtable;
                    }
                    let has_vtable = info.base_members().iter().any(|x| self.ys.contains_key(&x.ty.into()));
                    if has_vtable {
                        y |= Result::BaseHasVtable;
                    }
                    self.insert(id, y)
                },
                TyKind::TemplateInstantiation(ref x) => self.forward(x.template_definition(), id),
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

    impl<'ctx> From<Analysis<'ctx>> for HashMap<ItemId, Result> {
        fn from(x: Analysis<'ctx>) -> Self {
            extra_assert!(x.ys.values().all(|x| { *x != Result::No }));
            x.ys
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HashMap, HashSet};

    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    struct Node(usize);

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct Graph(HashMap<Node, Vec<Node>>);

    impl Graph {
        fn make_test_graph() -> Graph {
            let mut y = Graph::default();
            y.0.insert(Node(1), vec![Node(3)]);
            y.0.insert(Node(2), vec![Node(2)]);
            y.0.insert(Node(3), vec![Node(4), Node(5)]);
            y.0.insert(Node(4), vec![Node(7)]);
            y.0.insert(Node(5), vec![Node(6), Node(7)]);
            y.0.insert(Node(6), vec![Node(8)]);
            y.0.insert(Node(7), vec![Node(3)]);
            y.0.insert(Node(8), vec![]);
            y
        }

        fn reverse(&self) -> Graph {
            let mut y = Graph::default();
            for (node, edges) in self.0.iter() {
                y.0.entry(*node).or_insert_with(Vec::new);
                for referent in edges.iter() {
                    y.0.entry(*referent).or_insert_with(Vec::new).push(*node);
                }
            }
            y
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ReachableFrom<'a> {
        reachable: HashMap<Node, HashSet<Node>>,
        graph: &'a Graph,
        reversed: Graph,
    }

    impl<'a> Monotone for ReachableFrom<'a> {
        type Node = Node;
        type Extra = &'a Graph;
        type Output = HashMap<Node, HashSet<Node>>;

        fn new(graph: &'a Graph) -> ReachableFrom {
            let reversed = graph.reverse();
            ReachableFrom {
                reachable: Default::default(),
                graph,
                reversed,
            }
        }

        fn initial_worklist(&self) -> Vec<Node> {
            self.graph.0.keys().cloned().collect()
        }

        fn constrain(&mut self, n: Node) -> YConstrain {
            let s = self.reachable.entry(n).or_insert_with(HashSet::default).len();
            for n2 in self.graph.0[&n].iter() {
                self.reachable.get_mut(&n).unwrap().insert(*n2);
                let r2 = self.reachable.entry(*n2).or_insert_with(HashSet::default).clone();
                for transitive in r2 {
                    self.reachable.get_mut(&n).unwrap().insert(transitive);
                }
            }
            let s2 = self.reachable[&n].len();
            if s != s2 {
                YConstrain::Changed
            } else {
                YConstrain::Same
            }
        }

        fn each_depending_on<F>(&self, n: Node, mut f: F)
        where
            F: FnMut(Node),
        {
            for dep in self.reversed.0[&n].iter() {
                f(*dep);
            }
        }
    }

    impl<'a> From<ReachableFrom<'a>> for HashMap<Node, HashSet<Node>> {
        fn from(x: ReachableFrom<'a>) -> Self {
            x.reachable
        }
    }

    #[test]
    fn monotone() {
        let g = Graph::make_test_graph();
        let y = analyze::<ReachableFrom>(&g);
        println!("reachable = {:#?}", y);
        fn nodes<A>(x: A) -> HashSet<Node>
        where
            A: AsRef<[usize]>,
        {
            x.as_ref().iter().cloned().map(Node).collect()
        }
        let mut y2 = HashMap::default();
        y2.insert(Node(1), nodes([3, 4, 5, 6, 7, 8]));
        y2.insert(Node(2), nodes([2]));
        y2.insert(Node(3), nodes([3, 4, 5, 6, 7, 8]));
        y2.insert(Node(4), nodes([3, 4, 5, 6, 7, 8]));
        y2.insert(Node(5), nodes([3, 4, 5, 6, 7, 8]));
        y2.insert(Node(6), nodes([8]));
        y2.insert(Node(7), nodes([3, 4, 5, 6, 7, 8]));
        y2.insert(Node(8), nodes([]));
        println!("expected = {:#?}", y2);
        assert_eq!(y, y2);
    }
}
