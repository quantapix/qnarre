use super::item::ItemSet;
use super::{Context, ItemId};
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EdgeKind {
    Generic,
    TemplParamDef,
    TemplDecl,
    TemplArg,
    BaseMember,
    Field,
    InnerType,
    InnerVar,
    Method,
    Constructor,
    Destructor,
    FnReturn,
    FnParameter,
    VarType,
    TypeRef,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge {
    to: ItemId,
    kind: EdgeKind,
}
impl Edge {
    pub fn new(to: ItemId, kind: EdgeKind) -> Edge {
        Edge { to, kind }
    }
}
impl From<Edge> for ItemId {
    fn from(x: Edge) -> Self {
        x.to
    }
}

pub type Predicate = for<'a> fn(&'a Context, Edge) -> bool;

pub trait Storage<'ctx> {
    fn new(ctx: &'ctx Context) -> Self;
    fn add(&mut self, from: Option<ItemId>, id: ItemId) -> bool;
}
impl<'ctx> Storage<'ctx> for ItemSet {
    fn new(_: &'ctx Context) -> Self {
        ItemSet::new()
    }
    fn add(&mut self, _: Option<ItemId>, id: ItemId) -> bool {
        self.insert(id)
    }
}

#[derive(Debug)]
pub struct Paths<'ctx>(BTreeMap<ItemId, ItemId>, &'ctx Context);
impl<'ctx> Storage<'ctx> for Paths<'ctx> {
    fn new(ctx: &'ctx Context) -> Self {
        Paths(BTreeMap::new(), ctx)
    }
    fn add(&mut self, from: Option<ItemId>, id: ItemId) -> bool {
        let y = self.0.insert(id, from.unwrap_or(id)).is_none();
        if self.1.resolve_item_fallible(id).is_none() {
            let mut path = vec![];
            let mut i = id;
            loop {
                let x = *self.0.get(&i).expect("Must have a predecessor");
                if x == i {
                    break;
                }
                path.push(x);
                i = x;
            }
            path.reverse();
            panic!("Reference to dangling id = {:?} via path = {:?}", id, path);
        }
        y
    }
}

pub trait Queue: Default {
    fn push(&mut self, id: ItemId);
    fn next(&mut self) -> Option<ItemId>;
}
impl Queue for Vec<ItemId> {
    fn push(&mut self, id: ItemId) {
        self.push(id);
    }
    fn next(&mut self) -> Option<ItemId> {
        self.pop()
    }
}
impl Queue for VecDeque<ItemId> {
    fn push(&mut self, id: ItemId) {
        self.push_back(id);
    }
    fn next(&mut self) -> Option<ItemId> {
        self.pop_front()
    }
}

pub struct Traversal<'ctx, S, Q>
where
    S: Storage<'ctx>,
    Q: Queue,
{
    ctx: &'ctx Context,
    seen: S,
    queue: Q,
    pred: Predicate,
    current: Option<ItemId>,
}
impl<'ctx, S, Q> Traversal<'ctx, S, Q>
where
    S: Storage<'ctx>,
    Q: Queue,
{
    pub fn new<R>(ctx: &'ctx Context, roots: R, pred: Predicate) -> Traversal<'ctx, S, Q>
    where
        R: IntoIterator<Item = ItemId>,
    {
        let mut seen = S::new(ctx);
        let mut queue = Q::default();
        for x in roots {
            seen.add(None, x);
            queue.push(x);
        }
        Traversal {
            ctx,
            seen,
            queue,
            pred,
            current: None,
        }
    }
}
impl<'ctx, S, Q> Tracer for Traversal<'ctx, S, Q>
where
    S: Storage<'ctx>,
    Q: Queue,
{
    fn visit_kind(&mut self, id: ItemId, kind: EdgeKind) {
        let x = Edge::new(id, kind);
        if !(self.pred)(self.ctx, x) {
            return;
        }
        let newly = self.seen.add(self.current, id);
        if newly {
            self.queue.push(id)
        }
    }
}
impl<'ctx, S, Q> Iterator for Traversal<'ctx, S, Q>
where
    S: Storage<'ctx>,
    Q: Queue,
{
    type Item = ItemId;
    fn next(&mut self) -> Option<Self::Item> {
        let y = self.queue.next()?;
        let newly = self.seen.add(None, y);
        debug_assert!(!newly, "should have already seen anything we get out of our queue");
        debug_assert!(
            self.ctx.resolve_item_fallible(y).is_some(),
            "should only get IDs of actual items in our context during traversal"
        );
        self.current = Some(y);
        y.trace(self.ctx, self, &());
        self.current = None;
        Some(y)
    }
}

pub type AssertNoDanglingItemsTraversal<'ctx> = Traversal<'ctx, Paths<'ctx>, VecDeque<ItemId>>;

pub fn all_edges(_: &Context, _: Edge) -> bool {
    true
}

pub fn only_inner_types(_: &Context, x: Edge) -> bool {
    x.kind == EdgeKind::InnerType
}

pub fn enabled_edges(ctx: &Context, x: Edge) -> bool {
    let y = &ctx.opts().config;
    match x.kind {
        EdgeKind::Generic => ctx.resolve_item(x.to).is_enabled_for_codegen(ctx),
        EdgeKind::TemplParamDef
        | EdgeKind::TemplArg
        | EdgeKind::TemplDecl
        | EdgeKind::BaseMember
        | EdgeKind::Field
        | EdgeKind::InnerType
        | EdgeKind::FnReturn
        | EdgeKind::FnParameter
        | EdgeKind::VarType
        | EdgeKind::TypeRef => y.types(),
        EdgeKind::InnerVar => y.vars(),
        EdgeKind::Method => y.methods(),
        EdgeKind::Constructor => y.constructors(),
        EdgeKind::Destructor => y.destructors(),
    }
}

pub trait Tracer {
    fn visit_kind(&mut self, id: ItemId, kind: EdgeKind);
    fn visit(&mut self, id: ItemId) {
        self.visit_kind(id, EdgeKind::Generic);
    }
}
impl<F> Tracer for F
where
    F: FnMut(ItemId, EdgeKind),
{
    fn visit_kind(&mut self, id: ItemId, kind: EdgeKind) {
        (*self)(id, kind)
    }
}

pub trait Trace {
    type Extra;
    fn trace<T>(&self, ctx: &Context, tracer: &mut T, x: &Self::Extra)
    where
        T: Tracer;
}
impl<Id> Trace for Id
where
    Id: Copy + Into<ItemId>,
{
    type Extra = ();
    fn trace<T>(&self, ctx: &Context, tracer: &mut T, x: &Self::Extra)
    where
        T: Tracer,
    {
        ctx.resolve_item(*self).trace(ctx, tracer, x);
    }
}
