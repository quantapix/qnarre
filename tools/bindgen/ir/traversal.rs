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

pub type TraversPredicate = for<'a> fn(&'a Context, Edge) -> bool;

pub trait TraversStorage<'ctx> {
    fn new(ctx: &'ctx Context) -> Self;
    fn add(&mut self, from: Option<ItemId>, id: ItemId) -> bool;
}
impl<'ctx> TraversStorage<'ctx> for ItemSet {
    fn new(_: &'ctx Context) -> Self {
        ItemSet::new()
    }
    fn add(&mut self, _: Option<ItemId>, id: ItemId) -> bool {
        self.insert(id)
    }
}
#[derive(Debug)]

pub struct Paths<'ctx>(BTreeMap<ItemId, ItemId>, &'ctx Context);
impl<'ctx> TraversStorage<'ctx> for Paths<'ctx> {
    fn new(ctx: &'ctx Context) -> Self {
        Paths(BTreeMap::new(), ctx)
    }
    fn add(&mut self, from: Option<ItemId>, id: ItemId) -> bool {
        let y = self.0.insert(id, from.unwrap_or(id)).is_none();
        if self.1.resolve_item_fallible(id).is_none() {
            let mut path = vec![];
            let mut current = id;
            loop {
                let predecessor = *self.0.get(&current).expect(
                    "We know we found this item id, so it must have a \
                     predecessor",
                );
                if predecessor == current {
                    break;
                }
                path.push(predecessor);
                current = predecessor;
            }
            path.reverse();
            panic!("Found reference to dangling id = {:?}\nvia path = {:?}", id, path);
        }
        y
    }
}

pub trait TraversQueue: Default {
    fn push(&mut self, id: ItemId);
    fn next(&mut self) -> Option<ItemId>;
}
impl TraversQueue for Vec<ItemId> {
    fn push(&mut self, id: ItemId) {
        self.push(id);
    }
    fn next(&mut self) -> Option<ItemId> {
        self.pop()
    }
}
impl TraversQueue for VecDeque<ItemId> {
    fn push(&mut self, id: ItemId) {
        self.push_back(id);
    }
    fn next(&mut self) -> Option<ItemId> {
        self.pop_front()
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
    fn trace<T>(&self, ctx: &Context, tracer: &mut T, extra: &Self::Extra)
    where
        T: Tracer;
}
impl<Id> Trace for Id
where
    Id: Copy + Into<ItemId>,
{
    type Extra = ();
    fn trace<T>(&self, ctx: &Context, tracer: &mut T, extra: &())
    where
        T: Tracer,
    {
        ctx.resolve_item(*self).trace(ctx, tracer, extra);
    }
}

pub struct ItemTraversal<'ctx, Storage, Queue>
where
    Storage: TraversStorage<'ctx>,
    Queue: TraversQueue,
{
    ctx: &'ctx Context,
    seen: Storage,
    queue: Queue,
    pred: TraversPredicate,
    current: Option<ItemId>,
}
impl<'ctx, Storage, Queue> ItemTraversal<'ctx, Storage, Queue>
where
    Storage: TraversStorage<'ctx>,
    Queue: TraversQueue,
{
    pub fn new<R>(ctx: &'ctx Context, roots: R, pred: TraversPredicate) -> ItemTraversal<'ctx, Storage, Queue>
    where
        R: IntoIterator<Item = ItemId>,
    {
        let mut seen = Storage::new(ctx);
        let mut queue = Queue::default();
        for id in roots {
            seen.add(None, id);
            queue.push(id);
        }
        ItemTraversal {
            ctx,
            seen,
            queue,
            pred,
            current: None,
        }
    }
}
impl<'ctx, Storage, Queue> Tracer for ItemTraversal<'ctx, Storage, Queue>
where
    Storage: TraversStorage<'ctx>,
    Queue: TraversQueue,
{
    fn visit_kind(&mut self, id: ItemId, kind: EdgeKind) {
        let edge = Edge::new(id, kind);
        if !(self.pred)(self.ctx, edge) {
            return;
        }
        let is_newly_discovered = self.seen.add(self.current, id);
        if is_newly_discovered {
            self.queue.push(id)
        }
    }
}
impl<'ctx, Storage, Queue> Iterator for ItemTraversal<'ctx, Storage, Queue>
where
    Storage: TraversStorage<'ctx>,
    Queue: TraversQueue,
{
    type Item = ItemId;
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.queue.next()?;
        let newly_discovered = self.seen.add(None, id);
        debug_assert!(
            !newly_discovered,
            "should have already seen anything we get out of our queue"
        );
        debug_assert!(
            self.ctx.resolve_item_fallible(id).is_some(),
            "should only get IDs of actual items in our context during traversal"
        );
        self.current = Some(id);
        id.trace(self.ctx, self, &());
        self.current = None;
        Some(id)
    }
}
pub type AssertNoDanglingItemsTraversal<'ctx> = ItemTraversal<'ctx, Paths<'ctx>, VecDeque<ItemId>>;

pub fn all_edges(_: &Context, _: Edge) -> bool {
    true
}

pub fn only_inner_type_edges(_: &Context, edge: Edge) -> bool {
    edge.kind == EdgeKind::InnerType
}

pub fn codegen_edges(ctx: &Context, edge: Edge) -> bool {
    let cc = &ctx.opts().codegen_config;
    match edge.kind {
        EdgeKind::Generic => ctx.resolve_item(edge.to).is_enabled_for_codegen(ctx),
        EdgeKind::TemplParamDef
        | EdgeKind::TemplArg
        | EdgeKind::TemplDecl
        | EdgeKind::BaseMember
        | EdgeKind::Field
        | EdgeKind::InnerType
        | EdgeKind::FnReturn
        | EdgeKind::FnParameter
        | EdgeKind::VarType
        | EdgeKind::TypeRef => cc.types(),
        EdgeKind::InnerVar => cc.vars(),
        EdgeKind::Method => cc.methods(),
        EdgeKind::Constructor => cc.constructors(),
        EdgeKind::Destructor => cc.destructors(),
    }
}
