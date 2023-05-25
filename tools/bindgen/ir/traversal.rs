use super::context::{BindgenContext, ItemId};
use super::item::ItemSet;
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Edge {
    to: ItemId,
    kind: EdgeKind,
}

impl Edge {
    pub(crate) fn new(to: ItemId, kind: EdgeKind) -> Edge {
        Edge { to, kind }
    }
}

impl From<Edge> for ItemId {
    fn from(val: Edge) -> Self {
        val.to
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum EdgeKind {
    Generic,

    TemplateParameterDefinition,

    TemplateDeclaration,

    TemplateArgument,

    BaseMember,

    Field,

    InnerType,

    InnerVar,

    Method,

    Constructor,

    Destructor,

    FunctionReturn,

    FunctionParameter,

    VarType,

    TypeReference,
}

pub(crate) type TraversalPredicate = for<'a> fn(&'a BindgenContext, Edge) -> bool;

pub(crate) fn all_edges(_: &BindgenContext, _: Edge) -> bool {
    true
}

pub(crate) fn only_inner_type_edges(_: &BindgenContext, edge: Edge) -> bool {
    edge.kind == EdgeKind::InnerType
}

pub(crate) fn codegen_edges(ctx: &BindgenContext, edge: Edge) -> bool {
    let cc = &ctx.options().codegen_config;
    match edge.kind {
        EdgeKind::Generic => ctx.resolve_item(edge.to).is_enabled_for_codegen(ctx),

        EdgeKind::TemplateParameterDefinition
        | EdgeKind::TemplateArgument
        | EdgeKind::TemplateDeclaration
        | EdgeKind::BaseMember
        | EdgeKind::Field
        | EdgeKind::InnerType
        | EdgeKind::FunctionReturn
        | EdgeKind::FunctionParameter
        | EdgeKind::VarType
        | EdgeKind::TypeReference => cc.types(),
        EdgeKind::InnerVar => cc.vars(),
        EdgeKind::Method => cc.methods(),
        EdgeKind::Constructor => cc.constructors(),
        EdgeKind::Destructor => cc.destructors(),
    }
}

pub(crate) trait TraversalStorage<'ctx> {
    fn new(ctx: &'ctx BindgenContext) -> Self;

    fn add(&mut self, from: Option<ItemId>, item: ItemId) -> bool;
}

impl<'ctx> TraversalStorage<'ctx> for ItemSet {
    fn new(_: &'ctx BindgenContext) -> Self {
        ItemSet::new()
    }

    fn add(&mut self, _: Option<ItemId>, item: ItemId) -> bool {
        self.insert(item)
    }
}

#[derive(Debug)]
pub(crate) struct Paths<'ctx>(BTreeMap<ItemId, ItemId>, &'ctx BindgenContext);

impl<'ctx> TraversalStorage<'ctx> for Paths<'ctx> {
    fn new(ctx: &'ctx BindgenContext) -> Self {
        Paths(BTreeMap::new(), ctx)
    }

    fn add(&mut self, from: Option<ItemId>, item: ItemId) -> bool {
        let newly_discovered = self.0.insert(item, from.unwrap_or(item)).is_none();

        if self.1.resolve_item_fallible(item).is_none() {
            let mut path = vec![];
            let mut current = item;
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
            panic!("Found reference to dangling id = {:?}\nvia path = {:?}", item, path);
        }

        newly_discovered
    }
}

pub(crate) trait TraversalQueue: Default {
    fn push(&mut self, item: ItemId);

    fn next(&mut self) -> Option<ItemId>;
}

impl TraversalQueue for Vec<ItemId> {
    fn push(&mut self, item: ItemId) {
        self.push(item);
    }

    fn next(&mut self) -> Option<ItemId> {
        self.pop()
    }
}

impl TraversalQueue for VecDeque<ItemId> {
    fn push(&mut self, item: ItemId) {
        self.push_back(item);
    }

    fn next(&mut self) -> Option<ItemId> {
        self.pop_front()
    }
}

pub(crate) trait Tracer {
    fn visit_kind(&mut self, item: ItemId, kind: EdgeKind);

    fn visit(&mut self, item: ItemId) {
        self.visit_kind(item, EdgeKind::Generic);
    }
}

impl<F> Tracer for F
where
    F: FnMut(ItemId, EdgeKind),
{
    fn visit_kind(&mut self, item: ItemId, kind: EdgeKind) {
        (*self)(item, kind)
    }
}

pub(crate) trait Trace {
    type Extra;

    fn trace<T>(&self, context: &BindgenContext, tracer: &mut T, extra: &Self::Extra)
    where
        T: Tracer;
}

pub(crate) struct ItemTraversal<'ctx, Storage, Queue>
where
    Storage: TraversalStorage<'ctx>,
    Queue: TraversalQueue,
{
    ctx: &'ctx BindgenContext,

    seen: Storage,

    queue: Queue,

    predicate: TraversalPredicate,

    currently_traversing: Option<ItemId>,
}

impl<'ctx, Storage, Queue> ItemTraversal<'ctx, Storage, Queue>
where
    Storage: TraversalStorage<'ctx>,
    Queue: TraversalQueue,
{
    pub(crate) fn new<R>(
        ctx: &'ctx BindgenContext,
        roots: R,
        predicate: TraversalPredicate,
    ) -> ItemTraversal<'ctx, Storage, Queue>
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
            predicate,
            currently_traversing: None,
        }
    }
}

impl<'ctx, Storage, Queue> Tracer for ItemTraversal<'ctx, Storage, Queue>
where
    Storage: TraversalStorage<'ctx>,
    Queue: TraversalQueue,
{
    fn visit_kind(&mut self, item: ItemId, kind: EdgeKind) {
        let edge = Edge::new(item, kind);
        if !(self.predicate)(self.ctx, edge) {
            return;
        }

        let is_newly_discovered = self.seen.add(self.currently_traversing, item);
        if is_newly_discovered {
            self.queue.push(item)
        }
    }
}

impl<'ctx, Storage, Queue> Iterator for ItemTraversal<'ctx, Storage, Queue>
where
    Storage: TraversalStorage<'ctx>,
    Queue: TraversalQueue,
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

        self.currently_traversing = Some(id);
        id.trace(self.ctx, self, &());
        self.currently_traversing = None;

        Some(id)
    }
}

pub(crate) type AssertNoDanglingItemsTraversal<'ctx> = ItemTraversal<'ctx, Paths<'ctx>, VecDeque<ItemId>>;
