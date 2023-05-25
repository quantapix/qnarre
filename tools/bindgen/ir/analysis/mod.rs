mod template_params;
pub(crate) use self::template_params::UsedTemplateParameters;
mod derive;
pub use self::derive::DeriveTrait;
pub(crate) use self::derive::{as_cannot_derive_set, CannotDerive};
mod has_vtable;
pub(crate) use self::has_vtable::{HasVtable, HasVtableAnalysis, HasVtableResult};
mod has_destructor;
pub(crate) use self::has_destructor::HasDestructorAnalysis;
mod has_type_param_in_array;
pub(crate) use self::has_type_param_in_array::HasTypeParameterInArray;
mod has_float;
pub(crate) use self::has_float::HasFloat;
mod sizedness;
pub(crate) use self::sizedness::{Sizedness, SizednessAnalysis, SizednessResult};

use crate::ir::context::{BindgenContext, ItemId};

use crate::ir::traversal::{EdgeKind, Trace};
use crate::HashMap;
use std::fmt;
use std::ops;

pub(crate) trait MonotoneFramework: Sized + fmt::Debug {
    type Node: Copy;

    type Extra: Sized;

    type Output: From<Self> + fmt::Debug;

    fn new(extra: Self::Extra) -> Self;

    fn initial_worklist(&self) -> Vec<Self::Node>;

    fn constrain(&mut self, node: Self::Node) -> ConstrainResult;

    fn each_depending_on<F>(&self, node: Self::Node, f: F)
    where
        F: FnMut(Self::Node);
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum ConstrainResult {
    Changed,

    Same,
}

impl Default for ConstrainResult {
    fn default() -> Self {
        ConstrainResult::Same
    }
}

impl ops::BitOr for ConstrainResult {
    type Output = Self;

    fn bitor(self, rhs: ConstrainResult) -> Self::Output {
        if self == ConstrainResult::Changed || rhs == ConstrainResult::Changed {
            ConstrainResult::Changed
        } else {
            ConstrainResult::Same
        }
    }
}

impl ops::BitOrAssign for ConstrainResult {
    fn bitor_assign(&mut self, rhs: ConstrainResult) {
        *self = *self | rhs;
    }
}

pub(crate) fn analyze<Analysis>(extra: Analysis::Extra) -> Analysis::Output
where
    Analysis: MonotoneFramework,
{
    let mut analysis = Analysis::new(extra);
    let mut worklist = analysis.initial_worklist();

    while let Some(node) = worklist.pop() {
        if let ConstrainResult::Changed = analysis.constrain(node) {
            analysis.each_depending_on(node, |needs_work| {
                worklist.push(needs_work);
            });
        }
    }

    analysis.into()
}

pub(crate) fn generate_dependencies<F>(ctx: &BindgenContext, consider_edge: F) -> HashMap<ItemId, Vec<ItemId>>
where
    F: Fn(EdgeKind) -> bool,
{
    let mut dependencies = HashMap::default();

    for &item in ctx.allowlisted_items() {
        dependencies.entry(item).or_insert_with(Vec::new);

        {
            item.trace(
                ctx,
                &mut |sub_item: ItemId, edge_kind| {
                    if ctx.allowlisted_items().contains(&sub_item) && consider_edge(edge_kind) {
                        dependencies.entry(sub_item).or_insert_with(Vec::new).push(item);
                    }
                },
                &(),
            );
        }
    }
    dependencies
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
            let mut g = Graph::default();
            g.0.insert(Node(1), vec![Node(3)]);
            g.0.insert(Node(2), vec![Node(2)]);
            g.0.insert(Node(3), vec![Node(4), Node(5)]);
            g.0.insert(Node(4), vec![Node(7)]);
            g.0.insert(Node(5), vec![Node(6), Node(7)]);
            g.0.insert(Node(6), vec![Node(8)]);
            g.0.insert(Node(7), vec![Node(3)]);
            g.0.insert(Node(8), vec![]);
            g
        }

        fn reverse(&self) -> Graph {
            let mut reversed = Graph::default();
            for (node, edges) in self.0.iter() {
                reversed.0.entry(*node).or_insert_with(Vec::new);
                for referent in edges.iter() {
                    reversed.0.entry(*referent).or_insert_with(Vec::new).push(*node);
                }
            }
            reversed
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ReachableFrom<'a> {
        reachable: HashMap<Node, HashSet<Node>>,
        graph: &'a Graph,
        reversed: Graph,
    }

    impl<'a> MonotoneFramework for ReachableFrom<'a> {
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

        fn constrain(&mut self, node: Node) -> ConstrainResult {
            let original_size = self.reachable.entry(node).or_insert_with(HashSet::default).len();

            for sub_node in self.graph.0[&node].iter() {
                self.reachable.get_mut(&node).unwrap().insert(*sub_node);

                let sub_reachable = self.reachable.entry(*sub_node).or_insert_with(HashSet::default).clone();

                for transitive in sub_reachable {
                    self.reachable.get_mut(&node).unwrap().insert(transitive);
                }
            }

            let new_size = self.reachable[&node].len();
            if original_size != new_size {
                ConstrainResult::Changed
            } else {
                ConstrainResult::Same
            }
        }

        fn each_depending_on<F>(&self, node: Node, mut f: F)
        where
            F: FnMut(Node),
        {
            for dep in self.reversed.0[&node].iter() {
                f(*dep);
            }
        }
    }

    impl<'a> From<ReachableFrom<'a>> for HashMap<Node, HashSet<Node>> {
        fn from(reachable: ReachableFrom<'a>) -> Self {
            reachable.reachable
        }
    }

    #[test]
    fn monotone() {
        let g = Graph::make_test_graph();
        let reachable = analyze::<ReachableFrom>(&g);
        println!("reachable = {:#?}", reachable);

        fn nodes<A>(nodes: A) -> HashSet<Node>
        where
            A: AsRef<[usize]>,
        {
            nodes.as_ref().iter().cloned().map(Node).collect()
        }

        let mut expected = HashMap::default();
        expected.insert(Node(1), nodes([3, 4, 5, 6, 7, 8]));
        expected.insert(Node(2), nodes([2]));
        expected.insert(Node(3), nodes([3, 4, 5, 6, 7, 8]));
        expected.insert(Node(4), nodes([3, 4, 5, 6, 7, 8]));
        expected.insert(Node(5), nodes([3, 4, 5, 6, 7, 8]));
        expected.insert(Node(6), nodes([8]));
        expected.insert(Node(7), nodes([3, 4, 5, 6, 7, 8]));
        expected.insert(Node(8), nodes([]));
        println!("expected = {:#?}", expected);

        assert_eq!(reachable, expected);
    }
}
