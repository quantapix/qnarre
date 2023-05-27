mod template_params;
pub(crate) use self::template_params::UsedTemplateParams;
mod derive;
pub use self::derive::DeriveTrait;
pub(crate) use self::derive::{as_cannot_derive_set, CannotDerive};
mod has_vtable;
pub(crate) use self::has_vtable::{HasVtable, HasVtableAnalysis, YHasVtable};
mod has_destructor;
pub(crate) use self::has_destructor::HasDestructorAnalysis;
mod has_type_param_in_array;
pub(crate) use self::has_type_param_in_array::HasTypeParameterInArray;
mod has_float;
pub(crate) use self::has_float::HasFloat;
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
