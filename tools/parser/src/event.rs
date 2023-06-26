use std::mem;

use crate::{
    output::Output,
    SyntaxKind::{self, *},
};

#[derive(Debug)]
pub(crate) enum Event {
    Start {
        kind: SyntaxKind,
        forward_parent: Option<u32>,
    },

    Finish,

    Token {
        kind: SyntaxKind,
        n_raw_tokens: u8,
    },
    FloatSplitHack {
        ends_in_dot: bool,
    },
    Error {
        msg: String,
    },
}

impl Event {
    pub(crate) fn tombstone() -> Self {
        Event::Start {
            kind: TOMBSTONE,
            forward_parent: None,
        }
    }
}

pub(super) fn process(mut events: Vec<Event>) -> Output {
    let mut res = Output::default();
    let mut forward_parents = Vec::new();

    for i in 0..events.len() {
        match mem::replace(&mut events[i], Event::tombstone()) {
            Event::Start { kind, forward_parent } => {
                // For events[A, B, C], B is A's forward_parent, C is B's forward_parent,
                // in the normal control flow, the parent-child relation: `A -> B -> C`,
                // while with the magic forward_parent, it writes: `C <- B <- A`.

                // append `A` into parents.
                forward_parents.push(kind);
                let mut idx = i;
                let mut fp = forward_parent;
                while let Some(fwd) = fp {
                    idx += fwd as usize;
                    // append `A`'s forward_parent `B`
                    fp = match mem::replace(&mut events[idx], Event::tombstone()) {
                        Event::Start { kind, forward_parent } => {
                            forward_parents.push(kind);
                            forward_parent
                        },
                        _ => unreachable!(),
                    };
                    // append `B`'s forward_parent `C` in the next stage.
                }

                for kind in forward_parents.drain(..).rev() {
                    if kind != TOMBSTONE {
                        res.enter_node(kind);
                    }
                }
            },
            Event::Finish => res.leave_node(),
            Event::Token { kind, n_raw_tokens } => {
                res.token(kind, n_raw_tokens);
            },
            Event::FloatSplitHack { ends_in_dot } => {
                res.float_split_hack(ends_in_dot);
                let ev = mem::replace(&mut events[i + 1], Event::tombstone());
                assert!(matches!(ev, Event::Finish), "{ev:?}");
            },
            Event::Error { msg } => res.error(msg),
        }
    }

    res
}
