use drop_bomb::DropBomb;
use limit::Limit;
use std::cell::Cell;

use crate::{
    event::Event,
    input::Input,
    SyntaxKind::{self, EOF, ERROR, TOMBSTONE},
    TokenSet, T,
};

pub struct Parser<'t> {
    inp: &'t Input,
    pos: usize,
    events: Vec<Event>,
    steps: Cell<u32>,
}

static PARSER_STEP_LIMIT: Limit = Limit::new(15_000_000);

impl<'t> Parser<'t> {
    pub fn new(inp: &'t Input) -> Parser<'t> {
        Parser {
            inp,
            pos: 0,
            events: Vec::new(),
            steps: Cell::new(0),
        }
    }

    pub fn finish(self) -> Vec<Event> {
        self.events
    }

    pub fn current(&self) -> SyntaxKind {
        self.nth(0)
    }

    pub fn nth(&self, n: usize) -> SyntaxKind {
        assert!(n <= 3);

        let steps = self.steps.get();
        assert!(
            PARSER_STEP_LIMIT.check(steps as usize).is_ok(),
            "the parser seems stuck"
        );
        self.steps.set(steps + 1);

        self.inp.kind(self.pos + n)
    }

    pub fn at(&self, kind: SyntaxKind) -> bool {
        self.nth_at(0, kind)
    }

    pub fn nth_at(&self, n: usize, kind: SyntaxKind) -> bool {
        match kind {
            T![-=] => self.at_composite2(n, T![-], T![=]),
            T![->] => self.at_composite2(n, T![-], T![>]),
            T![::] => self.at_composite2(n, T![:], T![:]),
            T![!=] => self.at_composite2(n, T![!], T![=]),
            T![..] => self.at_composite2(n, T![.], T![.]),
            T![*=] => self.at_composite2(n, T![*], T![=]),
            T![/=] => self.at_composite2(n, T![/], T![=]),
            T![&&] => self.at_composite2(n, T![&], T![&]),
            T![&=] => self.at_composite2(n, T![&], T![=]),
            T![%=] => self.at_composite2(n, T![%], T![=]),
            T![^=] => self.at_composite2(n, T![^], T![=]),
            T![+=] => self.at_composite2(n, T![+], T![=]),
            T![<<] => self.at_composite2(n, T![<], T![<]),
            T![<=] => self.at_composite2(n, T![<], T![=]),
            T![==] => self.at_composite2(n, T![=], T![=]),
            T![=>] => self.at_composite2(n, T![=], T![>]),
            T![>=] => self.at_composite2(n, T![>], T![=]),
            T![>>] => self.at_composite2(n, T![>], T![>]),
            T![|=] => self.at_composite2(n, T![|], T![=]),
            T![||] => self.at_composite2(n, T![|], T![|]),

            T![...] => self.at_composite3(n, T![.], T![.], T![.]),
            T![..=] => self.at_composite3(n, T![.], T![.], T![=]),
            T![<<=] => self.at_composite3(n, T![<], T![<], T![=]),
            T![>>=] => self.at_composite3(n, T![>], T![>], T![=]),

            _ => self.inp.kind(self.pos + n) == kind,
        }
    }

    pub fn eat(&mut self, kind: SyntaxKind) -> bool {
        if !self.at(kind) {
            return false;
        }
        let n_raw_tokens = match kind {
            T![-=]
            | T![->]
            | T![::]
            | T![!=]
            | T![..]
            | T![*=]
            | T![/=]
            | T![&&]
            | T![&=]
            | T![%=]
            | T![^=]
            | T![+=]
            | T![<<]
            | T![<=]
            | T![==]
            | T![=>]
            | T![>=]
            | T![>>]
            | T![|=]
            | T![||] => 2,

            T![...] | T![..=] | T![<<=] | T![>>=] => 3,
            _ => 1,
        };
        self.do_bump(kind, n_raw_tokens);
        true
    }

    fn at_composite2(&self, n: usize, k1: SyntaxKind, k2: SyntaxKind) -> bool {
        self.inp.kind(self.pos + n) == k1 && self.inp.kind(self.pos + n + 1) == k2 && self.inp.is_joint(self.pos + n)
    }

    fn at_composite3(&self, n: usize, k1: SyntaxKind, k2: SyntaxKind, k3: SyntaxKind) -> bool {
        self.inp.kind(self.pos + n) == k1
            && self.inp.kind(self.pos + n + 1) == k2
            && self.inp.kind(self.pos + n + 2) == k3
            && self.inp.is_joint(self.pos + n)
            && self.inp.is_joint(self.pos + n + 1)
    }

    pub fn at_ts(&self, kinds: TokenSet) -> bool {
        kinds.contains(self.current())
    }

    pub fn at_contextual_kw(&self, kw: SyntaxKind) -> bool {
        self.inp.contextual_kind(self.pos) == kw
    }

    pub fn nth_at_contextual_kw(&self, n: usize, kw: SyntaxKind) -> bool {
        self.inp.contextual_kind(self.pos + n) == kw
    }

    pub fn start(&mut self) -> Marker {
        let pos = self.events.len() as u32;
        self.push_event(Event::tombstone());
        Marker::new(pos)
    }

    pub fn bump(&mut self, kind: SyntaxKind) {
        assert!(self.eat(kind));
    }

    pub fn bump_any(&mut self) {
        let kind = self.nth(0);
        if kind == EOF {
            return;
        }
        self.do_bump(kind, 1);
    }

    pub fn split_float(&mut self, mut marker: Marker) -> (bool, Marker) {
        assert!(self.at(SyntaxKind::FLOAT_NUMBER));
        let ends_in_dot = !self.inp.is_joint(self.pos);
        if !ends_in_dot {
            let new_marker = self.start();
            let idx = marker.pos as usize;
            match &mut self.events[idx] {
                Event::Start { forward_parent, kind } => {
                    *kind = SyntaxKind::FIELD_EXPR;
                    *forward_parent = Some(new_marker.pos - marker.pos);
                },
                _ => unreachable!(),
            }
            marker.bomb.defuse();
            marker = new_marker;
        };
        self.pos += 1;
        self.push_event(Event::FloatSplitHack { ends_in_dot });
        (ends_in_dot, marker)
    }

    pub fn bump_remap(&mut self, kind: SyntaxKind) {
        if self.nth(0) == EOF {
            return;
        }
        self.do_bump(kind, 1);
    }

    pub fn error<T: Into<String>>(&mut self, message: T) {
        let msg = message.into();
        self.push_event(Event::Error { msg });
    }

    pub fn expect(&mut self, kind: SyntaxKind) -> bool {
        if self.eat(kind) {
            return true;
        }
        self.error(format!("expected {kind:?}"));
        false
    }

    pub fn err_and_bump(&mut self, message: &str) {
        self.err_recover(message, TokenSet::EMPTY);
    }

    pub fn err_recover(&mut self, message: &str, recovery: TokenSet) {
        match self.current() {
            T!['{'] | T!['}'] => {
                self.error(message);
                return;
            },
            _ => (),
        }

        if self.at_ts(recovery) {
            self.error(message);
            return;
        }

        let m = self.start();
        self.error(message);
        self.bump_any();
        m.complete(self, ERROR);
    }

    fn do_bump(&mut self, kind: SyntaxKind, n_raw_tokens: u8) {
        self.pos += n_raw_tokens as usize;
        self.steps.set(0);
        self.push_event(Event::Token { kind, n_raw_tokens });
    }

    fn push_event(&mut self, event: Event) {
        self.events.push(event);
    }
}

pub struct Marker {
    pos: u32,
    bomb: DropBomb,
}

impl Marker {
    fn new(pos: u32) -> Marker {
        Marker {
            pos,
            bomb: DropBomb::new("Marker must be either completed or abandoned"),
        }
    }

    pub fn complete(mut self, p: &mut Parser<'_>, kind: SyntaxKind) -> CompletedMarker {
        self.bomb.defuse();
        let idx = self.pos as usize;
        match &mut p.events[idx] {
            Event::Start { kind: slot, .. } => {
                *slot = kind;
            },
            _ => unreachable!(),
        }
        p.push_event(Event::Finish);
        CompletedMarker::new(self.pos, kind)
    }

    pub fn abandon(mut self, p: &mut Parser<'_>) {
        self.bomb.defuse();
        let idx = self.pos as usize;
        if idx == p.events.len() - 1 {
            match p.events.pop() {
                Some(Event::Start {
                    kind: TOMBSTONE,
                    forward_parent: None,
                }) => (),
                _ => unreachable!(),
            }
        }
    }
}

pub struct CompletedMarker {
    pos: u32,
    kind: SyntaxKind,
}

impl CompletedMarker {
    fn new(pos: u32, kind: SyntaxKind) -> Self {
        CompletedMarker { pos, kind }
    }

    pub fn precede(self, p: &mut Parser<'_>) -> Marker {
        let new_pos = p.start();
        let idx = self.pos as usize;
        match &mut p.events[idx] {
            Event::Start { forward_parent, .. } => {
                *forward_parent = Some(new_pos.pos - self.pos);
            },
            _ => unreachable!(),
        }
        new_pos
    }

    pub fn extend_to(self, p: &mut Parser<'_>, mut m: Marker) -> CompletedMarker {
        m.bomb.defuse();
        let idx = m.pos as usize;
        match &mut p.events[idx] {
            Event::Start { forward_parent, .. } => {
                *forward_parent = Some(self.pos - m.pos);
            },
            _ => unreachable!(),
        }
        self
    }

    pub fn kind(&self) -> SyntaxKind {
        self.kind
    }
}
