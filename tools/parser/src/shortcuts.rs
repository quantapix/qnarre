use std::mem;

use crate::{
    Lexed, Step,
    SyntaxKind::{self, *},
};

#[derive(Debug)]
pub enum StrStep<'a> {
    Token { kind: SyntaxKind, text: &'a str },
    Enter { kind: SyntaxKind },
    Exit,
    Error { msg: &'a str, pos: usize },
}

impl<'a> Lexed<'a> {
    pub fn to_input(&self) -> crate::Input {
        let mut res = crate::Input::default();
        let mut was_joint = false;
        for i in 0..self.len() {
            let kind = self.kind(i);
            if kind.is_trivia() {
                was_joint = false
            } else {
                if kind == SyntaxKind::IDENT {
                    let token_text = self.text(i);
                    let contextual_kw = SyntaxKind::from_contextual_keyword(token_text).unwrap_or(SyntaxKind::IDENT);
                    res.push_ident(contextual_kw);
                } else {
                    if was_joint {
                        res.was_joint();
                    }
                    res.push(kind);
                    if kind == SyntaxKind::FLOAT_NUMBER && !self.text(i).ends_with('.') {
                        res.was_joint();
                    }
                }

                was_joint = true;
            }
        }
        res
    }

    pub fn intersperse_trivia(&self, output: &crate::Output, sink: &mut dyn FnMut(StrStep<'_>)) -> bool {
        let mut builder = Builder {
            lexed: self,
            pos: 0,
            state: State::PendingEnter,
            sink,
        };

        for event in output.iter() {
            match event {
                Step::Token {
                    kind,
                    n_input_toks: n_raw_tokens,
                } => builder.token(kind, n_raw_tokens),
                Step::FloatSplit {
                    ends_in_dot: has_pseudo_dot,
                } => builder.float_split(has_pseudo_dot),
                Step::Enter { kind } => builder.enter(kind),
                Step::Exit => builder.exit(),
                Step::Error { msg } => {
                    let text_pos = builder.lexed.text_start(builder.pos);
                    (builder.sink)(StrStep::Error { msg, pos: text_pos });
                },
            }
        }

        match mem::replace(&mut builder.state, State::Normal) {
            State::PendingExit => {
                builder.eat_trivias();
                (builder.sink)(StrStep::Exit);
            },
            State::PendingEnter | State::Normal => unreachable!(),
        }

        builder.pos == builder.lexed.len()
    }
}

struct Builder<'a, 'b> {
    lexed: &'a Lexed<'a>,
    pos: usize,
    state: State,
    sink: &'b mut dyn FnMut(StrStep<'_>),
}

enum State {
    PendingEnter,
    Normal,
    PendingExit,
}

impl Builder<'_, '_> {
    fn token(&mut self, kind: SyntaxKind, n_tokens: u8) {
        match mem::replace(&mut self.state, State::Normal) {
            State::PendingEnter => unreachable!(),
            State::PendingExit => (self.sink)(StrStep::Exit),
            State::Normal => (),
        }
        self.eat_trivias();
        self.do_token(kind, n_tokens as usize);
    }

    fn float_split(&mut self, has_pseudo_dot: bool) {
        match mem::replace(&mut self.state, State::Normal) {
            State::PendingEnter => unreachable!(),
            State::PendingExit => (self.sink)(StrStep::Exit),
            State::Normal => (),
        }
        self.eat_trivias();
        self.do_float_split(has_pseudo_dot);
    }

    fn enter(&mut self, kind: SyntaxKind) {
        match mem::replace(&mut self.state, State::Normal) {
            State::PendingEnter => {
                (self.sink)(StrStep::Enter { kind });
                return;
            },
            State::PendingExit => (self.sink)(StrStep::Exit),
            State::Normal => (),
        }

        let n_trivias = (self.pos..self.lexed.len())
            .take_while(|&x| self.lexed.kind(x).is_trivia())
            .count();
        let leading_trivias = self.pos..self.pos + n_trivias;
        let n_attached_trivias = n_attached_trivias(
            kind,
            leading_trivias.rev().map(|x| (self.lexed.kind(x), self.lexed.text(x))),
        );
        self.eat_n_trivias(n_trivias - n_attached_trivias);
        (self.sink)(StrStep::Enter { kind });
        self.eat_n_trivias(n_attached_trivias);
    }

    fn exit(&mut self) {
        match mem::replace(&mut self.state, State::PendingExit) {
            State::PendingEnter => unreachable!(),
            State::PendingExit => (self.sink)(StrStep::Exit),
            State::Normal => (),
        }
    }

    fn eat_trivias(&mut self) {
        while self.pos < self.lexed.len() {
            let kind = self.lexed.kind(self.pos);
            if !kind.is_trivia() {
                break;
            }
            self.do_token(kind, 1);
        }
    }

    fn eat_n_trivias(&mut self, n: usize) {
        for _ in 0..n {
            let kind = self.lexed.kind(self.pos);
            assert!(kind.is_trivia());
            self.do_token(kind, 1);
        }
    }

    fn do_token(&mut self, kind: SyntaxKind, n_tokens: usize) {
        let text = &self.lexed.range_text(self.pos..self.pos + n_tokens);
        self.pos += n_tokens;
        (self.sink)(StrStep::Token { kind, text });
    }

    fn do_float_split(&mut self, has_pseudo_dot: bool) {
        let text = &self.lexed.range_text(self.pos..self.pos + 1);
        self.pos += 1;
        match text.split_once('.') {
            Some((left, right)) => {
                assert!(!left.is_empty());
                (self.sink)(StrStep::Enter {
                    kind: SyntaxKind::NAME_REF,
                });
                (self.sink)(StrStep::Token {
                    kind: SyntaxKind::INT_NUMBER,
                    text: left,
                });
                (self.sink)(StrStep::Exit);

                (self.sink)(StrStep::Exit);

                (self.sink)(StrStep::Token {
                    kind: SyntaxKind::DOT,
                    text: ".",
                });

                if has_pseudo_dot {
                    assert!(right.is_empty(), "{left}.{right}");
                    self.state = State::Normal;
                } else {
                    (self.sink)(StrStep::Enter {
                        kind: SyntaxKind::NAME_REF,
                    });
                    (self.sink)(StrStep::Token {
                        kind: SyntaxKind::INT_NUMBER,
                        text: right,
                    });
                    (self.sink)(StrStep::Exit);

                    self.state = State::PendingExit;
                }
            },
            None => unreachable!(),
        }
    }
}

fn n_attached_trivias<'a>(kind: SyntaxKind, trivias: impl Iterator<Item = (SyntaxKind, &'a str)>) -> usize {
    match kind {
        CONST | ENUM | FN | IMPL | MACRO_CALL | MACRO_DEF | MACRO_RULES | MODULE | RECORD_FIELD | STATIC | STRUCT
        | TRAIT | TUPLE_FIELD | TYPE_ALIAS | UNION | USE | VARIANT => {
            let mut res = 0;
            let mut trivias = trivias.enumerate().peekable();

            while let Some((i, (kind, text))) = trivias.next() {
                match kind {
                    WHITESPACE if text.contains("\n\n") => {
                        if let Some((COMMENT, peek_text)) = trivias.peek().map(|(_, pair)| pair) {
                            if is_outer(peek_text) {
                                continue;
                            }
                        }
                        break;
                    },
                    COMMENT => {
                        if is_inner(text) {
                            break;
                        }
                        res = i + 1;
                    },
                    _ => (),
                }
            }
            res
        },
        _ => 0,
    }
}

fn is_outer(text: &str) -> bool {
    if text.starts_with("////") || text.starts_with("/***") {
        return false;
    }
    text.starts_with("///") || text.starts_with("/**")
}

fn is_inner(text: &str) -> bool {
    text.starts_with("//!") || text.starts_with("/*!")
}
