use crate::SyntaxKind;

#[derive(Default)]
pub struct Output {
    event: Vec<u32>,
    error: Vec<String>,
}

#[derive(Debug)]
pub enum Step<'a> {
    Token { kind: SyntaxKind, n_input_tokens: u8 },
    FloatSplit { ends_in_dot: bool },
    Enter { kind: SyntaxKind },
    Exit,
    Error { msg: &'a str },
}

impl Output {
    const EVENT_MASK: u32 = 0b1;
    const TAG_MASK: u32 = 0x0000_00F0;
    const N_INPUT_TOKEN_MASK: u32 = 0x0000_FF00;
    const KIND_MASK: u32 = 0xFFFF_0000;

    const ERROR_SHIFT: u32 = Self::EVENT_MASK.trailing_ones();
    const TAG_SHIFT: u32 = Self::TAG_MASK.trailing_zeros();
    const N_INPUT_TOKEN_SHIFT: u32 = Self::N_INPUT_TOKEN_MASK.trailing_zeros();
    const KIND_SHIFT: u32 = Self::KIND_MASK.trailing_zeros();

    const TOKEN_EVENT: u8 = 0;
    const ENTER_EVENT: u8 = 1;
    const EXIT_EVENT: u8 = 2;
    const SPLIT_EVENT: u8 = 3;

    pub fn iter(&self) -> impl Iterator<Item = Step<'_>> {
        self.event.iter().map(|&event| {
            if event & Self::EVENT_MASK == 0 {
                return Step::Error {
                    msg: self.error[(event as usize) >> Self::ERROR_SHIFT].as_str(),
                };
            }
            let tag = ((event & Self::TAG_MASK) >> Self::TAG_SHIFT) as u8;
            match tag {
                Self::TOKEN_EVENT => {
                    let kind: SyntaxKind = (((event & Self::KIND_MASK) >> Self::KIND_SHIFT) as u16).into();
                    let n_input_tokens = ((event & Self::N_INPUT_TOKEN_MASK) >> Self::N_INPUT_TOKEN_SHIFT) as u8;
                    Step::Token { kind, n_input_tokens }
                },
                Self::ENTER_EVENT => {
                    let kind: SyntaxKind = (((event & Self::KIND_MASK) >> Self::KIND_SHIFT) as u16).into();
                    Step::Enter { kind }
                },
                Self::EXIT_EVENT => Step::Exit,
                Self::SPLIT_EVENT => Step::FloatSplit {
                    ends_in_dot: event & Self::N_INPUT_TOKEN_MASK != 0,
                },
                _ => unreachable!(),
            }
        })
    }

    pub(crate) fn token(&mut self, kind: SyntaxKind, n_tokens: u8) {
        let e = ((kind as u16 as u32) << Self::KIND_SHIFT)
            | ((n_tokens as u32) << Self::N_INPUT_TOKEN_SHIFT)
            | Self::EVENT_MASK;
        self.event.push(e)
    }

    pub(crate) fn float_split_hack(&mut self, ends_in_dot: bool) {
        let e = (Self::SPLIT_EVENT as u32) << Self::TAG_SHIFT
            | ((ends_in_dot as u32) << Self::N_INPUT_TOKEN_SHIFT)
            | Self::EVENT_MASK;
        self.event.push(e);
    }

    pub(crate) fn enter_node(&mut self, kind: SyntaxKind) {
        let e = ((kind as u16 as u32) << Self::KIND_SHIFT)
            | ((Self::ENTER_EVENT as u32) << Self::TAG_SHIFT)
            | Self::EVENT_MASK;
        self.event.push(e)
    }

    pub(crate) fn leave_node(&mut self) {
        let e = (Self::EXIT_EVENT as u32) << Self::TAG_SHIFT | Self::EVENT_MASK;
        self.event.push(e)
    }

    pub(crate) fn error(&mut self, error: String) {
        let idx = self.error.len();
        self.error.push(error);
        let e = (idx as u32) << Self::ERROR_SHIFT;
        self.event.push(e);
    }
}
