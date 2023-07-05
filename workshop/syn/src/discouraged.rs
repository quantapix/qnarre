use super::*;
use proc_macro2::extra::DelimSpan;
pub trait Speculative {
    fn advance_to(&self, fork: &Self);
}
impl<'a> Speculative for ParseBuffer<'a> {
    fn advance_to(&self, fork: &Self) {
        if !crate::buffer::same_scope(self.cursor(), fork.cursor()) {
            panic!("Fork was not derived from the advancing parse stream");
        }
        let (self_unexp, self_sp) = inner_unexpected(self);
        let (fork_unexp, fork_sp) = inner_unexpected(fork);
        if !Rc::ptr_eq(&self_unexp, &fork_unexp) {
            match (fork_sp, self_sp) {
                (Some(span), None) => {
                    self_unexp.set(Unexpected::Some(span));
                },
                (None, None) => {
                    fork_unexp.set(Unexpected::Chain(self_unexp));
                    fork.unexpected.set(Some(Rc::new(Cell::new(Unexpected::None))));
                },
                (_, Some(_)) => {},
            }
        }
        self.cell
            .set(unsafe { mem::transmute::<Cursor, Cursor<'static>>(fork.cursor()) });
    }
}
pub trait AnyDelimiter {
    fn parse_any_delimiter(&self) -> Result<(Delimiter, DelimSpan, ParseBuffer)>;
}
impl<'a> AnyDelimiter for ParseBuffer<'a> {
    fn parse_any_delimiter(&self) -> Result<(Delimiter, DelimSpan, ParseBuffer)> {
        self.step(|cursor| {
            if let Some((content, delimiter, span, rest)) = cursor.any_group() {
                let scope = crate::buffer::close_span_of_group(*cursor);
                let nested = crate::parse::advance_step_cursor(cursor, content);
                let unexpected = crate::parse::get_unexpected(self);
                let content = crate::parse::new_parse_buffer(scope, nested, unexpected);
                Ok(((delimiter, span, content), rest))
            } else {
                Err(cursor.error("expected any delimiter"))
            }
        })
    }
}
