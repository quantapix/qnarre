use crate::error::Result;
use crate::parse::ParseBuffer;
use crate::token;
use proc_macro2::extra::DelimSpan;
use proc_macro2::Delimiter;
pub struct Parens<'a> {
    pub token: token::Paren,
    pub content: ParseBuffer<'a>,
}
pub struct Braces<'a> {
    pub token: token::Brace,
    pub content: ParseBuffer<'a>,
}
pub struct Brackets<'a> {
    pub token: token::Bracket,
    pub content: ParseBuffer<'a>,
}

pub struct Group<'a> {
    pub token: token::Group,
    pub content: ParseBuffer<'a>,
}
pub fn parse_parens<'a>(input: &ParseBuffer<'a>) -> Result<Parens<'a>> {
    parse_delimited(input, Delimiter::Parenthesis).map(|(span, content)| Parens {
        token: token::Paren(span),
        content,
    })
}
pub fn parse_braces<'a>(input: &ParseBuffer<'a>) -> Result<Braces<'a>> {
    parse_delimited(input, Delimiter::Brace).map(|(span, content)| Braces {
        token: token::Brace(span),
        content,
    })
}
pub fn parse_brackets<'a>(input: &ParseBuffer<'a>) -> Result<Brackets<'a>> {
    parse_delimited(input, Delimiter::Bracket).map(|(span, content)| Brackets {
        token: token::Bracket(span),
        content,
    })
}

pub(crate) fn parse_group<'a>(input: &ParseBuffer<'a>) -> Result<Group<'a>> {
    parse_delimited(input, Delimiter::None).map(|(span, content)| Group {
        token: token::Group(span.join()),
        content,
    })
}
fn parse_delimited<'a>(input: &ParseBuffer<'a>, delimiter: Delimiter) -> Result<(DelimSpan, ParseBuffer<'a>)> {
    input.step(|cursor| {
        if let Some((content, span, rest)) = cursor.group(delimiter) {
            let scope = crate::buffer::close_span_of_group(*cursor);
            let nested = crate::parse::advance_step_cursor(cursor, content);
            let unexpected = crate::parse::get_unexpected(input);
            let content = crate::parse::new_parse_buffer(scope, nested, unexpected);
            Ok(((span, content), rest))
        } else {
            let message = match delimiter {
                Delimiter::Parenthesis => "expected parentheses",
                Delimiter::Brace => "expected curly braces",
                Delimiter::Bracket => "expected square brackets",
                Delimiter::None => "expected invisible group",
            };
            Err(cursor.error(message))
        }
    })
}
#[macro_export]

macro_rules! parenthesized {
    ($content:ident in $cursor:expr) => {
        match $crate::__private::parse_parens(&$cursor) {
            $crate::__private::Ok(parens) => {
                $content = parens.content;
                parens.token
            },
            $crate::__private::Err(error) => {
                return $crate::__private::Err(error);
            },
        }
    };
}
#[macro_export]

macro_rules! braced {
    ($content:ident in $cursor:expr) => {
        match $crate::__private::parse_braces(&$cursor) {
            $crate::__private::Ok(braces) => {
                $content = braces.content;
                braces.token
            },
            $crate::__private::Err(error) => {
                return $crate::__private::Err(error);
            },
        }
    };
}
#[macro_export]

macro_rules! bracketed {
    ($content:ident in $cursor:expr) => {
        match $crate::__private::parse_brackets(&$cursor) {
            $crate::__private::Ok(brackets) => {
                $content = brackets.content;
                brackets.token
            },
            $crate::__private::Err(error) => {
                return $crate::__private::Err(error);
            },
        }
    };
}
