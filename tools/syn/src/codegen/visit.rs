#![allow(unused_variables)]

use crate::*;


pub fn visit_span<'a, V>(v: &mut V, self: &pm2::Span)
where
    V: Visitor + ?Sized,
{
}
}
