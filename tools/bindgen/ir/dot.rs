use super::context::{BindgenContext, ItemId};
use super::traversal::Trace;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

pub(crate) trait DotAttrs {
    fn dot_attrs<W>(&self, ctx: &BindgenContext, out: &mut W) -> io::Result<()>
    where
        W: io::Write;
}

pub(crate) fn write_dot_file<P>(ctx: &BindgenContext, path: P) -> io::Result<()>
where
    P: AsRef<Path>,
{
    let mut y = io::BufWriter::new(File::create(path)?);
    writeln!(&mut y, "digraph {{")?;
    let mut err: Option<io::Result<_>> = None;
    for (id, it) in ctx.items() {
        let is_allowed = ctx.allowed_items().contains(&id);
        writeln!(
            &mut y,
            r#"{} [fontname="courier", color={}, label=< <table border="0" align="left">"#,
            id.as_usize(),
            if is_allowed { "black" } else { "gray" }
        )?;
        it.dot_attrs(ctx, &mut y)?;
        writeln!(&mut y, r#"</table> >];"#)?;
        it.trace(
            ctx,
            &mut |id2: ItemId, kind| {
                if err.is_some() {
                    return;
                }
                match writeln!(
                    &mut y,
                    "{} -> {} [label={:?}, color={}];",
                    id.as_usize(),
                    id2.as_usize(),
                    kind,
                    if is_allowed { "black" } else { "gray" }
                ) {
                    Ok(_) => {},
                    Err(x) => err = Some(Err(x)),
                }
            },
            &(),
        );
        if let Some(x) = err {
            return x;
        }
        if let Some(x) = it.as_module() {
            for x in x.children() {
                writeln!(
                    &mut y,
                    "{} -> {} [style=dotted, color=gray]",
                    it.id().as_usize(),
                    x.as_usize()
                )?;
            }
        }
    }
    writeln!(&mut y, "}}")?;
    Ok(())
}
