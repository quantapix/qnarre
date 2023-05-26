#[derive(Debug, PartialEq, Eq)]
enum Kind {
    SingleLines,
    MultiLine,
}

pub(crate) fn preproc(x: &str) -> String {
    match self::kind(x) {
        Some(Kind::SingleLines) => preprocess_single_lines(x),
        Some(Kind::MultiLine) => preprocess_multi_line(x),
        None => x.to_owned(),
    }
}

fn kind(x: &str) -> Option<Kind> {
    if x.starts_with("/*") {
        Some(Kind::MultiLine)
    } else if x.starts_with("//") {
        Some(Kind::SingleLines)
    } else {
        None
    }
}

fn preprocess_single_lines(x: &str) -> String {
    debug_assert!(x.starts_with("//"), "comment is not single line");
    let ys: Vec<_> = x.lines().map(|l| l.trim().trim_start_matches('/')).collect();
    ys.join("\n")
}

fn preprocess_multi_line(x: &str) -> String {
    let x = x.trim_start_matches('/').trim_end_matches('/').trim_end_matches('*');
    let mut ys: Vec<_> = x
        .lines()
        .map(|x| x.trim().trim_start_matches('*').trim_start_matches('!'))
        .skip_while(|x| x.trim().is_empty())
        .collect();
    if ys.last().map_or(false, |x| x.trim().is_empty()) {
        ys.pop();
    }
    ys.join("\n")
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn picks_up_single_and_multi_line_doc_comments() {
        assert_eq!(kind("/// hello"), Some(Kind::SingleLines));
        assert_eq!(kind("/** world */"), Some(Kind::MultiLine));
    }

    #[test]
    fn processes_single_lines_correctly() {
        assert_eq!(preproc("///"), "");
        assert_eq!(preproc("/// hello"), " hello");
        assert_eq!(preproc("// hello"), " hello");
        assert_eq!(preproc("//    hello"), "    hello");
    }

    #[test]
    fn processes_multi_lines_correctly() {
        assert_eq!(preproc("/**/"), "");
        assert_eq!(preproc("/** hello \n * world \n * foo \n */"), " hello\n world\n foo");
        assert_eq!(preproc("/**\nhello\n*world\n*foo\n*/"), "hello\nworld\nfoo");
    }
}
