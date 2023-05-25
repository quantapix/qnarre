#[derive(Debug, PartialEq, Eq)]
enum Kind {
    SingleLines,
    MultiLine,
}

pub(crate) fn preprocess(comment: &str) -> String {
    match self::kind(comment) {
        Some(Kind::SingleLines) => preprocess_single_lines(comment),
        Some(Kind::MultiLine) => preprocess_multi_line(comment),
        None => comment.to_owned(),
    }
}

fn kind(comment: &str) -> Option<Kind> {
    if comment.starts_with("/*") {
        Some(Kind::MultiLine)
    } else if comment.starts_with("//") {
        Some(Kind::SingleLines)
    } else {
        None
    }
}

fn preprocess_single_lines(comment: &str) -> String {
    debug_assert!(comment.starts_with("//"), "comment is not single line");

    let lines: Vec<_> = comment.lines().map(|l| l.trim().trim_start_matches('/')).collect();
    lines.join("\n")
}

fn preprocess_multi_line(comment: &str) -> String {
    let comment = comment
        .trim_start_matches('/')
        .trim_end_matches('/')
        .trim_end_matches('*');

    let mut lines: Vec<_> = comment
        .lines()
        .map(|line| line.trim().trim_start_matches('*').trim_start_matches('!'))
        .skip_while(|line| line.trim().is_empty()) // Skip the first empty lines.
        .collect();

    if lines.last().map_or(false, |l| l.trim().is_empty()) {
        lines.pop();
    }

    lines.join("\n")
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
        assert_eq!(preprocess("///"), "");
        assert_eq!(preprocess("/// hello"), " hello");
        assert_eq!(preprocess("// hello"), " hello");
        assert_eq!(preprocess("//    hello"), "    hello");
    }

    #[test]
    fn processes_multi_lines_correctly() {
        assert_eq!(preprocess("/**/"), "");

        assert_eq!(
            preprocess("/** hello \n * world \n * foo \n */"),
            " hello\n world\n foo"
        );

        assert_eq!(preprocess("/**\nhello\n*world\n*foo\n*/"), "hello\nworld\nfoo");
    }
}
