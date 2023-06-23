use crate::token::CommentKind;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, CharPos, FileName, Pos, Symbol};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CommentStyle {
    Isolated,
    Trailing,
    Mixed,
    BlankLine,
}

#[derive(Clone)]
pub struct Comment {
    pub style: CommentStyle,
    pub lines: Vec<String>,
    pub pos: BytePos,
}

#[inline]
pub fn may_have_doc_links(s: &str) -> bool {
    s.contains('[')
}

pub fn beautify_doc_string(data: Symbol, kind: CommentKind) -> Symbol {
    fn get_vertical_trim(lines: &[&str]) -> Option<(usize, usize)> {
        let mut i = 0;
        let mut j = lines.len();
        if !lines.is_empty() && lines[0].chars().all(|c| c == '*') {
            i += 1;
        }

        if j > i && !lines[j - 1].is_empty() && lines[j - 1].chars().all(|c| c == '*') {
            j -= 1;
        }

        if i != 0 || j != lines.len() {
            Some((i, j))
        } else {
            None
        }
    }

    fn get_horizontal_trim(lines: &[&str], kind: CommentKind) -> Option<String> {
        let mut i = usize::MAX;
        let mut first = true;

        let lines = match kind {
            CommentKind::Block => {
                let mut i = lines
                    .get(0)
                    .map(|l| if l.trim_start().starts_with('*') { 0 } else { 1 })
                    .unwrap_or(0);
                let mut j = lines.len();

                while i < j && lines[i].trim().is_empty() {
                    i += 1;
                }
                while j > i && lines[j - 1].trim().is_empty() {
                    j -= 1;
                }
                &lines[i..j]
            },
            CommentKind::Line => lines,
        };

        for line in lines {
            for (j, c) in line.chars().enumerate() {
                if j > i || !"* \t".contains(c) {
                    return None;
                }
                if c == '*' {
                    if first {
                        i = j;
                        first = false;
                    } else if i != j {
                        return None;
                    }
                    break;
                }
            }
            if i >= line.len() {
                return None;
            }
        }
        if lines.is_empty() {
            None
        } else {
            Some(lines[0][..i].into())
        }
    }

    let data_s = data.as_str();
    if data_s.contains('\n') {
        let mut lines = data_s.lines().collect::<Vec<&str>>();
        let mut changes = false;
        let lines = if let Some((i, j)) = get_vertical_trim(&lines) {
            changes = true;
            &mut lines[i..j]
        } else {
            &mut lines
        };
        if let Some(horizontal) = get_horizontal_trim(lines, kind) {
            changes = true;
            for line in lines.iter_mut() {
                if let Some(tmp) = line.strip_prefix(&horizontal) {
                    *line = tmp;
                    if kind == CommentKind::Block && (*line == "*" || line.starts_with("* ") || line.starts_with("**"))
                    {
                        *line = &line[1..];
                    }
                }
            }
        }
        if changes {
            return Symbol::intern(&lines.join("\n"));
        }
    }
    data
}

fn all_whitespace(s: &str, col: CharPos) -> Option<usize> {
    let mut idx = 0;
    for (i, ch) in s.char_indices().take(col.to_usize()) {
        if !ch.is_whitespace() {
            return None;
        }
        idx = i + ch.len_utf8();
    }
    Some(idx)
}

fn trim_whitespace_prefix(s: &str, col: CharPos) -> &str {
    let len = s.len();
    match all_whitespace(s, col) {
        Some(col) => {
            if col < len {
                &s[col..]
            } else {
                ""
            }
        },
        None => s,
    }
}

fn split_block_comment_into_lines(text: &str, col: CharPos) -> Vec<String> {
    let mut res: Vec<String> = vec![];
    let mut lines = text.lines();
    res.extend(lines.next().map(|it| it.to_string()));
    for line in lines {
        res.push(trim_whitespace_prefix(line, col).to_string())
    }
    res
}

pub fn gather_comments(sm: &SourceMap, path: FileName, src: String) -> Vec<Comment> {
    let sm = SourceMap::new(sm.path_mapping().clone());
    let source_file = sm.new_source_file(path, src);
    let text = (*source_file.src.as_ref().unwrap()).clone();

    let text: &str = text.as_str();
    let start_bpos = source_file.start_pos;
    let mut pos = 0;
    let mut comments: Vec<Comment> = Vec::new();
    let mut code_to_the_left = false;

    if let Some(shebang_len) = lexer::strip_shebang(text) {
        comments.push(Comment {
            style: CommentStyle::Isolated,
            lines: vec![text[..shebang_len].to_string()],
            pos: start_bpos,
        });
        pos += shebang_len;
    }

    for token in lexer::tokenize(&text[pos..]) {
        let token_text = &text[pos..pos + token.len as usize];
        match token.kind {
            lexer::TokenKind::Whitespace => {
                if let Some(mut idx) = token_text.find('\n') {
                    code_to_the_left = false;
                    while let Some(next_newline) = &token_text[idx + 1..].find('\n') {
                        idx += 1 + next_newline;
                        comments.push(Comment {
                            style: CommentStyle::BlankLine,
                            lines: vec![],
                            pos: start_bpos + BytePos((pos + idx) as u32),
                        });
                    }
                }
            },
            lexer::TokenKind::BlockComment { doc_style, .. } => {
                if doc_style.is_none() {
                    let code_to_the_right =
                        !matches!(text[pos + token.len as usize..].chars().next(), Some('\r' | '\n'));
                    let style = match (code_to_the_left, code_to_the_right) {
                        (_, true) => CommentStyle::Mixed,
                        (false, false) => CommentStyle::Isolated,
                        (true, false) => CommentStyle::Trailing,
                    };

                    let pos_in_file = start_bpos + BytePos(pos as u32);
                    let line_begin_in_file = source_file.line_begin_pos(pos_in_file);
                    let line_begin_pos = (line_begin_in_file - start_bpos).to_usize();
                    let col = CharPos(text[line_begin_pos..pos].chars().count());

                    let lines = split_block_comment_into_lines(token_text, col);
                    comments.push(Comment {
                        style,
                        lines,
                        pos: pos_in_file,
                    })
                }
            },
            lexer::TokenKind::LineComment { doc_style } => {
                if doc_style.is_none() {
                    comments.push(Comment {
                        style: if code_to_the_left {
                            CommentStyle::Trailing
                        } else {
                            CommentStyle::Isolated
                        },
                        lines: vec![token_text.to_string()],
                        pos: start_bpos + BytePos(pos as u32),
                    })
                }
            },
            _ => {
                code_to_the_left = true;
            },
        }
        pos += token.len as usize;
    }

    comments
}
