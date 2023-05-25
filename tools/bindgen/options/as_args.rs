use std::path::PathBuf;

use crate::RegexSet;

pub(super) trait AsArgs {
    fn as_args(&self, args: &mut Vec<String>, flag: &str);
}

impl AsArgs for bool {
    fn as_args(&self, args: &mut Vec<String>, flag: &str) {
        if *self {
            args.push(flag.to_string());
        }
    }
}

impl AsArgs for RegexSet {
    fn as_args(&self, args: &mut Vec<String>, flag: &str) {
        for item in self.get_items() {
            args.extend_from_slice(&[flag.to_owned(), item.clone()]);
        }
    }
}

impl AsArgs for Option<String> {
    fn as_args(&self, args: &mut Vec<String>, flag: &str) {
        if let Some(string) = self {
            args.extend_from_slice(&[flag.to_owned(), string.clone()]);
        }
    }
}

impl AsArgs for Option<PathBuf> {
    fn as_args(&self, args: &mut Vec<String>, flag: &str) {
        if let Some(path) = self {
            args.extend_from_slice(&[flag.to_owned(), path.display().to_string()]);
        }
    }
}
