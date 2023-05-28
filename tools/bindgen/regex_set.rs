#![deny(clippy::missing_docs_in_private_items)]
use std::cell::Cell;

#[derive(Clone, Debug, Default)]
pub struct RegexSet {
    items: Vec<String>,
    matched: Vec<Cell<bool>>,
    set: Option<regex::RegexSet>,
    record_matches: bool,
}

impl RegexSet {
    pub fn new() -> RegexSet {
        RegexSet::default()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    pub fn insert<S>(&mut self, x: S)
    where
        S: AsRef<str>,
    {
        self.items.push(x.as_ref().to_owned());
        self.matched.push(Cell::new(false));
        self.set = None;
    }
    pub fn get_items(&self) -> &[String] {
        &self.items[..]
    }
    pub fn unmatched_items(&self) -> impl Iterator<Item = &String> {
        self.items.iter().enumerate().filter_map(move |(i, x)| {
            if !self.record_matches || self.matched[i].get() {
                return None;
            }
            Some(x)
        })
    }
    #[inline]
    pub fn build(&mut self, record_matches: bool) {
        self.build_inner(record_matches, None)
    }
    fn build_inner(&mut self, record_matches: bool, _name: Option<&'static str>) {
        let items = self.items.iter().map(|item| format!("^({})$", item));
        self.record_matches = record_matches;
        self.set = match regex::RegexSet::new(items) {
            Ok(x) => Some(x),
            Err(e) => {
                warn!("Invalid regex in {:?}: {:?}", self.items, e);
                None
            },
        }
    }
    pub fn matches<S>(&self, string: S) -> bool
    where
        S: AsRef<str>,
    {
        let s = string.as_ref();
        let set = match self.set {
            Some(ref set) => set,
            None => return false,
        };
        if !self.record_matches {
            return set.is_match(s);
        }
        let matches = set.matches(s);
        if !matches.matched_any() {
            return false;
        }
        for i in matches.iter() {
            self.matched[i].set(true);
        }
        true
    }
}
