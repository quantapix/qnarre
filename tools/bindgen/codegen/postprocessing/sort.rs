use syn::{
    visit_mut::{visit_file_mut, visit_item_mod_mut, VisitMut},
    File, Item, ItemMod,
};

pub(super) fn sort_semantically(x: &mut File) {
    Visitor.visit_file_mut(x)
}

struct Visitor;

impl VisitMut for Visitor {
    fn visit_file_mut(&mut self, x: &mut File) {
        visit_items(&mut x.items);
        visit_file_mut(self, x)
    }

    fn visit_item_mod_mut(&mut self, x: &mut ItemMod) {
        if let Some((_, ref mut xs)) = x.content {
            visit_items(xs);
        }
        visit_item_mod_mut(self, x)
    }
}

fn visit_items(xs: &mut [Item]) {
    xs.sort_by_key(|x| match x {
        Item::Type(_) => 0,
        Item::Struct(_) => 1,
        Item::Const(_) => 2,
        Item::Fn(_) => 3,
        Item::Enum(_) => 4,
        Item::Union(_) => 5,
        Item::Static(_) => 6,
        Item::Trait(_) => 7,
        Item::TraitAlias(_) => 8,
        Item::Impl(_) => 9,
        Item::Mod(_) => 10,
        Item::Use(_) => 11,
        Item::Verbatim(_) => 12,
        Item::ExternCrate(_) => 13,
        Item::ForeignMod(_) => 14,
        Item::Macro(_) => 15,
        _ => 18,
    });
}
