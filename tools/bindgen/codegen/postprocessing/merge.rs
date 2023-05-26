use syn::{
    visit_mut::{visit_file_mut, visit_item_mod_mut, VisitMut},
    File, Item, ItemForeignMod, ItemMod,
};

pub(super) fn merge_extern_blocks(x: &mut File) {
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

fn visit_items(xs: &mut Vec<Item>) {
    let mut ys = Vec::<ItemForeignMod>::new();
    for x in std::mem::take(xs) {
        if let Item::ForeignMod(ItemForeignMod {
            attrs,
            abi,
            brace_token,
            unsafety,
            items,
        }) = x
        {
            let mut exists = false;
            for y in &mut ys {
                if y.attrs == attrs && y.abi == abi {
                    y.items.extend_from_slice(&items);
                    exists = true;
                    break;
                }
            }
            if !exists {
                ys.push(ItemForeignMod {
                    attrs,
                    abi,
                    brace_token,
                    unsafety,
                    items,
                });
            }
        } else {
            xs.push(x);
        }
    }
    for y in ys {
        xs.push(Item::ForeignMod(y));
    }
}
