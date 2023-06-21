include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

fn main() {
    let structure = Structure { foo: 322, bar: 6.44 };
    println!("{:?}", structure);
}
