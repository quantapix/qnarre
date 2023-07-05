use builder;

#[derive(builder::Builder)]
pub struct Command {
    executable: String,
    args: Vec<String>,
    env: Vec<String>,
    current_dir: String,
}

mod utils {
    use std::mem;
    pub trait HeapSize {
        fn heap_size_of_children(&self) -> usize;
    }
    impl HeapSize for u8 {
        fn heap_size_of_children(&self) -> usize {
            0
        }
    }
    impl HeapSize for String {
        fn heap_size_of_children(&self) -> usize {
            self.capacity()
        }
    }
    impl<T> HeapSize for Box<T>
    where
        T: ?Sized + HeapSize,
    {
        fn heap_size_of_children(&self) -> usize {
            mem::size_of_val(&**self) + (**self).heap_size_of_children()
        }
    }
    impl<T> HeapSize for [T]
    where
        T: HeapSize,
    {
        fn heap_size_of_children(&self) -> usize {
            self.iter().map(HeapSize::heap_size_of_children).sum()
        }
    }
    impl<'a, T> HeapSize for &'a T
    where
        T: ?Sized,
    {
        fn heap_size_of_children(&self) -> usize {
            0
        }
    }
}
use utils::HeapSize;

#[derive(builder::HeapSize)]
struct Demo<'a, T: ?Sized> {
    a: Box<T>,
    b: u8,
    c: &'a str,
    d: String,
}

fn main() {
    let builder = Command::builder();
    let _ = builder;
    let demo = Demo {
        a: b"bytestring".to_vec().into_boxed_slice(),
        b: 255,
        c: "&'static str",
        d: "String".to_owned(),
    };
    // 10 + 0 + 0 + 6 = 16
    println!(
        "heap size = {} + {} + {} + {} = {}",
        demo.a.heap_size_of_children(),
        demo.b.heap_size_of_children(),
        demo.c.heap_size_of_children(),
        demo.d.heap_size_of_children(),
        demo.heap_size_of_children()
    );
}
