use lhash::sha512;

fn main() {
    fn say_hi() {
        println!("Hi! sha512(\"hi\") = {:x?}", sha512(b"hi"));
    }

    probe_lib::do_thrice! {{ say_hi(); }}
}
