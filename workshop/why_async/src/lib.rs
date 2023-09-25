#![cfg(test)]

use futures::{executor::block_on, join};
use std::thread;

fn download(_url: &str) {
    // ...
}

#[test]
fn get_two_sites() {
    let thread_one = thread::spawn(|| download("https://www.foo.com"));
    let thread_two = thread::spawn(|| download("https://www.bar.com"));
    thread_one.join().expect("thread one panicked");
    thread_two.join().expect("thread two panicked");
}

async fn download_async(_url: &str) {
    // ...
}

async fn get_two_sites_async() {
    let future_one = download_async("https://www.foo.com");
    let future_two = download_async("https://www.bar.com");
    join!(future_one, future_two);
}

#[test]
fn get_two_sites_async_test() {
    block_on(get_two_sites_async());
}
