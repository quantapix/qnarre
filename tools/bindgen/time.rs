use std::io::{self, Write};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct Timer<'a> {
    output: bool,
    name: &'a str,
    start: Instant,
}

impl<'a> Timer<'a> {
    pub fn new(name: &'a str) -> Self {
        Timer {
            output: true,
            name,
            start: Instant::now(),
        }
    }
    pub fn with_output(mut self, x: bool) -> Self {
        self.output = x;
        self
    }
    pub fn elapsed(&self) -> Duration {
        Instant::now() - self.start
    }
    fn print_elapsed(&mut self) {
        if self.output {
            let d = self.elapsed();
            let ms = (d.as_secs() as f64) * 1e3 + (d.subsec_nanos() as f64) / 1e6;
            let e = io::stderr();
            writeln!(e.lock(), "  time: {:>9.3} ms.\t{}", ms, self.name).expect("should not fail");
        }
    }
}

impl<'a> Drop for Timer<'a> {
    fn drop(&mut self) {
        self.print_elapsed();
    }
}
