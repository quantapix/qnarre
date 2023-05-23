#![feature(box_syntax)]
#![feature(convert)]
#![feature(plugin)]
#![plugin(regex_macros)]

extern crate regex;

extern crate iron_llvm;
extern crate llvm;

pub mod builder;
pub mod driver;
pub mod filer;
pub mod jitter;
pub mod lexer;
pub mod parser;
