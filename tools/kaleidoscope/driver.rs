use std::io;
use std::io::Write;

use iron_llvm::core::value::Value;
use iron_llvm::target;

use builder;
use builder::{IRBuilder, ModuleProvider};
use filer;
use jitter;
use jitter::JITter;
use lexer::*;
use parser::*;

pub use self::Stage::{Exec, Tokens, AST, IR};

#[derive(PartialEq, Clone, Debug)]
pub enum Stage {
    Exec,
    IR,
    AST,
    Tokens,
}

pub fn main_loop(stage: Stage) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut input = String::new();
    let mut parser_settings = default_parser_settings();
    // let mut ir_container = builder::SimpleModuleProvider::new("main");
    let mut ir_container: Box<JITter> = if stage == Exec {
        target::initilalize_native_target();
        target::initilalize_native_asm_printer();
        jitter::init();
        Box::new(jitter::MCJITter::new("main"))
    } else {
        Box::new(builder::SimpleModuleProvider::new("main"))
    };
    let mut builder_context = builder::Context::new();
    match filer::load_stdlib(
        &mut parser_settings,
        &mut builder_context,
        ir_container.get_module_provider(),
    ) {
        Ok((value, runnable)) => {
            if runnable && stage == Exec {
                ir_container.run_function(value);
            }
        }
        Err(err) => print!("Error occured during stdlib loading: {}\n", err),
    };

    'main: loop {
        print!("> ");
        stdout.flush().unwrap();
        input.clear();
        stdin
            .read_line(&mut input)
            .ok()
            .expect("Failed to read line");
        if input.as_str() == ".quit\n" {
            break;
        } else if &input[0..5] == ".load" {
            let mut path = input[6..].to_string();
            match path.pop() {
                Some(_) => (),
                None => {
                    print!("Error occured during loading: empty path\n");
                    continue;
                }
            };
            match filer::load_ks(
                path,
                &mut parser_settings,
                &mut builder_context,
                ir_container.get_module_provider(),
            ) {
                Ok((value, runnable)) => {
                    if runnable && stage == Exec {
                        ir_container.run_function(value);
                    }
                }
                Err(err) => print!("Error occured during loading: {}\n", err),
            };
            continue;
        } else if &input[0..5] == ".dump" {
            let mut path = input[6..].to_string();
            match path.pop() {
                Some(_) => (),
                None => {
                    print!("Error occured during dumping: empty path\n");
                    continue;
                }
            };
            match filer::dump_bitcode(&path, ir_container.get_module_provider()) {
                Ok(_) => (),
                Err(_) => print!("Error occured during dumping\n"),
            };
            continue;
        } else if input.as_str() == ".help\n" {
            print!("Enter Kaleidoscope expressions or special commands.\n");
            print!("Special commands are:\n");
            print!(".quit -- quit\n");
            print!(".load <path> -- load .ks file\n");
            print!(".dump <path> -- dump bitcode of currently open module\n");
            print!(".help -- show this help message\n");
            continue;
        }

        let mut ast = Vec::new();
        let mut prev = Vec::new();
        loop {
            let tokens = tokenize(input.as_str());
            if stage == Tokens {
                println!("{:?}", tokens);
                continue 'main;
            }
            prev.extend(tokens.into_iter());

            let parsing_result = parse(prev.as_slice(), ast.as_slice(), &mut parser_settings);
            match parsing_result {
                Ok((parsed_ast, rest)) => {
                    ast.extend(parsed_ast.into_iter());
                    if rest.is_empty() {
                        break;
                    } else {
                        prev = rest;
                    }
                }
                Err(message) => {
                    println!("Error occured: {}", message);
                    continue 'main;
                }
            }
            print!(". ");
            stdout.flush().unwrap();
            input.clear();
            stdin
                .read_line(&mut input)
                .ok()
                .expect("Failed to read line");
        }

        if stage == AST {
            println!("{:?}", ast);
            continue;
        }

        match ast.codegen(
            &mut builder_context,
            ir_container.get_module_provider(), /*
                                                                          &mut ir_container) {
                                                            Ok((value, _)) => value.dump(),
                                                */
        ) {
            Ok((value, runnable)) => {
                if runnable && stage == Exec {
                    println!("=> {}", ir_container.run_function(value));
                } else {
                    value.dump();
                }
            }
            Err(message) => println!("Error occured: {}", message),
        }
    }

    if stage == IR || stage == Exec {
        ir_container.dump();
    }
}
