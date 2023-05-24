use std::{
    env,
    error::Error,
    fs, io,
    path::Path,
    process::{exit, Command},
    str,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("{}", error);
        exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    dbg!(llvm_config("--version")?);
    let v = llvm_config("--version")?;
    if !v.ends_with("git") {
        return Err(format!("failed to find dev version (X.X.Xgit) of llvm-config (found {})", v).into());
    }
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rustc-link-search={}", llvm_config("--libdir")?);
    for name in fs::read_dir(llvm_config("--libdir")?)?
        .map(|x| {
            Ok(if let Some(y) = x?.path().file_name() {
                y.to_str().map(String::from)
            } else {
                None
            })
        })
        .collect::<Result<Vec<_>, io::Error>>()?
        .into_iter()
        .flatten()
    {
        if name.starts_with("libMLIR")
            && name.ends_with(".a")
            && !name.contains("Main")
            && name != "libMLIRSupportIndentedOstream.a"
        {
            if let Some(x) = trim_lib_name(&name) {
                println!("cargo:rustc-link-lib=static={}", x);
            }
        }
    }
    for name in llvm_config("--libnames")?.trim().split(' ') {
        if let Some(x) = trim_lib_name(name) {
            println!("cargo:rustc-link-lib={}", x);
        }
    }
    for flag in llvm_config("--system-libs")?.trim().split(' ') {
        let flag = flag.trim_start_matches("-l");
        if flag.starts_with('/') {
            let p = Path::new(flag);
            println!("cargo:rustc-link-search={}", p.parent().unwrap().display());
            println!(
                "cargo:rustc-link-lib={}",
                p.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .split_once('.')
                    .unwrap()
                    .0
                    .trim_start_matches("lib")
            );
        } else {
            println!("cargo:rustc-link-lib={}", flag);
        }
    }
    if let Some(x) = get_libcpp() {
        println!("cargo:rustc-link-lib={}", x);
    }
    bindgen::builder()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", llvm_config("--includedir")?))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .unwrap()
        .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;
    Ok(())
}

const LLVM_CONFIG: &str = "/usr/local/llvm/bin/llvm-config";

fn llvm_config(args: &str) -> Result<String, Box<dyn Error>> {
    let p = env::var("LLVM_CONFIG_PATH").unwrap_or(LLVM_CONFIG.into());
    let p = format!("{} --link-static {}", p, args);
    Ok(
        str::from_utf8(&{ Command::new("sh").arg("-c").arg(&p).output()? }.stdout)?
            .trim()
            .to_string(),
    )
}

fn get_libcpp() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some("c++")
    } else {
        Some("stdc++")
    }
}

fn trim_lib_name(n: &str) -> Option<&str> {
    if let Some(x) = n.strip_prefix("lib") {
        x.strip_suffix(".a")
    } else {
        None
    }
}
