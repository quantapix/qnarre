use std::{env, process::Command, str, u32};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let y = match rustc_version() {
        Some(x) => x,
        None => return,
    };
    let _ = y.minor;
    if !cfg!(feature = "proc-macro") {
        return;
    }
    println!("cargo:rustc-cfg=wrap_proc_macro");
    if y.nightly {
        println!("cargo:rustc-cfg=proc_macro_span");
    }
    if !y.nightly {
        println!("cargo:rustc-cfg=syn_disable_nightly_tests");
    }
}

struct Compiler {
    minor: u32,
    nightly: bool,
}

fn rustc_version() -> Option<Compiler> {
    let x = env::var_os("RUSTC")?;
    let y = Command::new(x).arg("--version").output().ok()?;
    let y = str::from_utf8(&y.stdout).ok()?;
    let nightly = y.contains("nightly") || y.ends_with("-dev");
    let mut ys = y.split('.');
    if ys.next() != Some("rustc 1") {
        return None;
    }
    let minor = ys.next()?.parse().ok()?;
    Some(Compiler { minor, nightly })
}

fn feature_allowed(feature: &str) -> bool {
    // Recognized formats:
    //     -Z allow-features=feature1,feature2
    //     -Zallow-features=feature1,feature2
    let flags_var;
    let flags_var_string;
    let ys = if let Some(x) = env::var_os("CARGO_ENCODED_RUSTFLAGS") {
        flags_var = x;
        flags_var_string = flags_var.to_string_lossy();
        flags_var_string.split('\x1f')
    } else {
        return true;
    };
    for mut y in ys {
        if y.starts_with("-Z") {
            y = &y["-Z".len()..];
        }
        if y.starts_with("allow-features=") {
            y = &y["allow-features=".len()..];
            return y.split(',').any(|x| x == feature);
        }
    }
    true
}
