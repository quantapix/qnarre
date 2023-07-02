use std::{
    env,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

fn project_root() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_owned()))
}

fn run(x: &str, dir: impl AsRef<Path>) -> Result<(), ()> {
    let mut xs: Vec<_> = x.split_whitespace().collect();
    let bin = xs.remove(0);
    println!("> {}", x);
    let y = Command::new(bin)
        .args(xs)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .map_err(drop)?;
    if y.status.success() {
        Ok(())
    } else {
        let y = String::from_utf8(y.stdout).map_err(drop)?;
        print!("{}", y);
        Err(())
    }
}

#[test]
fn check_code_formatting() {
    let x = project_root();
    if run("rustfmt +stable --version", &x).is_err() {
        panic!("failed to run rustfmt from toolchain 'stable'",);
    }
    if run("cargo +stable fmt -- --check", &x).is_err() {
        panic!("code is not properly formatted")
    }
}
