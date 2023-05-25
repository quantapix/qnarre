use std::env;
use std::fs::File;
use std::io::{self, Error, ErrorKind, Read};
use std::path::{Path, PathBuf};

use super::common;

fn parse_elf_header(path: &Path) -> io::Result<u8> {
    let mut f = File::open(path)?;
    let mut buf = [0; 5];
    f.read_exact(&mut buf)?;
    if buf[..4] == [127, 69, 76, 70] {
        Ok(buf[4])
    } else {
        Err(Error::new(ErrorKind::InvalidData, "invalid ELF header"))
    }
}

fn validate_lib(path: &Path) -> Result<(), String> {
    if target_os!("linux") {
        let class = parse_elf_header(path).map_err(|x| x.to_string())?;
        if target_pointer_width!("32") && class != 1 {
            return Err("invalid ELF class (64-bit)".into());
        }
        if target_pointer_width!("64") && class != 2 {
            return Err("invalid ELF class (32-bit)".into());
        }
        Ok(())
    } else {
        Ok(())
    }
}

fn parse_version(file: &str) -> Vec<u32> {
    let y = if let Some(x) = file.strip_prefix("libclang.so.") {
        x
    } else if file.starts_with("libclang-") {
        &file[9..file.len() - 3]
    } else {
        return vec![];
    };
    y.split('.').map(|s| s.parse().unwrap_or(0)).collect()
}

fn search_clang_dirs(runtime: bool) -> Result<Vec<(PathBuf, String, Vec<u32>)>, String> {
    let mut ys = vec![format!("{}clang{}", env::consts::DLL_PREFIX, env::consts::DLL_SUFFIX)];
    if target_os!("linux") {
        ys.push("libclang-*.so".into());
        if runtime {
            ys.push("libclang.so.*".into());
            ys.push("libclang-*.so.*".into());
        }
    }
    let mut valid = vec![];
    let mut invalid = vec![];
    for (dir, file) in common::search_clang_dirs(&ys, "LIBCLANG_PATH") {
        let p = dir.join(&file);
        match validate_lib(&p) {
            Ok(()) => {
                let v = parse_version(&file);
                valid.push((dir, file, v))
            },
            Err(x) => invalid.push(format!("({}: {})", p.display(), x)),
        }
    }
    if !valid.is_empty() {
        return Ok(valid);
    }
    let msg = format!(
        "couldn't find any valid shared libraries matching: [{}], set the \
         `LIBCLANG_PATH` environment variable to a path where one of these files \
         can be found (invalid: [{}])",
        ys.iter().map(|x| format!("'{}'", x)).collect::<Vec<_>>().join(", "),
        invalid.join(", "),
    );
    Err(msg)
}

pub fn find(runtime: bool) -> Result<(PathBuf, String), String> {
    search_clang_dirs(runtime)?
        .iter()
        .rev()
        .max_by_key(|x| &x.2)
        .cloned()
        .map(|(dir, file, _)| (dir, file))
        .ok_or_else(|| "unreachable".into())
}

#[cfg(not(feature = "runtime"))]
pub fn link() {
    let cep = common::CmdErrorPrinter::default();
    let (dir, file) = find(false).unwrap();
    println!("cargo:rustc-link-search={}", dir.display());
    let name = file.trim_start_matches("lib");
    let name = match name.find(".dylib").or_else(|| name.find(".so")) {
        Some(i) => &name[0..i],
        None => name,
    };
    println!("cargo:rustc-link-lib=dylib={}", name);
    cep.discard();
}
