use std::{env, process::Command, str, u32};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let y = match rustc_version() {
        Some(x) => x,
        None => return,
    };
    let _ = y.minor;
    let exempt = cfg!(procmacro2_semver_exempt);
    if exempt {
        println!("cargo:rustc-cfg=procmacro2_semver_exempt");
    }
    if exempt || cfg!(feature = "span-locations") {
        println!("cargo:rustc-cfg=span_locations");
    }
    if !cfg!(feature = "proc-macro") {
        return;
    }
    if y.nightly || !exempt {
        println!("cargo:rustc-cfg=wrap_proc_macro");
    }
    if y.nightly && feature_allowed("proc_macro_span") {
        println!("cargo:rustc-cfg=proc_macro_span");
    }
    if exempt && y.nightly {
        println!("cargo:rustc-cfg=super_unstable");
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

// rustc-cfg emitted by the build script:
//
// "wrap_proc_macro"
//     Wrap types from libproc_macro rather than polyfilling the whole API.
//     Enabled on rustc 1.29+ as long as procmacro2_semver_exempt is not set,
//     because we can't emulate the unstable API without emulating everything
//     else. Also enabled unconditionally on nightly, in which case the
//     procmacro2_semver_exempt surface area is implemented by using the
//     nightly-only proc_macro API.
//
// "hygiene"
//    Enable Span::mixed_site() and non-dummy behavior of Span::resolved_at
//    and Span::located_at. Enabled on Rust 1.45+.
//
// "proc_macro_span"
//     Enable non-dummy behavior of Span::start and Span::end methods which
//     requires an unstable compiler feature. Enabled when building with
//     nightly, unless `-Z allow-feature` in RUSTFLAGS disallows unstable
//     features.
//
// "super_unstable"
//     Implement the semver exempt API in terms of the nightly-only proc_macro
//     API. Enabled when using procmacro2_semver_exempt on a nightly compiler.
//
// "span_locations"
//     Provide methods Span::start and Span::end which give the line/column
//     location of a token. Enabled by procmacro2_semver_exempt or the
//     "span-locations" Cargo cfg. This is behind a cfg because tracking
//     location inside spans is a performance hit.
//
// "is_available"
//     Use proc_macro::is_available() to detect if the proc macro API is
//     available or needs to be polyfilled instead of trying to use the proc
//     macro API and catching a panic if it isn't available. Enabled on Rust
//     1.57+.
