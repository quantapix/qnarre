extern crate glob;
extern crate serial_test;
extern crate tempfile;

use serial_test::serial;
use std::{collections::HashMap, env, fs, path::PathBuf, sync::Arc, sync::Mutex};
use tempfile::TempDir;

#[macro_use]
#[path = "../runtime/macros.rs"]
mod macros;

#[path = "../runtime/common.rs"]
mod common;
#[path = "../runtime/dynamic.rs"]
mod dynamic;
#[path = "../runtime/static.rs"]
mod r#static;

#[derive(Debug, Default)]
struct RunCmdMock {
    invocations: Vec<String>,
    responses: HashMap<Vec<String>, String>,
}

#[derive(Debug)]
struct Env {
    os: String,
    ptr_width: String,
    env: Option<String>,
    vars: HashMap<String, (Option<String>, Option<String>)>,
    cwd: PathBuf,
    tmp: TempDir,
    files: Vec<String>,
    commands: Arc<Mutex<RunCmdMock>>,
}

impl Env {
    fn new(os: &str, ptr_width: &str) -> Self {
        Env {
            os: os.into(),
            ptr_width: ptr_width.into(),
            env: None,
            vars: HashMap::new(),
            cwd: env::current_dir().unwrap(),
            tmp: tempfile::Builder::new().prefix("clang_test").tempdir().unwrap(),
            files: vec![],
            commands: Default::default(),
        }
        .var("CLANG_PATH", None)
        .var("LD_LIBRARY_PATH", None)
        .var("LIBCLANG_PATH", None)
        .var("LIBCLANG_STATIC_PATH", None)
        .var("LLVM_CONFIG_PATH", None)
        .var("PATH", None)
    }
    fn env(mut self, env: &str) -> Self {
        self.env = Some(env.into());
        self
    }
    fn var(mut self, name: &str, value: Option<&str>) -> Self {
        let previous = env::var(name).ok();
        self.vars.insert(name.into(), (value.map(|x| x.into()), previous));
        self
    }
    fn dir(mut self, path: &str) -> Self {
        self.files.push(path.into());
        let path = self.tmp.path().join(path);
        fs::create_dir_all(path).unwrap();
        self
    }
    fn file(mut self, path: &str, contents: &[u8]) -> Self {
        self.files.push(path.into());
        let path = self.tmp.path().join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(self.tmp.path().join(path), contents).unwrap();
        self
    }
    fn so(self, path: &str, pointer_width: &str) -> Self {
        let class = if pointer_width == "64" { 2 } else { 1 };
        let contents = [127, 69, 76, 70, class];
        self.file(path, &contents)
    }
    fn command(self, command: &str, args: &[&str], response: &str) -> Self {
        let command = command.to_string();
        let args = args.iter().map(|a| a.to_string()).collect::<Vec<_>>();
        let mut key = vec![command];
        key.extend(args);
        self.commands.lock().unwrap().responses.insert(key, response.into());
        self
    }
    fn enable(self) -> Self {
        env::set_var("_CLANG_TEST", "yep");
        env::set_var("_CLANG_TEST_OS", &self.os);
        env::set_var("_CLANG_TEST_POINTER_WIDTH", &self.ptr_width);
        if let Some(env) = &self.env {
            env::set_var("_CLANG_TEST_ENV", env);
        }
        for (name, (value, _)) in &self.vars {
            if let Some(value) = value {
                env::set_var(name, value);
            } else {
                env::remove_var(name);
            }
        }
        env::set_current_dir(&self.tmp).unwrap();
        let commands = self.commands.clone();
        let mock = &mut *common::RUN_CMD_MOCK.lock().unwrap();
        *mock = Some(Box::new(move |args| {
            let mut ys = commands.lock().unwrap();
            ys.invocations.push(args.to_string());
            let mut key = vec![String::from("llvm-config")];
            key.push(args.to_string());
            ys.responses.get(&key).cloned()
        }));
        self
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        env::remove_var("_CLANG_TEST");
        env::remove_var("_CLANG_TEST_OS");
        env::remove_var("_CLANG_TEST_POINTER_WIDTH");
        env::remove_var("_CLANG_TEST_ENV");
        for (name, (_, previous)) in &self.vars {
            if let Some(previous) = previous {
                env::set_var(name, previous);
            } else {
                env::remove_var(name);
            }
        }
        if let Err(error) = env::set_current_dir(&self.cwd) {
            println!("Failed to reset working directory: {:?}", error);
        }
    }
}

#[test]
#[serial]
fn test_linux_directory_preference() {
    let _env = Env::new("linux", "64")
        .so("usr/lib/libclang.so.1", "64")
        .so("usr/local/lib/libclang.so.1", "64")
        .enable();
    assert_eq!(
        dynamic::find(true),
        Ok(("usr/local/lib".into(), "libclang.so.1".into())),
    );
}

#[test]
#[serial]
fn test_linux_version_preference() {
    let _env = Env::new("linux", "64")
        .so("usr/lib/libclang-3.so", "64")
        .so("usr/lib/libclang-3.5.so", "64")
        .so("usr/lib/libclang-3.5.0.so", "64")
        .enable();
    assert_eq!(dynamic::find(true), Ok(("usr/lib".into(), "libclang-3.5.0.so".into())),);
}

#[test]
#[serial]
fn test_linux_directory_and_version_preference() {
    let _env = Env::new("linux", "64")
        .so("usr/local/llvm/lib/libclang-3.so", "64")
        .so("usr/local/lib/libclang-3.5.so", "64")
        .so("usr/lib/libclang-3.5.0.so", "64")
        .enable();
    assert_eq!(dynamic::find(true), Ok(("usr/lib".into(), "libclang-3.5.0.so".into())),);
}
