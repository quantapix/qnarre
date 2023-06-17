extern crate glob;
extern crate serial_test;
extern crate tempfile;

use serial_test::serial;
use std::{collections::HashMap, env, fs, path::PathBuf, sync::Arc, sync::Mutex};
use tempfile::TempDir;

#[path = "../runtime/main.rs"]
mod runtime;

#[derive(Debug, Default)]
struct RunCmdMock {
    invocations: Vec<String>,
    responses: HashMap<Vec<String>, String>,
}

#[derive(Debug)]
struct Env {
    vars: HashMap<String, (Option<String>, Option<String>)>,
    cwd: PathBuf,
    tmp: TempDir,
    files: Vec<String>,
    commands: Arc<Mutex<RunCmdMock>>,
}

impl Env {
    fn new() -> Self {
        Env {
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
    fn var(mut self, name: &str, x: Option<&str>) -> Self {
        let old = env::var(name).ok();
        self.vars.insert(name.into(), (x.map(|x| x.into()), old));
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
        for (name, (value, _)) in &self.vars {
            if let Some(value) = value {
                env::set_var(name, value);
            } else {
                env::remove_var(name);
            }
        }
        env::set_current_dir(&self.tmp).unwrap();
        let commands = self.commands.clone();
        let mock = &mut *runtime::MOCK.lock().unwrap();
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
    let _env = Env::new()
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
    let _env = Env::new()
        .so("usr/lib/libclang-3.so", "64")
        .so("usr/lib/libclang-3.5.so", "64")
        .so("usr/lib/libclang-3.5.0.so", "64")
        .enable();
    assert_eq!(dynamic::find(true), Ok(("usr/lib".into(), "libclang-3.5.0.so".into())),);
}

#[test]
#[serial]
fn test_linux_directory_and_version_preference() {
    let _env = Env::new()
        .so("usr/local/llvm/lib/libclang-3.so", "64")
        .so("usr/local/lib/libclang-3.5.so", "64")
        .so("usr/lib/libclang-3.5.0.so", "64")
        .enable();
    assert_eq!(dynamic::find(true), Ok(("usr/lib".into(), "libclang-3.5.0.so".into())),);
}
