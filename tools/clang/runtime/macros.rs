macro_rules! test {
    () => {
        cfg!(test)
    };
}

macro_rules! target_os {
    ($os:expr) => {
        cfg!(target_os = $os)
    };
}
