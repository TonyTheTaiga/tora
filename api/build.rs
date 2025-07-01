use std::process::Command;
fn main() {
    let status = Command::new("npm")
        .args(["run", "build:production"])
        .current_dir("../web-new")
        .status()
        .expect("failed to run npm build");
    assert!(status.success());
    println!("cargo:rerun-if-changed=../web-new");
}
