use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Never use the host interpreter's paths for a cross-compiled target.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var("HOST") != env::var("TARGET")
    {
        return;
    }

    let config = pyo3_build_config::get();
    let (Some(lib_dir), Some(lib_name), Some(executable)) =
        (config.lib_dir(), config.lib_name(), config.executable())
    else {
        return;
    };
    if Path::new(lib_dir).is_dir() {
        return;
    }

    // Apple's Command Line Tools Python can retain sysconfig paths pointing
    // into Xcode. Its runtime base prefix identifies the actual installation,
    // including when the selected interpreter belongs to a virtual environment.
    let output = Command::new(executable)
        .args(["-c", "import sys; print(sys.base_prefix)"])
        .output()
        .expect("failed to query Python's installation prefix");
    assert!(output.status.success(), "failed to query Python's prefix");
    let prefix = String::from_utf8(output.stdout).expect("Python prefix is not UTF-8");
    let prefix = Path::new(prefix.trim());
    let actual_lib_dir = prefix.join("lib");
    if !actual_lib_dir
        .join(format!("lib{lib_name}.dylib"))
        .is_file()
    {
        return;
    }

    println!(
        "cargo:rustc-link-search=native={}",
        actual_lib_dir.display()
    );
    // Apple framework libraries use an @rpath install name.
    if let Some(framework_dir) = prefix
        .ancestors()
        .find(|path| path.extension().is_some_and(|ext| ext == "framework"))
        .and_then(Path::parent)
    {
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            framework_dir.display()
        );
    }
}
