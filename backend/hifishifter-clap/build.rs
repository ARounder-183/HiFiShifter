//! CLAP 产物的构建身份：与 App / VST3 共用产品版本单一来源。
#[path = "../build-support/product_version.rs"]
mod product_version;

fn main() {
    product_version::emit(
        &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        env!("CARGO_PKG_VERSION"),
    );
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if matches!(os.as_str(), "linux" | "freebsd" | "android") {
        println!("cargo:rustc-link-arg=-Wl,--enable-new-dtags");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    } else if os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../Frameworks");
    }
}
