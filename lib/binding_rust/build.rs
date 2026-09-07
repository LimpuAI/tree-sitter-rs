use std::{env, fs, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();

    #[cfg(feature = "bindgen")]
    generate_bindings(&out_dir);

    fs::copy(
        "src/wasm-stdlib/imports.txt",
        out_dir.join("stdlib-symbols.txt"),
    )
    .unwrap();

    let mut config = cc::Build::new();

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_WASM");
    if env::var("CARGO_FEATURE_WASM").is_ok() {
        // The Wasm store itself is implemented in Rust (binding_rust/wasm_store);
        // this define only gates `lib.c`'s inclusion of `wasm_store.c` and the
        // C stub branch when the feature is off.
        config
            .define("TREE_SITTER_FEATURE_WASM", "")
            .define("static_assert(...)", "");
        emit_stdlib_wasm(&out_dir);
    }

    let manifest_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let include_path = manifest_path.join("include");
    let src_path = manifest_path.join("src");
    let wasm_path = src_path.join("wasm");

    if target.starts_with("wasm32-unknown") {
        configure_wasm_build(&mut config);
    }

    for entry in fs::read_dir(&src_path).unwrap() {
        let entry = entry.unwrap();
        let path = src_path.join(entry.file_name());
        println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
    }

    config
        .std("c11")
        .flag_if_supported("-fvisibility=hidden")
        .flag_if_supported("-Wshadow")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-incompatible-pointer-types")
        .include(&src_path)
        .include(&wasm_path)
        .include(&include_path)
        .define("_POSIX_C_SOURCE", "200112L")
        .define("_DEFAULT_SOURCE", None)
        .define("_BSD_SOURCE", None)
        .define("_DARWIN_C_SOURCE", None)
        .warnings(false)
        .file(src_path.join("lib.c"))
        .compile("tree-sitter");

    println!("cargo:include={}", include_path.display());
}

fn configure_wasm_build(config: &mut cc::Build) {
    let Ok(wasm_headers) = env::var("DEP_TREE_SITTER_LANGUAGE_WASM_HEADERS") else {
        panic!(
            "Environment variable DEP_TREE_SITTER_LANGUAGE_WASM_HEADERS must be set by the language crate"
        );
    };
    config
        .define("TREE_SITTER_WASM_STDLIB", "")
        .include(&wasm_headers);
}

// Extracts the `STDLIB_WASM[]` byte array from the generated C header into a
// Rust source file so the Rust wasm store can instantiate the stdlib module.
fn emit_stdlib_wasm(out_dir: &std::path::Path) {
    println!("cargo:rerun-if-changed=src/wasm-stdlib/external_scanner_stdlib.h");
    let header = fs::read_to_string("src/wasm-stdlib/external_scanner_stdlib.h").unwrap();
    let array_start = header.find("STDLIB_WASM[]").unwrap();
    let open = header[array_start..].find('{').unwrap() + array_start;
    let close = header[open..].find('}').unwrap() + open;
    let bytes: Vec<String> = header[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| {
            let value = u8::from_str_radix(token.trim_start_matches("0x"), 16).unwrap();
            format!("{value:#04x}")
        })
        .collect();
    let source = format!(
        "pub(crate) static STDLIB_WASM: [u8; {}] = [{}];\n",
        bytes.len(),
        bytes.join(", ")
    );
    fs::write(out_dir.join("stdlib_wasm.rs"), source).unwrap();
}

#[cfg(feature = "bindgen")]
fn generate_bindings(out_dir: &std::path::Path) {
    use std::str::FromStr;

    use bindgen::RustTarget;

    const HEADER_PATH: &str = "include/tree_sitter/api.h";

    println!("cargo:rerun-if-changed={HEADER_PATH}");

    let no_copy = [
        "TSInput",
        "TSLanguage",
        "TSLogger",
        "TSLookaheadIterator",
        "TSParser",
        "TSTree",
        "TSQuery",
        "TSQueryCursor",
        "TSQueryCapture",
        "TSQueryMatch",
        "TSQueryPredicateStep",
    ];

    let rust_version = env!("CARGO_PKG_RUST_VERSION");

    let bindings = bindgen::Builder::default()
        .header(HEADER_PATH)
        .layout_tests(false)
        .allowlist_type("^TS.*")
        .allowlist_function("^ts_.*")
        .allowlist_var("^TREE_SITTER.*")
        .no_copy(no_copy.join("|"))
        .prepend_enum_name(false)
        .use_core()
        .clang_arg("-D TREE_SITTER_FEATURE_WASM")
        .rust_target(RustTarget::from_str(rust_version).unwrap())
        .generate()
        .expect("Failed to generate bindings");

    let bindings_rs = out_dir.join("bindings.rs");
    bindings.write_to_file(&bindings_rs).unwrap_or_else(|_| {
        panic!(
            "Failed to write bindings into path: {}",
            bindings_rs.display()
        )
    });
}
