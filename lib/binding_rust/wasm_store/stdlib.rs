//! The vendored Wasm stdlib module and its exported symbol list.
//!
//! The module bytes are extracted at build time from the generated C header
//! `lib/src/wasm-stdlib/external_scanner_stdlib.h` (see `build.rs`), keeping
//! that header the single source of truth for both the C and Rust backends.

include!(concat!(env!("OUT_DIR"), "/stdlib_wasm.rs"));

/// Symbols the stdlib module exports and external scanners may import,
/// parsed from `lib/src/wasm-stdlib/imports.txt` at compile time.
pub(crate) fn stdlib_symbols() -> &'static [&'static str] {
    static SYMBOLS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    SYMBOLS.get_or_init(|| {
        const IMPORTS: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/wasm-stdlib/imports.txt"
        ));
        IMPORTS
            .lines()
            .map(|line| line.trim().trim_matches(|c| c == '"' || c == ','))
            .filter(|line| !line.is_empty())
            .collect()
    })
}
