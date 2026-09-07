#include "./alloc.c"
#include "./get_changed_ranges.c"
#include "./language.c"
#include "./lexer.c"
#include "./node.c"
#include "./parser.c"
#include "./point.c"
#include "./query.c"
#include "./stack.c"
#include "./subtree.c"
#include "./tree_cursor.c"
#include "./tree.c"

// With the `wasm` feature the Wasm store is implemented in Rust
// (binding_rust/wasm_store) instead of C, so this translation unit must be
// excluded from the compilation. Without the feature, wasm_store.c compiles
// its no-op stub implementations.
#ifndef TREE_SITTER_FEATURE_WASM
#include "./wasm_store.c"
#endif

#ifdef TREE_SITTER_WASM_STDLIB
#include "./wasm-stdlib/libc.c"
#include "./wasm-stdlib/stdio.c"
#endif
