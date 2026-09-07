//! Rust implementation of the Wasm store, replacing `lib/src/wasm_store.c`.
//!
//! This module provides every `ts_wasm_store_*` / `ts_wasm_language_*` symbol
//! that the C core (`parser.c`, `language.c`) references, so the C translation
//! unit is excluded from the build when the `wasm` feature is enabled.

// The entire module is private, so `pub(crate)` adds no information, and the
// host-callback signatures are dictated by wasmtime.
#![allow(clippy::redundant_pub_crate, clippy::missing_const_for_fn)]

mod bridge;
mod engine;
mod memory;
mod module;
mod stdlib;

use bridge::LexFunction;

use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use wasmtime::Module;

use crate::ffi::{TSLanguage, TSWasmEngine, TSWasmError, TSWasmStore};
use crate::ffi::{
    TSWasmErrorKindAllocate, TSWasmErrorKindCompile, TSWasmErrorKindInstantiate,
    TSWasmErrorKindParse,
};

pub(crate) use engine::StoreInner;

type WasmErrorKind = u32;

/// Opaque stand-in for the C `TSLexer` (declared in `lib/src/parser.h`, not
/// part of the public api.h). The Rust side only ever handles it as a pointer
/// and accesses fields through the [`CLexer`] mirror.
#[repr(C)]
pub(crate) struct TSLexer {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    static ts_current_malloc: unsafe extern "C" fn(size: usize) -> *mut core::ffi::c_void;
}

/// Allocates through tree-sitter's own allocator. Anything reachable from the
/// returned `TSLanguage` or written into a `TSWasmError` must use this, since
/// consumers free it with `ts_free`.
unsafe fn ts_malloc(size: usize) -> *mut u8 {
    unsafe {
        let f = core::ptr::addr_of!(ts_current_malloc).read();
        f(size).cast::<u8>()
    }
}

unsafe fn set_error(error: *mut TSWasmError, kind: WasmErrorKind, message: &str) {
    debug_assert!(!error.is_null());
    let bytes = message.as_bytes();
    let buffer = unsafe { ts_malloc(bytes.len() + 1) };
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), buffer, bytes.len());
        buffer.add(bytes.len()).write(0);
        (*error).kind = kind;
        (*error).message = buffer.cast::<core::ffi::c_char>();
    }
}

unsafe fn store_inner<'a>(store: *mut TSWasmStore) -> &'a mut StoreInner {
    debug_assert!(!store.is_null());
    unsafe { &mut *(store.cast::<StoreInner>()) }
}

// ---------------------------------------------------------------------------
// C layout mirrors
// ---------------------------------------------------------------------------

/// Mirror of `struct TSLanguage` (lib/src/parser.h). `#[repr(C)]` keeps the
/// field offsets identical to the C definition on every platform, which is
/// required because `parser.c` dereferences these fields directly.
#[repr(C)]
pub(crate) struct CLanguage {
    abi_version: u32,
    symbol_count: u32,
    alias_count: u32,
    token_count: u32,
    external_token_count: u32,
    state_count: u32,
    large_state_count: u32,
    production_id_count: u32,
    field_count: u32,
    max_alias_sequence_length: u16,
    parse_table: *const u16,
    small_parse_table: *const u16,
    small_parse_table_map: *const u32,
    parse_actions: *mut core::ffi::c_void,
    symbol_names: *const *const core::ffi::c_char,
    field_names: *const *const core::ffi::c_char,
    field_map_slices: *const CMapSlice,
    field_map_entries: *mut core::ffi::c_void,
    symbol_metadata: *const CSymbolMetadata,
    public_symbol_map: *const u16,
    alias_map: *const u16,
    alias_sequences: *const u16,
    lex_modes: *const CLexerMode,
    lex_fn: Option<LexFn>,
    keyword_lex_fn: Option<LexFn>,
    keyword_capture_token: u16,
    external_scanner: CExternalScanner,
    primary_state_ids: *const u16,
    name: *const core::ffi::c_char,
    reserved_words: *const u16,
    max_reserved_word_set_size: u16,
    supertype_count: u32,
    supertype_symbols: *const u16,
    supertype_map_slices: *const CMapSlice,
    supertype_map_entries: *const u16,
    metadata: CLanguageMetadata,
}

pub(crate) type LexFn = unsafe extern "C" fn(*mut TSLexer, u16) -> bool;

impl CLanguage {
    /// Equivalent of the C code's `ts_calloc(1, sizeof(TSWasmLanguage))`: all
    /// integer fields are zero, all pointers are null, all optional function
    /// pointers are `None`.
    fn zeroed() -> Self {
        unsafe { core::mem::MaybeUninit::zeroed().assume_init() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CMapSlice {
    pub index: u16,
    pub length: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CSymbolMetadata {
    pub visible: bool,
    pub named: bool,
    pub supertype: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CFieldMapEntry {
    pub field_id: u16,
    pub child_index: u8,
    pub inherited: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CLexerMode {
    pub lex_state: u16,
    pub external_lex_state: u16,
    pub reserved_word_set_id: u16,
}

#[repr(C)]
pub(crate) struct CExternalScanner {
    /// Wasm linear-memory offset of the valid-tokens table, never a native
    /// pointer (see `wasm-store-mapping.md` §3.3).
    pub states: *const bool,
    pub symbol_map: *const u16,
    pub create: Option<unsafe extern "C" fn() -> *mut core::ffi::c_void>,
    pub destroy: Option<unsafe extern "C" fn(*mut core::ffi::c_void)>,
    pub scan: Option<
        unsafe extern "C" fn(*mut core::ffi::c_void, *mut TSLexer, *const bool) -> bool,
    >,
    pub serialize:
        Option<unsafe extern "C" fn(*mut core::ffi::c_void, *mut core::ffi::c_char) -> u32>,
    pub deserialize: Option<
        unsafe extern "C" fn(*mut core::ffi::c_void, *const core::ffi::c_char, u32),
    >,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(clippy::struct_field_names)] // field names mirror the C ABI
pub(crate) struct CLanguageMetadata {
    pub major_version: u8,
    pub minor_version: u8,
    pub patch_version: u8,
}

/// Mirror of `TSLexer`'s native layout; used only to invoke the native lexer
/// function pointers from the host callbacks. The first eight bytes are the
/// copy-in/copy-out prefix shared with [`memory::TSLexerDataPrefix`].
#[repr(C)]
pub(crate) struct CLexer {
    pub lookahead: i32,
    pub result_symbol: u16,
    _padding: u16,
    pub advance: unsafe extern "C" fn(*mut TSLexer, bool),
    pub mark_end: unsafe extern "C" fn(*mut TSLexer),
    pub get_column: unsafe extern "C" fn(*mut TSLexer) -> u32,
    pub is_at_included_range_start: unsafe extern "C" fn(*mut TSLexer) -> bool,
    pub eof: unsafe extern "C" fn(*mut TSLexer) -> bool,
    pub log: *const core::ffi::c_void,
}

// ---------------------------------------------------------------------------
// Language objects
// ---------------------------------------------------------------------------

/// Shared identity of a loaded language. Instances in individual stores keep
/// `Arc` clones; `is_deleted` is a counter incremented once when the language
/// is released, mirroring the C `WasmLanguageId`.
pub(crate) struct LanguageId {
    is_deleted: AtomicU32,
}

impl LanguageId {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            is_deleted: AtomicU32::new(0),
        })
    }

    fn is_deleted(&self) -> bool {
        self.is_deleted.load(Ordering::Relaxed) != 0
    }
}

/// Mirror of `TSWasmLanguage` (`lib/src/wasm_store.c:56`). The `language` field
/// must remain the first field so a `*const TSLanguage` handed out over FFI
/// can be mapped back to the enclosing allocation with a zero offset.
///
/// `symbol_name_buffer`/`field_name_buffer` are `ts_malloc`-allocated
/// concatenations of the symbol/field name strings; the `TSLanguage`'s
/// `symbol_names`/`field_names` entries point into them, and both are freed
/// via the tree-sitter allocator in `ts_wasm_language_release`.
#[repr(C)]
pub(crate) struct WasmLanguage {
    pub language: CLanguage,
    ref_count: AtomicU32,
    language_id: Arc<LanguageId>,
    module: Module,
    name: CString,
    symbol_name_buffer: *mut u8,
    field_name_buffer: *mut u8,
    dylink_info: memory::DylinkInfo,
}

const _: () = assert!(core::mem::offset_of!(WasmLanguage, language) == 0);

/// The sentinel `lex_fn` stored on every Wasm-backed language. Pointer
/// identity with this function is what `ts_language_is_wasm` tests, so the
/// `no_mangle` is load-bearing: it keeps the symbol single despite release
/// optimization.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub extern "C" fn ts_wasm_store__sentinel_lex_fn(_lexer: *mut TSLexer, _state: u16) -> bool {
    false
}

/// Frees the `ts_malloc`-backed arrays reachable from the language. Covers
/// exactly the pointer set that the C `delete_partially_loaded_language` and
/// `ts_wasm_language_release` free.
unsafe fn free_language_data(language: &mut CLanguage) {
    let pointers: [*mut core::ffi::c_void; 20] = [
        language.alias_map.cast_mut().cast(),
        language.alias_sequences.cast_mut().cast(),
        language.external_scanner.symbol_map.cast_mut().cast(),
        language.field_map_entries,
        language.field_map_slices.cast_mut().cast(),
        language.field_names.cast_mut().cast(),
        language.lex_modes.cast_mut().cast(),
        language.name.cast_mut().cast(),
        language.parse_actions,
        language.parse_table.cast_mut().cast(),
        language.primary_state_ids.cast_mut().cast(),
        language.public_symbol_map.cast_mut().cast(),
        language.reserved_words.cast_mut().cast(),
        language.small_parse_table.cast_mut().cast(),
        language.small_parse_table_map.cast_mut().cast(),
        language.supertype_map_entries.cast_mut().cast(),
        language.supertype_map_slices.cast_mut().cast(),
        language.supertype_symbols.cast_mut().cast(),
        language.symbol_metadata.cast_mut().cast(),
        language.symbol_names.cast_mut().cast(),
    ];
    for pointer in pointers {
        unsafe { crate::ts_free(pointer) };
    }
}

// ---------------------------------------------------------------------------
// Public ABI (api.h)
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_new(
    engine: *mut TSWasmEngine,
    error: *mut TSWasmError,
) -> *mut TSWasmStore {
    match engine::store_new(engine) {
        Ok(inner) => Box::into_raw(Box::new(inner)).cast::<TSWasmStore>(),
        Err((kind, message)) => unsafe {
            set_error(error, kind, &message);
            ptr::null_mut()
        },
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_delete(store: *mut TSWasmStore) {
    if store.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(store.cast::<StoreInner>()) });
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_load_language(
    store: *mut TSWasmStore,
    name: *const core::ffi::c_char,
    wasm: *const core::ffi::c_char,
    wasm_len: u32,
    error: *mut TSWasmError,
) -> *const TSLanguage {
    let language_name = unsafe { core::ffi::CStr::from_ptr(name) };
    let bytes = unsafe { core::slice::from_raw_parts(wasm.cast::<u8>(), wasm_len as usize) };
    match module::load_language(
        unsafe { store_inner(store) },
        language_name.to_string_lossy().as_ref(),
        bytes,
    ) {
        Ok(language) => language,
        Err((kind, message)) => unsafe {
            set_error(error, kind, &message);
            ptr::null()
        },
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_language_count(store: *const TSWasmStore) -> usize {
    unsafe { store_inner(store.cast_mut()) }
        .instances
        .iter()
        .filter(|instance| !instance.language_id.is_deleted())
        .count()
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_language_is_wasm(language: *const TSLanguage) -> bool {
    let sentinel: LexFn = ts_wasm_store__sentinel_lex_fn;
    match unsafe { (*language.cast::<CLanguage>()).lex_fn } {
        Some(lex_fn) => core::ptr::fn_addr_eq(lex_fn, sentinel),
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Internal ABI (wasm_store.h), consumed by parser.c
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_start(
    store: *mut TSWasmStore,
    lexer: *mut TSLexer,
    language: *const TSLanguage,
) -> bool {
    let inner = unsafe { store_inner(store) };
    let Some(instance_index) = module::add_language(inner, language) else {
        return false;
    };
    inner.store.data_mut().current_lexer = lexer;
    inner.current_instance = Some(instance_index);
    inner.has_error = false;
    bridge::reset_heap(inner);
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_reset(store: *mut TSWasmStore) {
    let inner = unsafe { store_inner(store) };
    inner.store.data_mut().current_lexer = ptr::null_mut();
    inner.current_instance = None;
    inner.has_error = false;
    bridge::reset_heap(inner);
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_has_error(store: *const TSWasmStore) -> bool {
    unsafe { store_inner(store.cast_mut()) }.has_error
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_call_lex_main(store: *mut TSWasmStore, state: u16) -> bool {
    bridge::call_lex_function(unsafe { store_inner(store) }, LexFunction::Main, state)
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_call_lex_keyword(store: *mut TSWasmStore, state: u16) -> bool {
    bridge::call_lex_function(unsafe { store_inner(store) }, LexFunction::Keyword, state)
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_call_scanner_create(store: *mut TSWasmStore) -> u32 {
    bridge::call_scanner_create(unsafe { store_inner(store) })
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_call_scanner_destroy(
    store: *mut TSWasmStore,
    scanner_address: u32,
) {
    bridge::call_scanner_destroy(unsafe { store_inner(store) }, scanner_address);
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_call_scanner_scan(
    store: *mut TSWasmStore,
    scanner_address: u32,
    valid_tokens_ix: u32,
) -> bool {
    bridge::call_scanner_scan(
        unsafe { store_inner(store) },
        scanner_address,
        valid_tokens_ix,
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_call_scanner_serialize(
    store: *mut TSWasmStore,
    scanner_address: u32,
    buffer: *mut core::ffi::c_char,
) -> u32 {
    bridge::call_scanner_serialize(unsafe { store_inner(store) }, scanner_address, buffer)
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_store_call_scanner_deserialize(
    store: *mut TSWasmStore,
    scanner_address: u32,
    buffer: *const core::ffi::c_char,
    length: u32,
) {
    bridge::call_scanner_deserialize(
        unsafe { store_inner(store) },
        scanner_address,
        buffer,
        length,
    );
}

// ---------------------------------------------------------------------------
// Language lifetime (wasm_store.h)
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_language_retain(language: *const TSLanguage) {
    let wasm_language = unsafe { &*(language.cast::<WasmLanguage>()) };
    debug_assert!(wasm_language.ref_count.load(Ordering::SeqCst) > 0);
    wasm_language.ref_count.fetch_add(1, Ordering::SeqCst);
}

#[unsafe(no_mangle)]
pub extern "C" fn ts_wasm_language_release(language: *const TSLanguage) {
    let wasm_language = unsafe { &*(language.cast::<WasmLanguage>()) };
    debug_assert!(wasm_language.ref_count.load(Ordering::SeqCst) > 0);
    if wasm_language.ref_count.fetch_sub(1, Ordering::SeqCst) == 1 {
        wasm_language
            .language_id
            .is_deleted
            .fetch_add(1, Ordering::SeqCst);
        let mut boxed = unsafe { Box::from_raw(language.cast_mut().cast::<WasmLanguage>()) };
        unsafe {
            free_language_data(&mut boxed.language);
            crate::ts_free(boxed.symbol_name_buffer.cast());
            crate::ts_free(boxed.field_name_buffer.cast());
        }
        drop(boxed);
    }
}

// Shared with the load path; kept here so both sides use one error vocabulary.
pub(crate) type WasmResult<T> = Result<T, (WasmErrorKind, String)>;

pub(crate) const PARSE: WasmErrorKind = TSWasmErrorKindParse;
pub(crate) const COMPILE: WasmErrorKind = TSWasmErrorKindCompile;
pub(crate) const INSTANTIATE: WasmErrorKind = TSWasmErrorKindInstantiate;
pub(crate) const ALLOCATE: WasmErrorKind = TSWasmErrorKindAllocate;
