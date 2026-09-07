//! Host functions exposed to Wasm modules and the parse hot path.
//!
//! The five lexer callbacks and the builtin callbacks (abort/noop/debug)
//! mirror `lib/src/wasm_store.c:294-396` and `:719-790`; the hot-path entries
//! mirror `:1800-2015`. Every callback retrieves store state through
//! `Caller::data_mut` instead of the C code's `env` pointer, and linear
//! memory slices are re-derived after any Wasm call so a wasm-side
//! `memory.grow` can never leave a stale base pointer behind.

#![allow(clippy::unnecessary_wraps, clippy::needless_pass_by_value)] // callback signatures are fixed by wasmtime

use core::ffi::c_char;
use std::mem::MaybeUninit;
use std::ptr;

use wasmtime::{
    Caller, Engine, Error, Extern, Func, FuncType, Global, GlobalType, Memory, Mutability,
    Store, Val, ValRaw, ValType,
};

use super::engine::{BuiltinFunctions, State, StoreInner};
use super::memory::{
    serialization_buffer_address, LEXER_DATA_PREFIX_SIZE, SERIALIZATION_BUFFER_SIZE,
};
use super::{CLexer, TSLexer};

// ---------------------------------------------------------------------------
// Native callbacks exposed to Wasm modules
// ---------------------------------------------------------------------------

fn callback_abort(
    _caller: Caller<'_, State>,
    _args_and_results: &mut [MaybeUninit<ValRaw>],
) -> wasmtime::Result<()> {
    Err(Error::msg("Wasm module called abort"))
}

fn callback_noop(
    _caller: Caller<'_, State>,
    _args_and_results: &mut [MaybeUninit<ValRaw>],
) -> wasmtime::Result<()> {
    Ok(())
}

fn callback_debug_message(
    caller: Caller<'_, State>,
    args_and_results: &mut [MaybeUninit<ValRaw>],
) -> wasmtime::Result<()> {
    assert_eq!(args_and_results.len(), 2);
    let string_address = unsafe { args_and_results[0].assume_init().get_i32() };
    let value = unsafe { args_and_results[1].assume_init().get_i32() } as u32;
    let memory: Memory = *caller.data().memory();
    let data = memory.data(&caller);
    let message = if string_address >= 0 && string_address as usize <= data.len() {
        let start = string_address as usize;
        let end = start + data[start..].iter().position(|&byte| byte == 0).unwrap_or(0);
        String::from_utf8_lossy(&data[start..end]).into_owned()
    } else {
        String::new()
    };
    println!("DEBUG: {message} {value}");
    Ok(())
}

fn callback_lexer_advance(
    mut caller: Caller<'_, State>,
    args_and_results: &mut [MaybeUninit<ValRaw>],
) -> wasmtime::Result<()> {
    assert_eq!(args_and_results.len(), 2);
    let state = caller.data_mut();
    let lexer = state.current_lexer;
    let memory: Memory = *state.memory();
    let skip = unsafe { args_and_results[1].assume_init().get_i32() } != 0;

    unsafe {
        ((*lexer.cast::<CLexer>()).advance)(lexer, skip);
        // The native lexer advanced the input; publish the new lookahead into
        // Wasm memory (only the four lookahead bytes, never result_symbol).
        let lookahead = (lexer as *const i32).read();
        let address = state.lexer_address as usize;
        memory.data_mut(&mut caller)[address..address + 4].copy_from_slice(&lookahead.to_ne_bytes());
    }
    Ok(())
}

fn callback_lexer_mark_end(
    caller: Caller<'_, State>,
    _args_and_results: &mut [MaybeUninit<ValRaw>],
) -> wasmtime::Result<()> {
    let lexer = caller.data().current_lexer;
    unsafe { ((*lexer.cast::<CLexer>()).mark_end)(lexer) };
    Ok(())
}

fn callback_lexer_get_column(
    caller: Caller<'_, State>,
    args_and_results: &mut [MaybeUninit<ValRaw>],
) -> wasmtime::Result<()> {
    let lexer = caller.data().current_lexer;
    let result = unsafe { ((*lexer.cast::<CLexer>()).get_column)(lexer) };
    args_and_results[0] = MaybeUninit::new(ValRaw::i32(result as i32));
    Ok(())
}

fn callback_lexer_is_at_included_range_start(
    caller: Caller<'_, State>,
    args_and_results: &mut [MaybeUninit<ValRaw>],
) -> wasmtime::Result<()> {
    let lexer = caller.data().current_lexer;
    let result = unsafe { ((*lexer.cast::<CLexer>()).is_at_included_range_start)(lexer) };
    args_and_results[0] = MaybeUninit::new(ValRaw::i32(i32::from(result)));
    Ok(())
}

fn callback_lexer_eof(
    caller: Caller<'_, State>,
    args_and_results: &mut [MaybeUninit<ValRaw>],
) -> wasmtime::Result<()> {
    let lexer = caller.data().current_lexer;
    let result = unsafe { ((*lexer.cast::<CLexer>()).eof)(lexer) };
    args_and_results[0] = MaybeUninit::new(ValRaw::i32(i32::from(result)));
    Ok(())
}

// ---------------------------------------------------------------------------
// Host function creation
// ---------------------------------------------------------------------------

/// Creates the five lexer callbacks. Order must match the `LexerInWasmMemory`
/// function-slot order: `advance`, `mark_end`, `get_column`,
/// `is_at_included_range_start`, `eof`.
pub(crate) fn create_lexer_functions(engine: &Engine, store: &mut Store<State>) -> [Func; 5] {
    let i32_ty = ValType::I32;
    unsafe {
        [
            Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [i32_ty.clone(), i32_ty.clone()], []),
                callback_lexer_advance,
            ),
            Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [i32_ty.clone()], []),
                callback_lexer_mark_end,
            ),
            Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [i32_ty.clone()], [i32_ty.clone()]),
                callback_lexer_get_column,
            ),
            Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [i32_ty.clone()], [i32_ty.clone()]),
                callback_lexer_is_at_included_range_start,
            ),
            Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [i32_ty.clone()], [i32_ty]),
                callback_lexer_eof,
            ),
        ]
    }
}

/// Creates the builtin functions that modules may import: abort family
/// (`proc_exit`, `abort`, `__assert_fail`), no-ops (`__cxa_atexit`,
/// `args_get`, `args_sizes_get`, `emscripten_notify_memory_growth`), and the
/// debug message hook.
pub(crate) fn create_builtin_functions(engine: &Engine, store: &mut Store<State>) -> BuiltinFunctions {
    let i32_ty = ValType::I32;
    unsafe {
        BuiltinFunctions {
            reset_heap: None,
            proc_exit: Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [i32_ty.clone()], []),
                callback_abort,
            ),
            abort: Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [], []),
                callback_abort,
            ),
            assert_fail: Func::new_unchecked(
                &mut *store,
                FuncType::new(
                    engine,
                    [
                        i32_ty.clone(),
                        i32_ty.clone(),
                        i32_ty.clone(),
                        i32_ty.clone(),
                    ],
                    [],
                ),
                callback_abort,
            ),
            notify_memory_growth: Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [i32_ty.clone()], []),
                callback_noop,
            ),
            debug_message: Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [i32_ty.clone(), i32_ty.clone()], []),
                callback_debug_message,
            ),
            at_exit: Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [i32_ty.clone(), i32_ty.clone(), i32_ty.clone()], [i32_ty.clone()]),
                callback_noop,
            ),
            args_get: Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [i32_ty.clone(), i32_ty.clone()], [i32_ty.clone()]),
                callback_noop,
            ),
            args_sizes_get: Func::new_unchecked(
                &mut *store,
                FuncType::new(engine, [i32_ty.clone(), i32_ty.clone()], [i32_ty]),
                callback_noop,
            ),
        }
    }
}

/// Provides the dynamic-linking imports and builtin callbacks for a module.
/// `__memory_base`/`__table_base` are freshly created const globals per
/// instantiation (their values advance with each module placement).
pub(crate) fn provide_builtin_import(inner: &mut StoreInner, name: &str) -> Option<Extern> {
    match name {
        "__memory_base" => {
            let value = inner.current_memory_offset as i32;
            let global = Global::new(
                &mut inner.store,
                GlobalType::new(ValType::I32, Mutability::Const),
                Val::I32(value),
            )
            .expect("failed to create __memory_base global");
            Some(Extern::Global(global))
        }
        "__table_base" => {
            let value = inner.current_function_table_offset as i32;
            let global = Global::new(
                &mut inner.store,
                GlobalType::new(ValType::I32, Mutability::Const),
                Val::I32(value),
            )
            .expect("failed to create __table_base global");
            Some(Extern::Global(global))
        }
        "__stack_pointer" => Some(Extern::Global(inner.stack_pointer_global)),
        "__indirect_function_table" => Some(Extern::Table(*inner.store.data().table())),
        "memory" => Some(Extern::Memory(*inner.store.data().memory())),
        "__assert_fail" => Some(Extern::Func(inner.builtin_fns.assert_fail)),
        "__cxa_atexit" => Some(Extern::Func(inner.builtin_fns.at_exit)),
        "args_get" => Some(Extern::Func(inner.builtin_fns.args_get)),
        "args_sizes_get" => Some(Extern::Func(inner.builtin_fns.args_sizes_get)),
        "abort" => Some(Extern::Func(inner.builtin_fns.abort)),
        "proc_exit" => Some(Extern::Func(inner.builtin_fns.proc_exit)),
        "emscripten_notify_memory_growth" => {
            Some(Extern::Func(inner.builtin_fns.notify_memory_growth))
        }
        "tree_sitter_debug_message" => Some(Extern::Func(inner.builtin_fns.debug_message)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Hot path
// ---------------------------------------------------------------------------

/// Calls the stdlib's `reset_heap` with the current scanner heap address.
/// Called by `ts_wasm_store_start`/`ts_wasm_store_reset` so the scanner heap
/// is reset per parse.
pub(crate) fn reset_heap(inner: &mut StoreInner) {
    let func = inner
        .builtin_fns
        .reset_heap
        .expect("stdlib reset_heap function missing");
    let address = Val::I32(inner.heap_address() as i32);
    func.call(&mut inner.store, &[address], &mut [])
        .expect("reset_heap call failed");
}

pub(crate) enum LexFunction {
    Main,
    Keyword,
}

pub(crate) fn call_lex_function(inner: &mut StoreInner, which: LexFunction, state: u16) -> bool {
    let func = match which {
        LexFunction::Main => inner.current_instance().lex_main_fn,
        LexFunction::Keyword => inner.current_instance().lex_keyword_fn,
    }
    .expect("lex function handle missing");
    let lexer = inner.store.data().current_lexer;
    let memory: Memory = *inner.store.data().memory();
    let lexer_address = inner.lexer_address();

    let mut args = [ValRaw::i32(lexer_address as i32), ValRaw::i32(i32::from(state))];
    copy_prefix(
        &memory,
        &mut inner.store,
        lexer_address,
        CopyDirection::IntoWasm(lexer),
    );
    inner.call_unchecked(&func, &mut args);
    if inner.has_error {
        return false;
    }
    let result = args[0].get_i32();
    copy_prefix(
        &memory,
        &mut inner.store,
        lexer_address,
        CopyDirection::OutOfWasm(lexer),
    );
    result != 0
}

pub(crate) fn call_scanner_create(inner: &mut StoreInner) -> u32 {
    let func = inner
        .current_instance()
        .scanner_create_fn
        .expect("scanner create function handle missing");
    let mut args = [ValRaw::i32(0)];
    inner.call_unchecked(&func, &mut args);
    if inner.has_error {
        return 0;
    }
    args[0].get_i32() as u32
}

pub(crate) fn call_scanner_destroy(inner: &mut StoreInner, scanner_address: u32) {
    if inner.current_instance.is_some() {
        let func = inner
            .current_instance()
            .scanner_destroy_fn
            .expect("scanner destroy function handle missing");
        let mut args = [ValRaw::i32(scanner_address as i32)];
        inner.call_unchecked(&func, &mut args);
    }
}

pub(crate) fn call_scanner_scan(
    inner: &mut StoreInner,
    scanner_address: u32,
    valid_tokens_ix: u32,
) -> bool {
    let func = inner
        .current_instance()
        .scanner_scan_fn
        .expect("scanner scan function handle missing");
    let external_states_address = inner.current_instance().external_states_address;
    let lexer = inner.store.data().current_lexer;
    let memory: Memory = *inner.store.data().memory();
    let lexer_address = inner.lexer_address();
    let valid_tokens_address = external_states_address.wrapping_add(valid_tokens_ix as i32);

    copy_prefix(
        &memory,
        &mut inner.store,
        lexer_address,
        CopyDirection::IntoWasm(lexer),
    );
    let mut args = [
        ValRaw::i32(scanner_address as i32),
        ValRaw::i32(lexer_address as i32),
        ValRaw::i32(valid_tokens_address),
    ];
    inner.call_unchecked(&func, &mut args);
    if inner.has_error {
        return false;
    }
    let result = args[0].get_i32();
    copy_prefix(
        &memory,
        &mut inner.store,
        lexer_address,
        CopyDirection::OutOfWasm(lexer),
    );
    result != 0
}

pub(crate) fn call_scanner_serialize(
    inner: &mut StoreInner,
    scanner_address: u32,
    buffer: *mut c_char,
) -> u32 {
    let func = inner
        .current_instance()
        .scanner_serialize_fn
        .expect("scanner serialize function handle missing");
    let memory: Memory = *inner.store.data().memory();
    let serialization_address = serialization_buffer_address(inner.current_memory_offset);

    let mut args = [
        ValRaw::i32(scanner_address as i32),
        ValRaw::i32(serialization_address as i32),
    ];
    inner.call_unchecked(&func, &mut args);
    if inner.has_error {
        return 0;
    }
    let length = args[0].get_i32() as u32;
    if length > SERIALIZATION_BUFFER_SIZE as u32 {
        inner.has_error = true;
        return 0;
    }

    if length > 0 {
        // Re-derive the slice after the call: the serializer may have grown
        // the linear memory from within Wasm.
        let data = memory.data(&inner.store);
        let bytes = &data[serialization_address as usize..][..length as usize];
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), length as usize) };
    }
    length
}

pub(crate) fn call_scanner_deserialize(
    inner: &mut StoreInner,
    scanner_address: u32,
    buffer: *const c_char,
    length: u32,
) {
    let func = inner
        .current_instance()
        .scanner_deserialize_fn
        .expect("scanner deserialize function handle missing");
    let memory: Memory = *inner.store.data().memory();
    let serialization_address = serialization_buffer_address(inner.current_memory_offset);

    if length > 0 {
        let bytes = unsafe { core::slice::from_raw_parts(buffer.cast::<u8>(), length as usize) };
        let data = memory.data_mut(&mut inner.store);
        data[serialization_address as usize..][..length as usize].copy_from_slice(bytes);
    }

    let mut args = [
        ValRaw::i32(scanner_address as i32),
        ValRaw::i32(serialization_address as i32),
        ValRaw::i32(length as i32),
    ];
    inner.call_unchecked(&func, &mut args);
}

enum CopyDirection {
    IntoWasm(*mut TSLexer),
    OutOfWasm(*mut TSLexer),
}

/// Copies the eight-byte `TSLexer` data prefix to or from the in-memory lexer
/// image. `memory` is re-consulted at copy time (fresh base), never cached
/// across a Wasm call.
fn copy_prefix(
    memory: &Memory,
    store: &mut Store<State>,
    lexer_address: u32,
    direction: CopyDirection,
) {
    let address = lexer_address as usize;
    match direction {
        CopyDirection::IntoWasm(lexer) => unsafe {
            let prefix = core::slice::from_raw_parts(lexer.cast::<u8>(), LEXER_DATA_PREFIX_SIZE);
            memory.data_mut(store)[address..][..LEXER_DATA_PREFIX_SIZE].copy_from_slice(prefix);
        },
        CopyDirection::OutOfWasm(lexer) => unsafe {
            let source = &memory.data(&*store)[address..][..LEXER_DATA_PREFIX_SIZE];
            ptr::copy_nonoverlapping(source.as_ptr(), lexer.cast::<u8>(), LEXER_DATA_PREFIX_SIZE);
        },
    }
}
