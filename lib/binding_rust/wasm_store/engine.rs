//! Wasm store lifetime: `StoreInner` (the `TSWasmStore` payload), the
//! wasmtime `Store` data shared with host callbacks, and `ts_wasm_store_new`.

use std::ptr;

use wasmtime::{
    Engine, Extern, Func, Global, GlobalType, Instance, Memory, Module, Mutability, Ref, RefType,
    Store, Table, TableType, Val, ValType,
};

use super::bridge;
use super::memory::{
    stdlib_memory_minimum, LexerInWasmMemory, MAX_MEMORY_PAGES, MEMORY_PAGE_SIZE,
};
use super::stdlib::{stdlib_symbols, STDLIB_WASM};
use super::{ALLOCATE, COMPILE, INSTANTIATE, TSWasmEngine, WasmResult};

/// A language instantiation within one store: the shared identity, the live
/// wasmtime instance, and the seven function handles cached from the shared
/// function table (zero table lookups on the hot path). Addresses are wasm
/// linear-memory offsets, immune to memory.grow.
pub(crate) struct LanguageWasmInstance {
    pub language_id: std::sync::Arc<super::LanguageId>,
    #[allow(dead_code)] // kept alive so the cached functions remain valid
    pub instance: Instance,
    pub external_states_address: i32,
    pub lex_main_fn: Option<Func>,
    pub lex_keyword_fn: Option<Func>,
    pub scanner_create_fn: Option<Func>,
    pub scanner_destroy_fn: Option<Func>,
    pub scanner_serialize_fn: Option<Func>,
    pub scanner_deserialize_fn: Option<Func>,
    pub scanner_scan_fn: Option<Func>,
}

/// Host functions provided to modules (`wasm_store.c`'s `BuiltinFunctionIndices`).
/// Everything except `reset_heap` is created eagerly; `reset_heap` is resolved
/// from the stdlib module's exports.
pub(crate) struct BuiltinFunctions {
    pub reset_heap: Option<Func>,
    pub proc_exit: Func,
    pub abort: Func,
    pub assert_fail: Func,
    pub notify_memory_growth: Func,
    pub debug_message: Func,
    pub at_exit: Func,
    pub args_get: Func,
    pub args_sizes_get: Func,
}

/// The wasmtime `Store` data: only what host callbacks need while Wasm code
/// is on the stack. The memory and table are filled in by `store_new` before
/// any module is instantiated.
pub(crate) struct State {
    memory: Option<Memory>,
    table: Option<Table>,
    pub current_lexer: *mut super::TSLexer,
    pub lexer_address: u32,
}

impl State {
    fn empty() -> Self {
        Self {
            memory: None,
            table: None,
            current_lexer: ptr::null_mut(),
            lexer_address: 0,
        }
    }

    pub fn memory(&self) -> &Memory {
        self.memory.as_ref().expect("store memory not initialized")
    }

    pub fn table(&self) -> &Table {
        self.table.as_ref().expect("store table not initialized")
    }
}

/// The payload behind an opaque `TSWasmStore *`. Owns the wasmtime store (and
/// through it all instances, functions, the linear memory and the table), the
/// cloned engine reference, and the per-language instantiation records.
///
/// Mutable and not thread-safe by contract (one parser at a time), exactly
/// like the C struct; `wasm_language.rs` opts into `Send`/`Sync` for the
/// outer `WasmStore` handle.
pub(crate) struct StoreInner {
    pub(crate) engine: Engine,
    pub store: Store<State>,
    pub instances: Vec<LanguageWasmInstance>,
    /// Handles of the stdlib exports, aligned with `stdlib_symbols()`.
    pub stdlib_fns: Vec<Option<Func>>,
    pub builtin_fns: BuiltinFunctions,
    /// Created fresh by `store_new`, replaced by the stdlib's exported
    /// `__stack_pointer` global once the stdlib is instantiated.
    pub stack_pointer_global: Global,
    /// Placement offset for the next module's static data (`__memory_base`).
    pub current_memory_offset: u32,
    /// Placement offset for the next module's table entries (`__table_base`).
    pub current_function_table_offset: u32,
    /// Index into `instances` of the instance registered by
    /// `ts_wasm_store_start`. An index, never a pointer: `instances` may
    /// reallocate (`wasm_store.c` stored a raw pointer and relied on call-order
    /// discipline).
    pub current_instance: Option<usize>,
    pub has_error: bool,
}

impl StoreInner {
    pub fn lexer_address(&self) -> u32 {
        self.store.data().lexer_address
    }

    pub fn current_instance(&self) -> &LanguageWasmInstance {
        let index = self
            .current_instance
            .expect("no Wasm language is active in this store");
        &self.instances[index]
    }

    /// Calls a function through the unchecked path. Errors and traps are both
    /// swallowed into `has_error`, matching `ts_wasm_store__call`.
    pub fn call_unchecked(&mut self, func: &Func, args: &mut [wasmtime::ValRaw]) {
        // SAFETY: every call site passes exactly the argument/return count of
        // the callee's wasm signature, all i32s, with args[0] doubling as the
        // result slot.
        match unsafe { func.call_unchecked(&mut self.store, args) } {
            Ok(()) => {}
            Err(_) => self.has_error = true,
        }
    }

    /// Fetches a funcref from the shared function table by index. Missing
    /// entries (e.g. a grammar without external scanners) yield `None`,
    /// mirroring the C code's zeroed `wasmtime_func_t` sentinel.
    pub fn get_function(&mut self, function_index: i32) -> Option<Func> {
        let table: Table = *self.store.data().table();
        match table.get(&mut self.store, function_index as u64) {
            Some(Ref::Func(func)) => func,
            _ => panic!("invalid function table entry {function_index}"),
        }
    }

    pub fn heap_address(&self) -> u32 {
        super::memory::heap_address(self.current_memory_offset)
    }
}

/// Compile the embedded stdlib, create the store's memory/table/globals, and
/// instantiate the stdlib. On success the store is ready to load grammar
/// modules.
pub(crate) fn store_new(engine: *mut TSWasmEngine) -> WasmResult<StoreInner> {
    // The public API passes `&wasmtime::Engine` cast to `*mut TSWasmEngine`;
    // `TSWasmEngine`'s only payload is that engine (formerly the c-api's
    // `wasm_engine_t`).
    let engine: Engine = unsafe { (*engine.cast::<Engine>()).clone() };

    let mut store = Store::new(&engine, State::empty());

    let lexer_fns = bridge::create_lexer_functions(&engine, &mut store);
    let builtin_fns = bridge::create_builtin_functions(&engine, &mut store);

    let stdlib_module = Module::new(&engine, STDLIB_WASM)
        .map_err(|error| (COMPILE, format!("failed to compile Wasm stdlib: {error}")))?;

    let Some(initial_memory_pages) = stdlib_memory_minimum(&stdlib_module) else {
        return Err((
            COMPILE,
            "Wasm stdlib is missing the 'memory' import".to_string(),
        ));
    };

    let memory = Memory::new(
        &mut store,
        wasmtime::MemoryType::new(initial_memory_pages as u32, Some(MAX_MEMORY_PAGES)),
    )
    .map_err(|error| (ALLOCATE, format!("failed to allocate Wasm memory: {error}")))?;

    let table = Table::new(
        &mut store,
        TableType::new(RefType::FUNCREF, 1, Some(u32::MAX)),
        Ref::Func(None),
    )
    .map_err(|error| (ALLOCATE, format!("failed to allocate Wasm table: {error}")))?;

    let stack_pointer_global = Global::new(
        &mut store,
        GlobalType::new(ValType::I32, Mutability::Var),
        Val::I32(0),
    )
    .expect("failed to create the initial stack pointer global");

    {
        let state = store.data_mut();
        state.memory = Some(memory);
        state.table = Some(table);
    }

    let mut inner = StoreInner {
        engine,
        store,
        instances: Vec::new(),
        stdlib_fns: vec![None; stdlib_symbols().len()],
        builtin_fns: BuiltinFunctions {
            reset_heap: None,
            proc_exit: builtin_fns.proc_exit,
            abort: builtin_fns.abort,
            assert_fail: builtin_fns.assert_fail,
            notify_memory_growth: builtin_fns.notify_memory_growth,
            debug_message: builtin_fns.debug_message,
            at_exit: builtin_fns.at_exit,
            args_get: builtin_fns.args_get,
            args_sizes_get: builtin_fns.args_sizes_get,
        },
        stack_pointer_global,
        current_memory_offset: 0,
        current_function_table_offset: 0,
        current_instance: None,
        has_error: false,
    };

    let mut imports = Vec::new();
    for import in stdlib_module.imports() {
        let Some(import_extern) = bridge::provide_builtin_import(&mut inner, import.name())
        else {
            return Err((
                INSTANTIATE,
                format!("unexpected import in Wasm stdlib: {}\n", import.name()),
            ));
        };
        imports.push(import_extern);
    }

    let instance = Instance::new(&mut inner.store, &stdlib_module, &imports).map_err(
        |error| {
            (
                INSTANTIATE,
                format!("failed to instantiate Wasm stdlib module: {error}"),
            )
        },
    )?;

    // Register the stdlib's exports: the stack pointer global, the reset_heap
    // entry point, and every whitelisted libc symbol.
    for export_type in stdlib_module.exports() {
        let name = export_type.name();
        let Some(export) = instance.get_export(&mut inner.store, name) else {
            continue;
        };
        match export {
            Extern::Global(global) => {
                if name == "__stack_pointer" {
                    inner.stack_pointer_global = global;
                }
            }
            Extern::Func(func) => {
                if super::module::is_module_initializer(name) {
                    func.call(&mut inner.store, &[], &mut []).map_err(|error| {
                        (
                            INSTANTIATE,
                            format!("trap when calling stdlib relocation function: {error}\n"),
                        )
                    })?;
                    continue;
                }
                if name == "reset_heap" {
                    inner.builtin_fns.reset_heap = Some(func);
                    continue;
                }
                if let Some(index) = stdlib_symbols().iter().position(|symbol| *symbol == name) {
                    inner.stdlib_fns[index] = Some(func);
                }
            }
            _ => {}
        }
    }

    if inner.builtin_fns.reset_heap.is_none() {
        return Err((
            INSTANTIATE,
            "missing malloc reset function in Wasm stdlib".to_string(),
        ));
    }
    for (index, symbol) in stdlib_symbols().iter().enumerate() {
        if inner.stdlib_fns[index].is_none() {
            return Err((
                INSTANTIATE,
                format!("missing exported symbol in Wasm stdlib: {symbol}"),
            ));
        }
    }

    // Add the lexer callback functions to the shared function table and write
    // their indices into the in-memory lexer image.
    let table_base = table
        .grow(&mut inner.store, lexer_fns.len() as u64, Ref::Func(None))
        .map_err(|error| {
            (
                ALLOCATE,
                format!("failed to grow Wasm table to initial size: {error}"),
            )
        })?;

    // Zero-initialize the whole image: its 28 bytes are raw-copied into wasm
    // memory, and the two padding bytes after `result_symbol` must not carry
    // undefined values.
    let mut in_memory_lexer: LexerInWasmMemory =
        unsafe { core::mem::MaybeUninit::zeroed().assume_init() };
    for (index, func) in lexer_fns.iter().enumerate() {
        table
            .set(&mut inner.store, table_base + index as u64, Ref::Func(Some(*func)))
            .expect("failed to store lexer function in function table");
    }
    in_memory_lexer.advance = table_base as i32;
    in_memory_lexer.mark_end = (table_base + 1) as i32;
    in_memory_lexer.get_column = (table_base + 2) as i32;
    in_memory_lexer.is_at_included_range_start = (table_base + 3) as i32;
    in_memory_lexer.eof = (table_base + 4) as i32;

    inner.current_function_table_offset = (table_base + lexer_fns.len() as u64) as u32;
    let lexer_address = initial_memory_pages as u32 * MEMORY_PAGE_SIZE;
    inner.current_memory_offset = lexer_address + core::mem::size_of::<LexerInWasmMemory>() as u32;
    inner.store.data_mut().lexer_address = lexer_address;

    // Grow the memory to hold the in-memory lexer and serialization buffer.
    // The C implementation ignores growth failures here as well; subsequent
    // loads grow on demand and report allocation errors.
    let pages_needed =
        (inner.current_memory_offset - lexer_address - 1) / MEMORY_PAGE_SIZE + 1;
    let _ = memory.grow(&mut inner.store, u64::from(pages_needed));

    let data = memory.data_mut(&mut inner.store);
    let lexer_bytes = unsafe {
        core::slice::from_raw_parts(
            ptr::addr_of!(in_memory_lexer).cast::<u8>(),
            core::mem::size_of::<LexerInWasmMemory>(),
        )
    };
    let range = lexer_address as usize
        ..lexer_address as usize + core::mem::size_of::<LexerInWasmMemory>();
    data[range].copy_from_slice(lexer_bytes);

    Ok(inner)
}
