//! Grammar module compilation, instantiation, and the static-data copy-out
//! from Wasm linear memory onto the native heap.

use std::ffi::CString;
use std::ptr;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;

use wasmtime::{Extern, Instance, Memory, Module, Ref, Val};

use super::bridge;
use super::engine::{LanguageWasmInstance, StoreInner};
use super::memory::{
    dylink_info_parse, heap_address, DylinkInfo, LanguageInWasmMemory, WasmMemoryView,
    MEMORY_PAGE_SIZE,
};
use super::stdlib::stdlib_symbols;
use super::{
    free_language_data, ts_malloc, CLanguage, CFieldMapEntry, CLexerMode, CMapSlice,
    CSymbolMetadata, LanguageId, TSLanguage, WasmLanguage, WasmResult, COMPILE, INSTANTIATE,
};
use crate::ffi::{
    TREE_SITTER_LANGUAGE_VERSION, TREE_SITTER_MIN_COMPATIBLE_LANGUAGE_VERSION,
};

/// From `lib/src/language.h`; gates fields newer grammar ABIs carry.
const LANGUAGE_VERSION_WITH_PRIMARY_STATES: u32 = 14;
const LANGUAGE_VERSION_WITH_RESERVED_WORDS: u32 = 15;

pub(crate) fn is_module_initializer(name: &str) -> bool {
    matches!(
        name,
        "_initialize" | "__wasm_apply_data_relocs" | "__wasm_call_ctors"
    )
}

fn invalid_language_memory() -> (u32, String) {
    (INSTANTIATE, "invalid language memory address".to_string())
}

// ---------------------------------------------------------------------------
// Static-data copy helpers (wasm_store.c:409-538)
// ---------------------------------------------------------------------------

/// Copies a sized array out of Wasm memory into a tree-sitter-allocated
/// buffer. No-op (without failing) once `ok` is false or the size is zero.
unsafe fn copy(view: &WasmMemoryView, address: i32, size: usize, ok: &mut bool) -> *mut u8 {
    if !*ok || size == 0 {
        return ptr::null_mut();
    }
    if !view.contains(address, size) {
        *ok = false;
        return ptr::null_mut();
    }
    let result = unsafe { ts_malloc(size) };
    unsafe { ptr::copy_nonoverlapping(view.data[address as usize..].as_ptr(), result, size) };
    result
}

/// Copies an array whose length is not recorded in the language struct: the
/// end is inferred as the next-highest address among the sentinel list
/// (`addresses` includes the language address and the next module's placement
/// offset).
unsafe fn copy_unsized_static_array(
    view: &WasmMemoryView,
    start_address: i32,
    addresses: &[i32],
    ok: &mut bool,
) -> *mut u8 {
    if !*ok || start_address == 0 {
        return ptr::null_mut();
    }
    if start_address < 0 {
        *ok = false;
        return ptr::null_mut();
    }
    let mut end_address = 0;
    for &address in addresses {
        if address > start_address && (end_address == 0 || address < end_address) {
            end_address = address;
        }
    }
    if end_address == 0 {
        return ptr::null_mut();
    }
    unsafe { copy(view, start_address, (end_address - start_address) as usize, ok) }
}

/// Copies a NUL-terminated string, including the terminator.
unsafe fn copy_string(view: &WasmMemoryView, address: i32, ok: &mut bool) -> *mut u8 {
    if !*ok {
        return ptr::null_mut();
    }
    let Some(length) = view.string_length(address) else {
        *ok = false;
        return ptr::null_mut();
    };
    unsafe { copy(view, address, length + 1, ok) }
}

/// A `char **` array copied out of Wasm memory. The pointer array is
/// allocated with tree-sitter's allocator (freed via the language free list);
/// string bytes accumulate in `scratch` and are moved into an exact-size
/// `ts_malloc`-allocated buffer by [`finalize_string_table`], after which the
/// slots are rewritten into real pointers.
struct StringTable {
    array: *mut usize,
    count: usize,
    scratch: Vec<u8>,
}

unsafe fn copy_strings(
    view: &WasmMemoryView,
    array_address: i32,
    count: usize,
    ok: &mut bool,
) -> Option<StringTable> {
    if !*ok {
        return None;
    }
    if count > usize::MAX / core::mem::size_of::<usize>()
        || !view.contains(array_address, count * core::mem::size_of::<i32>())
    {
        *ok = false;
        return None;
    }

    // ts_malloc wraps the platform allocator, which yields max-aligned
    // memory, so the usize array cast is sound.
    #[allow(clippy::cast_ptr_alignment)]
    let array = unsafe { ts_malloc(count * core::mem::size_of::<usize>()) }.cast::<usize>();
    let mut scratch = Vec::new();
    for index in 0..count {
        let address = i32::from_ne_bytes(
            view.data[array_address as usize + index * 4..][..4]
                .try_into()
                .expect("four bytes for an i32"),
        );
        let slot = if address == 0 {
            usize::MAX
        } else {
            let Some(length) = view.string_length(address) else {
                unsafe { crate::ts_free(array.cast()) };
                *ok = false;
                return None;
            };
            if length > u32::MAX as usize {
                unsafe { crate::ts_free(array.cast()) };
                *ok = false;
                return None;
            }
            let offset = scratch.len();
            scratch.extend_from_slice(&view.data[address as usize..=address as usize + length]);
            offset
        };
        unsafe { array.add(index).write(slot) };
    }
    Some(StringTable {
        array,
        count,
        scratch,
    })
}

/// Moves the accumulated string bytes into an exact-size tree-sitter-
/// allocated buffer (freed by `ts_wasm_language_release` alongside the
/// pointer array) and rewrites the offset slots into real pointers against
/// it. Returns the buffer pointer.
unsafe fn finalize_string_table(StringTable { array, count, scratch }: StringTable) -> *mut u8 {
    let buffer = unsafe { ts_malloc(scratch.len()) };
    if !scratch.is_empty() {
        unsafe {
            ptr::copy_nonoverlapping(scratch.as_ptr(), buffer, scratch.len());
        }
    }
    for index in 0..count {
        let entry = unsafe { array.add(index) };
        let value = unsafe { entry.read() };
        let relocated = if value == usize::MAX {
            ptr::null_mut::<u8>()
        } else {
            unsafe { buffer.add(value) }
        };
        unsafe { entry.write(relocated as usize) };
    }
    buffer
}

// ---------------------------------------------------------------------------
// Instantiation (wasm_store.c:1122-1313)
// ---------------------------------------------------------------------------

/// Instantiates a grammar module in this store: grows the shared table and
/// linear memory, wires up the dylink and builtin imports, runs the module
/// initializers, and returns the instance plus the address of its
/// `TSLanguage` object in linear memory. Offsets are rolled back on failure;
/// already-grown pages stay available for later modules.
fn instantiate(
    inner: &mut StoreInner,
    module: &Module,
    language_name: &str,
    dylink_info: &DylinkInfo,
) -> WasmResult<(Instance, i32)> {
    let initial_memory_offset = inner.current_memory_offset;
    let initial_function_table_offset = inner.current_function_table_offset;

    let result = instantiate_inner(inner, module, language_name, dylink_info);
    if result.is_err() {
        inner.current_memory_offset = initial_memory_offset;
        inner.current_function_table_offset = initial_function_table_offset;
    }
    result
}

fn instantiate_inner(
    inner: &mut StoreInner,
    module: &Module,
    language_name: &str,
    dylink_info: &DylinkInfo,
) -> WasmResult<(Instance, i32)> {
    let table: wasmtime::Table = *inner.store.data().table();
    let memory: Memory = *inner.store.data().memory();

    table
        .grow(&mut inner.store, u64::from(dylink_info.table_size), Ref::Func(None))
        .map_err(|_| {
            (
                INSTANTIATE,
                format!("invalid function table size {}", dylink_info.table_size),
            )
        })?;

    let needed_memory_size = heap_address(inner.current_memory_offset) + dylink_info.memory_size;
    let current_memory_size = memory.data_size(&inner.store) as u32;
    if needed_memory_size > current_memory_size {
        let pages_to_grow = (needed_memory_size - current_memory_size).div_ceil(MEMORY_PAGE_SIZE);
        memory
            .grow(&mut inner.store, u64::from(pages_to_grow))
            .map_err(|_| {
                (
                    INSTANTIATE,
                    format!("invalid memory size {}", dylink_info.memory_size),
                )
            })?;
    }

    let language_function_name = format!("tree_sitter_{language_name}");

    let mut imports = Vec::new();
    for import in module.imports() {
        let name = import.name();
        if name.is_empty() {
            return Err((INSTANTIATE, "empty import name".to_string()));
        }
        if let Some(import_extern) = bridge::provide_builtin_import(inner, name) {
            imports.push(import_extern);
            continue;
        }
        if let Some(index) = stdlib_symbols().iter().position(|symbol| *symbol == name) {
            // All stdlib symbols are resolved during store creation.
            let func = inner.stdlib_fns[index].expect("stdlib symbol");
            imports.push(Extern::Func(func));
            continue;
        }
        return Err((INSTANTIATE, format!("invalid import '{name}'\n")));
    }

    let instance = Instance::new(&mut inner.store, module, &imports).map_err(|error| {
        (
            INSTANTIATE,
            format!("error instantiating Wasm module: {error}\n"),
        )
    })?;

    inner.current_memory_offset += dylink_info.memory_size;
    inner.current_function_table_offset += dylink_info.table_size;

    let mut language_func = None;
    for export_type in module.exports() {
        let name = export_type.name();
        if is_module_initializer(name) {
            let Some(Extern::Func(func)) = instance.get_export(&mut inner.store, name) else {
                continue;
            };
            if let Err(error) = func.call(&mut inner.store, &[], &mut []) {
                return Err((
                    INSTANTIATE,
                    format!("trap when calling data relocation function: {error}\n"),
                ));
            }
        } else if name == language_function_name
            && let Some(Extern::Func(func)) = instance.get_export(&mut inner.store, name)
        {
            language_func = Some(func);
        }
    }

    let Some(language_func) = language_func else {
        return Err((
            INSTANTIATE,
            format!("module did not contain language function: {language_function_name}"),
        ));
    };

    let mut results = [Val::I32(0)];
    if let Err(error) = language_func.call(&mut inner.store, &[], &mut results) {
        return Err((
            INSTANTIATE,
            format!(
                "trapped when calling language function: {language_function_name}: {error}\n"
            ),
        ));
    }
    let Val::I32(language_address) = results[0] else {
        return Err((
            INSTANTIATE,
            format!("language function did not return an integer: {language_function_name}\n"),
        ));
    };

    Ok((instance, language_address))
}

// ---------------------------------------------------------------------------
// Instance records (wasm_store.c:1688-1710, 1737-1794)
// ---------------------------------------------------------------------------

/// The linear-memory addresses needed to wire an instance: the seven function
/// table indices and the external scanner states offset.
struct InstanceAddresses {
    lex_fn: i32,
    keyword_lex_fn: i32,
    external_states_address: i32,
    scanner_create: i32,
    scanner_destroy: i32,
    scanner_scan: i32,
    scanner_serialize: i32,
    scanner_deserialize: i32,
}

impl InstanceAddresses {
    fn from_wasm_language(language: &LanguageInWasmMemory) -> Self {
        Self {
            lex_fn: language.lex_fn,
            keyword_lex_fn: language.keyword_lex_fn,
            external_states_address: language.external_scanner.states,
            scanner_create: language.external_scanner.create,
            scanner_destroy: language.external_scanner.destroy,
            scanner_scan: language.external_scanner.scan,
            scanner_serialize: language.external_scanner.serialize,
            scanner_deserialize: language.external_scanner.deserialize,
        }
    }
}

fn gc_deleted_instances(inner: &mut StoreInner) {
    inner
        .instances
        .retain(|instance| !instance.language_id.is_deleted());
}

fn cache_instance(
    inner: &mut StoreInner,
    instance: Instance,
    language_id: Arc<LanguageId>,
    addresses: &InstanceAddresses,
) {
    let lex_main_fn = inner.get_function(addresses.lex_fn);
    let lex_keyword_fn = inner.get_function(addresses.keyword_lex_fn);
    let scanner_create_fn = inner.get_function(addresses.scanner_create);
    let scanner_destroy_fn = inner.get_function(addresses.scanner_destroy);
    let scanner_serialize_fn = inner.get_function(addresses.scanner_serialize);
    let scanner_deserialize_fn = inner.get_function(addresses.scanner_deserialize);
    let scanner_scan_fn = inner.get_function(addresses.scanner_scan);

    inner.instances.push(LanguageWasmInstance {
        language_id,
        instance,
        external_states_address: addresses.external_states_address,
        lex_main_fn,
        lex_keyword_fn,
        scanner_create_fn,
        scanner_destroy_fn,
        scanner_serialize_fn,
        scanner_deserialize_fn,
        scanner_scan_fn,
    });
}

// ---------------------------------------------------------------------------
// Static-data copy-out (wasm_store.c:1370-1668)
// ---------------------------------------------------------------------------

/// The result of copying a language's static data out of Wasm memory.
struct CopyOutput {
    language: CLanguage,
    symbol_name_buffer: *mut u8,
    field_name_buffer: *mut u8,
    addresses: InstanceAddresses,
}

/// Reads and validates the language struct at `language_address`, then copies
/// every static array onto the native heap with tree-sitter's allocator.
/// Size derivations follow `wasm_store.c` line for line; the external
/// scanner `states` address is deliberately kept as a wasm-memory offset.
fn copy_language_out(
    inner: &StoreInner,
    language_address: i32,
) -> WasmResult<CopyOutput> {
    let store = &inner.store;
    let memory: Memory = *store.data().memory();
    let view = WasmMemoryView {
        data: memory.data(store),
    };

    let Some(abi_version) = view.read_u32(language_address) else {
        return Err(invalid_language_memory());
    };
    if !(TREE_SITTER_MIN_COMPATIBLE_LANGUAGE_VERSION..=TREE_SITTER_LANGUAGE_VERSION)
        .contains(&abi_version)
    {
        return Err((
            INSTANTIATE,
            format!(
                "incompatible language ABI version {abi_version}; expected between \
                 {TREE_SITTER_MIN_COMPATIBLE_LANGUAGE_VERSION} and {TREE_SITTER_LANGUAGE_VERSION}"
            ),
        ));
    }

    let mut bytes = [0; core::mem::size_of::<LanguageInWasmMemory>()];
    if !view.read(language_address, &mut bytes) {
        return Err(invalid_language_memory());
    }
    let wasm_language: LanguageInWasmMemory = unsafe { ptr::read(bytes.as_ptr().cast()) };
    let addresses_of = InstanceAddresses::from_wasm_language(&wasm_language);

    let has_supertypes = wasm_language.abi_version >= LANGUAGE_VERSION_WITH_RESERVED_WORDS
        && wasm_language.supertype_count > 0;

    // Sentinel addresses bounding the unsized arrays: every static array
    // address, plus the language object itself and the next module's
    // placement offset as upper bounds.
    let mut sentinels: Vec<i32> = vec![
        wasm_language.parse_table,
        wasm_language.small_parse_table,
        wasm_language.small_parse_table_map,
        wasm_language.parse_actions,
        wasm_language.symbol_names,
        wasm_language.field_names,
        wasm_language.field_map_slices,
        wasm_language.field_map_entries,
        wasm_language.symbol_metadata,
        wasm_language.public_symbol_map,
        wasm_language.alias_map,
        wasm_language.alias_sequences,
        wasm_language.lex_modes,
        wasm_language.lex_fn,
        wasm_language.keyword_lex_fn,
        wasm_language.primary_state_ids,
        wasm_language.name,
        wasm_language.reserved_words,
    ];
    if has_supertypes {
        sentinels.extend([
            wasm_language.supertype_symbols,
            wasm_language.supertype_map_entries,
            wasm_language.supertype_map_slices,
        ]);
    }
    if wasm_language.external_token_count > 0 {
        sentinels.extend([
            wasm_language.external_scanner.states,
            wasm_language.external_scanner.symbol_map,
            wasm_language.external_scanner.create,
            wasm_language.external_scanner.destroy,
            wasm_language.external_scanner.scan,
            wasm_language.external_scanner.serialize,
            wasm_language.external_scanner.deserialize,
        ]);
    }
    sentinels.push(language_address);
    sentinels.push(inner.current_memory_offset as i32);

    let mut language = CLanguage::zeroed();
    language.abi_version = wasm_language.abi_version;
    language.symbol_count = wasm_language.symbol_count;
    language.alias_count = wasm_language.alias_count;
    language.token_count = wasm_language.token_count;
    language.external_token_count = wasm_language.external_token_count;
    language.state_count = wasm_language.state_count;
    language.large_state_count = wasm_language.large_state_count;
    language.production_id_count = wasm_language.production_id_count;
    language.field_count = wasm_language.field_count;
    language.supertype_count = wasm_language.supertype_count;
    language.max_alias_sequence_length = wasm_language.max_alias_sequence_length;
    language.keyword_capture_token = wasm_language.keyword_capture_token;
    language.metadata = super::CLanguageMetadata {
        major_version: wasm_language.metadata.major_version,
        minor_version: wasm_language.metadata.minor_version,
        patch_version: wasm_language.metadata.patch_version,
    };

    let mut ok = true;
    let invalid = || invalid_language_memory();
    let mut field_table: Option<StringTable> = None;

    language.parse_table = unsafe {
        copy(
            &view,
            wasm_language.parse_table,
            wasm_language.large_state_count as usize
                * wasm_language.symbol_count as usize
                * core::mem::size_of::<u16>(),
            &mut ok,
        )
    }
    .cast();
    language.parse_actions = unsafe {
        copy_unsized_static_array(&view, wasm_language.parse_actions, &sentinels, &mut ok)
    }
    .cast();
    let symbol_table = unsafe {
        copy_strings(
            &view,
            wasm_language.symbol_names,
            wasm_language.symbol_count as usize + wasm_language.alias_count as usize,
            &mut ok,
        )
    };
    language.symbol_metadata = unsafe {
        copy(
            &view,
            wasm_language.symbol_metadata,
            (wasm_language.symbol_count as usize + wasm_language.alias_count as usize)
                * core::mem::size_of::<CSymbolMetadata>(),
            &mut ok,
        )
    }
    .cast();
    language.public_symbol_map = unsafe {
        copy(
            &view,
            wasm_language.public_symbol_map,
            (wasm_language.symbol_count as usize + wasm_language.alias_count as usize)
                * core::mem::size_of::<u16>(),
            &mut ok,
        )
    }
    .cast();
    language.lex_modes = unsafe {
        copy(
            &view,
            wasm_language.lex_modes,
            wasm_language.state_count as usize * core::mem::size_of::<CLexerMode>(),
            &mut ok,
        )
    }
    .cast();
    // Past this point allocations have happened, so every error path must
    // free the arrays copied so far first, mirroring the C error label's
    // delete_partially_loaded_language call (wasm_store.c:1719-1722). The
    // string-table arrays are freed through `language.symbol_names` /
    // `language.field_names`, which are therefore assigned as soon as the
    // tables exist.
    if let Some(table) = &symbol_table {
        language.symbol_names = table.array.cast();
    }
    if !ok {
        unsafe { free_language_data(&mut language) };
        return Err(invalid());
    }
    let Some(symbol_table) = symbol_table else {
        unsafe { free_language_data(&mut language) };
        return Err(invalid());
    };

    if language.field_count > 0 && language.production_id_count > 0 {
        language.field_map_slices = unsafe {
            copy(
                &view,
                wasm_language.field_map_slices,
                wasm_language.production_id_count as usize * core::mem::size_of::<CMapSlice>(),
                &mut ok,
            )
        }
        .cast();
        if !ok {
            unsafe { free_language_data(&mut language) };
            return Err(invalid());
        }

        // The field map entry count is the greatest slice end across all
        // production ids.
        let mut field_map_entry_count: u32 = 0;
        for index in 0..wasm_language.production_id_count {
            let slice = unsafe { *language.field_map_slices.add(index as usize) };
            let slice_end = u32::from(slice.index) + u32::from(slice.length);
            if slice_end > field_map_entry_count {
                field_map_entry_count = slice_end;
            }
        }

        language.field_map_entries = unsafe {
            copy(
                &view,
                wasm_language.field_map_entries,
                field_map_entry_count as usize * core::mem::size_of::<CFieldMapEntry>(),
                &mut ok,
            )
        }
        .cast();
        field_table = unsafe {
            copy_strings(
                &view,
                wasm_language.field_names,
                wasm_language.field_count as usize + 1,
                &mut ok,
            )
        };
        if let Some(table) = &field_table {
            language.field_names = table.array.cast();
        }
        if !ok {
            unsafe { free_language_data(&mut language) };
            return Err(invalid());
        }
        let Some(unwrapped_field_table) = field_table else {
            unsafe { free_language_data(&mut language) };
            return Err(invalid());
        };
        field_table = Some(unwrapped_field_table);
    }

    // Supertypes (only in ABI 15+ grammars that declare any).
    if has_supertypes {
        language.supertype_symbols = unsafe {
            copy(
                &view,
                wasm_language.supertype_symbols,
                wasm_language.supertype_count as usize * core::mem::size_of::<u16>(),
                &mut ok,
            )
        }
        .cast();
        if !ok {
            unsafe { free_language_data(&mut language) };
            return Err(invalid());
        }

        // The supertype map slice count is the greatest supertype id + 1.
        let mut largest_supertype: u32 = 0;
        for index in 0..language.supertype_count {
            let supertype = unsafe { *language.supertype_symbols.add(index as usize) };
            if u32::from(supertype) > largest_supertype {
                largest_supertype = u32::from(supertype);
            }
        }

        language.supertype_map_slices = unsafe {
            copy(
                &view,
                wasm_language.supertype_map_slices,
                (largest_supertype as usize + 1) * core::mem::size_of::<CMapSlice>(),
                &mut ok,
            )
        }
        .cast();
        if !ok {
            unsafe { free_language_data(&mut language) };
            return Err(invalid());
        }

        let last_slice =
            unsafe { *language.supertype_map_slices.add(largest_supertype as usize) };
        let supertype_map_entry_count = u32::from(last_slice.index) + u32::from(last_slice.length);

        language.supertype_map_entries = unsafe {
            copy(
                &view,
                wasm_language.supertype_map_entries,
                supertype_map_entry_count as usize * core::mem::size_of::<u16>(),
                &mut ok,
            )
        }
        .cast();
        if !ok {
            unsafe { free_language_data(&mut language) };
            return Err(invalid());
        }
    }

    // Alias sequences (only in grammars with productions that need aliases).
    if language.max_alias_sequence_length > 0 && language.production_id_count > 0 {
        // The alias map is a (symbol, count, symbols...)-tuple list terminated
        // by a null symbol; walk it in Wasm memory to find its size.
        let mut alias_map_size = 0usize;
        loop {
            let Some(symbol) = view.read_u16(wasm_language.alias_map + alias_map_size as i32)
            else {
                unsafe { free_language_data(&mut language) };
                return Err(invalid());
            };
            alias_map_size += core::mem::size_of::<u16>();
            if symbol == 0 {
                break;
            }
            let Some(value_count) =
                view.read_u16(wasm_language.alias_map + alias_map_size as i32)
            else {
                unsafe { free_language_data(&mut language) };
                return Err(invalid());
            };
            alias_map_size += core::mem::size_of::<u16>();
            alias_map_size += value_count as usize * core::mem::size_of::<u16>();
        }
        language.alias_map = unsafe {
            copy(
                &view,
                wasm_language.alias_map,
                alias_map_size,
                &mut ok,
            )
        }
        .cast();
        language.alias_sequences = unsafe {
            copy(
                &view,
                wasm_language.alias_sequences,
                wasm_language.production_id_count as usize
                    * u32::from(wasm_language.max_alias_sequence_length) as usize
                    * core::mem::size_of::<u16>(),
                &mut ok,
            )
        }
        .cast();
        if !ok {
            unsafe { free_language_data(&mut language) };
            return Err(invalid());
        }
    }

    if language.state_count > language.large_state_count {
        let small_state_count = wasm_language.state_count - wasm_language.large_state_count;
        language.small_parse_table_map = unsafe {
            copy(
                &view,
                wasm_language.small_parse_table_map,
                small_state_count as usize * core::mem::size_of::<u32>(),
                &mut ok,
            )
        }
        .cast();
        language.small_parse_table = unsafe {
            copy_unsized_static_array(&view, wasm_language.small_parse_table, &sentinels, &mut ok)
        }
        .cast();
        if !ok {
            unsafe { free_language_data(&mut language) };
            return Err(invalid());
        }
    }

    if language.abi_version >= LANGUAGE_VERSION_WITH_PRIMARY_STATES {
        language.primary_state_ids = unsafe {
            copy(
                &view,
                wasm_language.primary_state_ids,
                wasm_language.state_count as usize * core::mem::size_of::<u16>(),
                &mut ok,
            )
        }
        .cast();
        if !ok {
            unsafe { free_language_data(&mut language) };
            return Err(invalid());
        }
    }

    if language.abi_version >= LANGUAGE_VERSION_WITH_RESERVED_WORDS {
        language.name = unsafe { copy_string(&view, wasm_language.name, &mut ok) }.cast();
        if !ok {
            unsafe { free_language_data(&mut language) };
            return Err(invalid());
        }
        language.max_reserved_word_set_size = wasm_language.max_reserved_word_set_size;

        // The reserved word set count is the maximum set id across all lex
        // modes.
        let mut max_reserved_word_set_id: u16 = 0;
        for index in 0..wasm_language.state_count {
            let mode = unsafe { *language.lex_modes.add(index as usize) };
            if mode.reserved_word_set_id > max_reserved_word_set_id {
                max_reserved_word_set_id = mode.reserved_word_set_id;
            }
        }

        if max_reserved_word_set_id > 0 && language.max_reserved_word_set_size > 0 {
            let reserved_word_count = (u32::from(max_reserved_word_set_id) as usize + 1)
                * u32::from(language.max_reserved_word_set_size) as usize;
            language.reserved_words = unsafe {
                copy(
                    &view,
                    wasm_language.reserved_words,
                    reserved_word_count * core::mem::size_of::<u16>(),
                    &mut ok,
                )
            }
            .cast();
            if !ok {
                unsafe { free_language_data(&mut language) };
                return Err(invalid());
            }
        }
    }

    if language.external_token_count > 0 {
        language.external_scanner.symbol_map = unsafe {
            copy(
                &view,
                wasm_language.external_scanner.symbol_map,
                wasm_language.external_token_count as usize * core::mem::size_of::<u16>(),
                &mut ok,
            )
        }
        .cast();
        if !ok {
            unsafe { free_language_data(&mut language) };
            return Err(invalid());
        }
        // Keep the valid-tokens table address as a linear-memory offset: it
        // stays valid across memory.grow, unlike any native pointer.
        language.external_scanner.states =
            wasm_language.external_scanner.states as usize as *const bool;
    }

    let symbol_name_buffer = unsafe { finalize_string_table(symbol_table) };
    let field_name_buffer = match field_table {
        Some(table) => unsafe { finalize_string_table(table) },
        None => ptr::null_mut(),
    };

    Ok(CopyOutput {
        language,
        symbol_name_buffer,
        field_name_buffer,
        addresses: addresses_of,
    })
}

// ---------------------------------------------------------------------------
// Load and cross-store reuse (wasm_store.c:1319-1725, 1727-1798)
// ---------------------------------------------------------------------------

pub(crate) fn load_language(
    inner: &mut StoreInner,
    language_name: &str,
    wasm: &[u8],
) -> WasmResult<*const TSLanguage> {
    let initial_memory_offset = inner.current_memory_offset;
    let initial_function_table_offset = inner.current_function_table_offset;

    let result = load_language_inner(inner, language_name, wasm);
    if result.is_err() {
        inner.current_memory_offset = initial_memory_offset;
        inner.current_function_table_offset = initial_function_table_offset;
    }
    result
}

fn load_language_inner(
    inner: &mut StoreInner,
    language_name: &str,
    wasm: &[u8],
) -> WasmResult<*const TSLanguage> {
    let dylink_info = dylink_info_parse(wasm)?;

    let module = Module::new(&inner.engine, wasm)
        .map_err(|error| (COMPILE, format!("error compiling Wasm module: {error}")))?;

    let (instance, language_address) =
        instantiate(inner, &module, language_name, &dylink_info)?;

    let CopyOutput {
        language,
        symbol_name_buffer,
        field_name_buffer,
        addresses,
    } = copy_language_out(inner, language_address)?;

    let language_id = LanguageId::new();
    let boxed_language = Box::new(WasmLanguage {
        language,
        ref_count: AtomicU32::new(1),
        language_id: Arc::clone(&language_id),
        module,
        name: CString::new(language_name).map_err(|_| {
            (
                INSTANTIATE,
                "invalid language name: embedded NUL byte".to_string(),
            )
        })?,
        symbol_name_buffer,
        field_name_buffer,
        dylink_info,
    });
    let language_ptr = Box::into_raw(boxed_language);

    unsafe {
        (*language_ptr).language.lex_fn = Some(super::ts_wasm_store__sentinel_lex_fn);
        (*language_ptr).language.keyword_lex_fn = None;
    }

    // Clear out instances of languages that have been deleted, then record
    // this store's instance of the newly loaded module.
    gc_deleted_instances(inner);
    cache_instance(inner, instance, language_id, &addresses);

    Ok(unsafe { ptr::addr_of!((*language_ptr).language).cast::<TSLanguage>() })
}

/// Reuses a loaded language in this store, instantiating its module here if
/// needed. Returns the index of this store's instance record.
pub(crate) fn add_language(inner: &mut StoreInner, language: *const TSLanguage) -> Option<usize> {
    let wasm_language = unsafe { &*(language.cast::<WasmLanguage>()) };

    gc_deleted_instances(inner);
    if let Some(index) = inner
        .instances
        .iter()
        .position(|instance| Arc::ptr_eq(&instance.language_id, &wasm_language.language_id))
    {
        return Some(index);
    }

    let language_name = wasm_language.name.to_string_lossy();
    let initial_memory_offset = inner.current_memory_offset;
    let initial_function_table_offset = inner.current_function_table_offset;
    let (instance, language_address) = instantiate(
        inner,
        &wasm_language.module,
        &language_name,
        &wasm_language.dylink_info,
    )
    .ok()?;

    let addresses = {
        let store = &inner.store;
        let memory: Memory = *store.data().memory();
        let view = WasmMemoryView {
            data: memory.data(store),
        };
        let mut bytes = [0; core::mem::size_of::<LanguageInWasmMemory>()];
        if !view.read(language_address, &mut bytes) {
            // Not covered by instantiate()'s internal rollback: the offsets
            // already advanced when instantiation succeeded.
            inner.current_memory_offset = initial_memory_offset;
            inner.current_function_table_offset = initial_function_table_offset;
            return None;
        }
        let wasm_language: LanguageInWasmMemory = unsafe { ptr::read(bytes.as_ptr().cast()) };
        InstanceAddresses::from_wasm_language(&wasm_language)
    };

    cache_instance(inner, instance, Arc::clone(&wasm_language.language_id), &addresses);
    Some(inner.instances.len() - 1)
}
