//! Wasm linear-memory layout and the `dylink.0` section parser.
//!
//! The layout mirrors `lib/src/wasm_store.c`:
//! `[ stack | stdlib statics | lexer | language statics | serialization buffer | heap ]`

use super::PARSE;
use super::WasmResult;

pub(crate) const MEMORY_PAGE_SIZE: u32 = 0x10000;
/// 128 MiB expressed in pages; the hard maximum of the store's memory.
pub(crate) const MAX_MEMORY_PAGES: u32 = 128 * 1024 * 1024 / MEMORY_PAGE_SIZE;
pub(crate) const SERIALIZATION_BUFFER_SIZE: usize = 1024;

/// The contents of the `dylink.0` custom section of a Wasm module, as
/// specified by the WebAssembly dynamic linking ABI proposal.
#[derive(Clone, Copy, Default)]
#[allow(dead_code)] // align fields are part of the format but, like the C
// implementation, only parsed, never consulted
pub(crate) struct DylinkInfo {
    pub memory_size: u32,
    pub memory_align: u32,
    pub table_size: u32,
    pub table_align: u32,
}

/// The memory layout of a `TSLanguage` when compiled to wasm32. Used to copy
/// static language data out of the Wasm memory. Field order must match
/// `LanguageInWasmMemory` in `wasm_store.c` (168 bytes, align 4).
#[repr(C)]
pub(crate) struct LanguageInWasmMemory {
    pub abi_version: u32,
    pub symbol_count: u32,
    pub alias_count: u32,
    pub token_count: u32,
    pub external_token_count: u32,
    pub state_count: u32,
    pub large_state_count: u32,
    pub production_id_count: u32,
    pub field_count: u32,
    pub max_alias_sequence_length: u16,
    pub parse_table: i32,
    pub small_parse_table: i32,
    pub small_parse_table_map: i32,
    pub parse_actions: i32,
    pub symbol_names: i32,
    pub field_names: i32,
    pub field_map_slices: i32,
    pub field_map_entries: i32,
    pub symbol_metadata: i32,
    pub public_symbol_map: i32,
    pub alias_map: i32,
    pub alias_sequences: i32,
    pub lex_modes: i32,
    pub lex_fn: i32,
    pub keyword_lex_fn: i32,
    pub keyword_capture_token: u16,
    pub external_scanner: WasmExternalScanner,
    pub primary_state_ids: i32,
    pub name: i32,
    pub reserved_words: i32,
    pub max_reserved_word_set_size: u16,
    pub supertype_count: u32,
    pub supertype_symbols: i32,
    pub supertype_map_slices: i32,
    pub supertype_map_entries: i32,
    pub metadata: WasmLanguageMetadata,
}

const _: () = assert!(core::mem::size_of::<LanguageInWasmMemory>() == 168);

#[repr(C)]
pub(crate) struct WasmExternalScanner {
    pub states: i32,
    pub symbol_map: i32,
    pub create: i32,
    pub destroy: i32,
    pub scan: i32,
    pub serialize: i32,
    pub deserialize: i32,
}

#[repr(C)]
#[allow(clippy::struct_field_names)] // field names mirror the C ABI
pub(crate) struct WasmLanguageMetadata {
    pub major_version: u8,
    pub minor_version: u8,
    pub patch_version: u8,
}

/// The memory layout of a `TSLexer` when compiled to wasm32: the mutable data
/// fields followed by int32 function-table indices instead of native function
/// pointers. Field order must match `LexerInWasmMemory` in `wasm_store.c`
/// (28 bytes, align 4).
#[repr(C)]
pub(crate) struct LexerInWasmMemory {
    pub lookahead: i32,
    pub result_symbol: u16,
    pub advance: i32,
    pub mark_end: i32,
    pub get_column: i32,
    pub is_at_included_range_start: i32,
    pub eof: i32,
}

const _: () = assert!(core::mem::size_of::<LexerInWasmMemory>() == 28);

/// The data prefix of `TSLexer` (no function pointers). This eight-byte slice
/// is copied in and out of the Wasm memory around lex and scan calls. The C
/// struct is 4 + 2 + 2 padding bytes.
pub(crate) const LEXER_DATA_PREFIX_SIZE: usize = 8;

const _: () = assert!(core::mem::size_of::<TSLexerDataPrefixMirror>() == 8);

#[repr(C)]
struct TSLexerDataPrefixMirror {
    _lookahead: i32,
    _result_symbol: u16,
}

/// A checked view of the current Wasm linear memory. Must be re-derived from
/// the store after every Wasm call: a memory.grow inside the module (e.g. via
/// the scanner bump allocator) invalidates the base pointer, while offsets
/// stay valid.
pub(crate) struct WasmMemoryView<'a> {
    pub data: &'a [u8],
}

impl WasmMemoryView<'_> {
    pub fn contains(&self, address: i32, size: usize) -> bool {
        if address < 0 {
            return false;
        }
        let start = address as usize;
        start <= self.data.len() && size <= self.data.len() - start
    }

    pub fn read(&self, address: i32, result: &mut [u8]) -> bool {
        if !self.contains(address, result.len()) {
            return false;
        }
        result.copy_from_slice(&self.data[address as usize..][..result.len()]);
        true
    }

    pub fn read_u16(&self, address: i32) -> Option<u16> {
        let mut bytes = [0; 2];
        if !self.read(address, &mut bytes) {
            return None;
        }
        Some(u16::from_ne_bytes(bytes))
    }

    pub fn read_u32(&self, address: i32) -> Option<u32> {
        let mut bytes = [0; 4];
        if !self.read(address, &mut bytes) {
            return None;
        }
        Some(u32::from_ne_bytes(bytes))
    }

    pub fn string_length(&self, address: i32) -> Option<usize> {
        if address < 0 || address as usize >= self.data.len() {
            return None;
        }
        let start = address as usize;
        self.data[start..]
            .iter()
            .position(|&byte| byte == 0)
    }
}

struct WasmReader<'a> {
    data: &'a [u8],
    offset: usize,
    size: usize,
}

impl WasmReader<'_> {
    fn read_u8(&mut self) -> Option<u8> {
        if self.offset >= self.size {
            return None;
        }
        let byte = self.data[self.offset];
        self.offset += 1;
        Some(byte)
    }

    fn read_uleb128(&mut self) -> Option<u32> {
        let mut value: u32 = 0;
        for shift in (0..32).step_by(7) {
            let byte = self.read_u8()?;
            if shift == 28 && (byte & 0xf0) != 0 {
                return None;
            }
            value |= u32::from(byte & 0x7f) << shift;
            if (byte & 0x80) == 0 {
                return Some(value);
            }
        }
        None
    }
}

pub(crate) fn dylink_info_parse(bytes: &[u8]) -> WasmResult<DylinkInfo> {
    const WASM_MAGIC_NUMBER: [u8; 4] = [0, b'a', b's', b'm'];
    const WASM_VERSION: [u8; 4] = [1, 0, 0, 0];
    const WASM_CUSTOM_SECTION: u8 = 0x0;
    const WASM_DYLINK_MEM_INFO: u8 = 0x1;

    let invalid = || (PARSE, "failed to parse dylink section of Wasm module".into());

    if bytes.len() < 8 || bytes[..4] != WASM_MAGIC_NUMBER || bytes[4..8] != WASM_VERSION {
        return Err(invalid());
    }

    let mut reader = WasmReader {
        data: bytes,
        offset: 8,
        size: bytes.len(),
    };

    while reader.offset < reader.size {
        let Some(section_id) = reader.read_u8() else {
            return Err(invalid());
        };
        let Some(section_length) = reader.read_uleb128() else {
            return Err(invalid());
        };
        if section_length as usize > reader.size - reader.offset {
            return Err(invalid());
        }
        let section_end = reader.offset + section_length as usize;

        if section_id == WASM_CUSTOM_SECTION {
            let previous_size = reader.size;
            reader.size = section_end;
            let Some(name_length) = reader.read_uleb128() else {
                return Err(invalid());
            };
            if name_length as usize > reader.size - reader.offset {
                return Err(invalid());
            }
            let name_end = reader.offset + name_length as usize;

            if name_length == 8 && &reader.data[reader.offset..name_end] == b"dylink.0" {
                reader.offset = name_end;
                while reader.offset < section_end {
                    let Some(subsection_type) = reader.read_u8() else {
                        return Err(invalid());
                    };
                    let Some(subsection_size) = reader.read_uleb128() else {
                        return Err(invalid());
                    };
                    if subsection_size as usize > section_end - reader.offset {
                        return Err(invalid());
                    }
                    let subsection_end = reader.offset + subsection_size as usize;
                    if subsection_type == WASM_DYLINK_MEM_INFO {
                        reader.size = subsection_end;
                        let Some(info) = (|| {
                            Some(DylinkInfo {
                                memory_size: reader.read_uleb128()?,
                                memory_align: reader.read_uleb128()?,
                                table_size: reader.read_uleb128()?,
                                table_align: reader.read_uleb128()?,
                            })
                        })() else {
                            return Err(invalid());
                        };
                        if reader.offset != subsection_end {
                            return Err(invalid());
                        }
                        return Ok(info);
                    }
                    reader.offset = subsection_end;
                }
            }
            reader.size = previous_size;
        }
        reader.offset = section_end;
    }
    Err(invalid())
}

/// Address one past the last module's data: the scanner heap starts here.
#[inline]
pub(crate) fn heap_address(current_memory_offset: u32) -> u32 {
    current_memory_offset + SERIALIZATION_BUFFER_SIZE as u32
}

/// The serialization buffer sits directly at the current placement offset.
#[inline]
pub(crate) fn serialization_buffer_address(current_memory_offset: u32) -> u32 {
    current_memory_offset
}

/// The stdlib module's `memory` import determines the store's initial page
/// count; this scans a compiled module's imports for it.
pub(crate) fn stdlib_memory_minimum(module: &wasmtime::Module) -> Option<u64> {
    let mut minimum = None;
    for import in module.imports() {
        if import.name() == "memory"
            && let wasmtime::ExternType::Memory(memory_type) = import.ty()
        {
            minimum = Some(memory_type.minimum());
        }
    }
    minimum
}
