/* @ts-self-types="./solver_wasm.d.ts" */

export class Solver {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Solver.prototype);
        obj.__wbg_ptr = ptr;
        SolverFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SolverFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_solver_free(ptr, 0);
    }
    /**
     * @returns {Uint8Array}
     */
    best_assignment() {
        const ret = wasm.solver_best_assignment(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * @returns {number}
     */
    best_cost_total() {
        const ret = wasm.solver_best_cost_total(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    current_iteration() {
        const ret = wasm.solver_current_iteration(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} max_iterations
     * @param {string} weights_json
     */
    constructor(max_iterations, weights_json) {
        const ptr0 = passStringToWasm0(weights_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.solver_new(max_iterations, ptr0, len0);
        this.__wbg_ptr = ret >>> 0;
        SolverFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} max_iterations
     * @param {string} weights_json
     * @param {Uint8Array} seed_flat
     * @returns {Solver}
     */
    static new_with_seed(max_iterations, weights_json, seed_flat) {
        const ptr0 = passStringToWasm0(weights_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray8ToWasm0(seed_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.solver_new_with_seed(max_iterations, ptr0, len0, ptr1, len1);
        return Solver.__wrap(ret);
    }
    /**
     * @returns {SolverResult}
     */
    result() {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.solver_result(ptr);
        return SolverResult.__wrap(ret);
    }
    /**
     * @param {number} chunk_size
     * @returns {boolean}
     */
    step(chunk_size) {
        const ret = wasm.solver_step(this.__wbg_ptr, chunk_size);
        return ret !== 0;
    }
}
if (Symbol.dispose) Solver.prototype[Symbol.dispose] = Solver.prototype.free;

export class SolverResult {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(SolverResult.prototype);
        obj.__wbg_ptr = ptr;
        SolverResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SolverResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_solverresult_free(ptr, 0);
    }
    /**
     * @returns {Uint8Array}
     */
    get assignment() {
        const ret = wasm.solverresult_assignment(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * @returns {WasmCostBreakdown}
     */
    get cost() {
        const ret = wasm.solverresult_cost(this.__wbg_ptr);
        return WasmCostBreakdown.__wrap(ret);
    }
}
if (Symbol.dispose) SolverResult.prototype[Symbol.dispose] = SolverResult.prototype.free;

export class WasmCostBreakdown {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmCostBreakdown.prototype);
        obj.__wbg_ptr = ptr;
        WasmCostBreakdownFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmCostBreakdownFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmcostbreakdown_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get commissioner_overlap() {
        const ret = wasm.__wbg_get_wasmcostbreakdown_commissioner_overlap(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get consecutive_opponents() {
        const ret = wasm.__wbg_get_wasmcostbreakdown_consecutive_opponents(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get early_late_alternation() {
        const ret = wasm.__wbg_get_wasmcostbreakdown_early_late_alternation(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get early_late_balance() {
        const ret = wasm.__wbg_get_wasmcostbreakdown_early_late_balance(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get lane_balance() {
        const ret = wasm.__wbg_get_wasmcostbreakdown_lane_balance(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get lane_switch_balance() {
        const ret = wasm.__wbg_get_wasmcostbreakdown_lane_switch_balance(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get late_lane_balance() {
        const ret = wasm.__wbg_get_wasmcostbreakdown_late_lane_balance(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get matchup_balance() {
        const ret = wasm.__wbg_get_wasmcostbreakdown_matchup_balance(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get total() {
        const ret = wasm.__wbg_get_wasmcostbreakdown_total(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set commissioner_overlap(arg0) {
        wasm.__wbg_set_wasmcostbreakdown_commissioner_overlap(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set consecutive_opponents(arg0) {
        wasm.__wbg_set_wasmcostbreakdown_consecutive_opponents(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set early_late_alternation(arg0) {
        wasm.__wbg_set_wasmcostbreakdown_early_late_alternation(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set early_late_balance(arg0) {
        wasm.__wbg_set_wasmcostbreakdown_early_late_balance(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set lane_balance(arg0) {
        wasm.__wbg_set_wasmcostbreakdown_lane_balance(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set lane_switch_balance(arg0) {
        wasm.__wbg_set_wasmcostbreakdown_lane_switch_balance(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set late_lane_balance(arg0) {
        wasm.__wbg_set_wasmcostbreakdown_late_lane_balance(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set matchup_balance(arg0) {
        wasm.__wbg_set_wasmcostbreakdown_matchup_balance(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set total(arg0) {
        wasm.__wbg_set_wasmcostbreakdown_total(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) WasmCostBreakdown.prototype[Symbol.dispose] = WasmCostBreakdown.prototype.free;

/**
 * @param {Uint8Array} flat
 * @param {string} weights_json
 * @returns {WasmCostBreakdown}
 */
export function evaluate_assignment(flat, weights_json) {
    const ptr0 = passArray8ToWasm0(flat, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(weights_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.evaluate_assignment(ptr0, len0, ptr1, len1);
    return WasmCostBreakdown.__wrap(ret);
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_df03e93053e0f4bc: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_getRandomValues_3f44b700395062e5: function() { return handleError(function (arg0, arg1) {
            globalThis.crypto.getRandomValues(getArrayU8FromWasm0(arg0, arg1));
        }, arguments); },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./solver_wasm_bg.js": import0,
    };
}

const SolverFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_solver_free(ptr >>> 0, 1));
const SolverResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_solverresult_free(ptr >>> 0, 1));
const WasmCostBreakdownFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmcostbreakdown_free(ptr >>> 0, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('solver_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
