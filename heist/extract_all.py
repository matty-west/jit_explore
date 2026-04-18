import frida
import sys
import json
import time
import subprocess

# ─────────────────────────────────────────────────────────────────────────────
# extract_all.py  (v3.4 "Golden Store" Edition - Sudo/Direct Mode)
# ─────────────────────────────────────────────────────────────────────────────

JS_CODE = """
'use strict';

// ── Helpers ──────────────────────────────────────────────────────────────────

function hexU32(op) {
    return '0x' + (op >>> 0).toString(16).padStart(8, '0');
}

function isUnconditionalRet(op) {
    return (op & 0xFFFFFC1F) === 0xD65F0000;
}

function isStoreOp(op) {
    const isAmx = (op & 0xFFF00000) === 0x00200000;
    const isSme = (op & 0xF0000000) === 0xE0000000;
    return isAmx || isSme;
}

function storeType(op) {
    if ((op & 0xFFF00000) === 0x00200000) return 'amx_store';
    if ((op & 0xF0000000) === 0xE0000000) return 'sme_store';
    return 'unknown_store';
}

// ── Core extraction ───────────────────────────────────────────────────────────

function extractFunction(address, limit, maxRets) {
    limit   = limit   || 50000;
    maxRets = maxRets || 3;

    let curr       = address;
    let count      = 0;
    let retsSeen   = 0;
    let block      = [];
    let stores     = [];

    while (count < limit) {
        let op;
        try {
            op = curr.readU32() >>> 0;
        } catch (e) {
            break;
        }

        const hex = hexU32(op);
        const byteOffset = count * 4;

        block.push(hex);

        if (isStoreOp(op)) {
            stores.push({ offset: byteOffset, opcode: hex, type: storeType(op) });
        }

        if (isUnconditionalRet(op)) {
            retsSeen++;
            if (retsSeen >= maxRets) break;
        }

        curr = curr.add(4);
        count++;
    }

    return { block: block, stores: stores };
}

function captureAbi(ctx) {
    const abi = {};
    for (let i = 0; i <= 28; i++) {
        abi['x' + i] = ctx['x' + i].toString();
    }
    return abi;
}

// ── Globals ───────────────────────────────────────────────────────────────────

const opcodes     = new Set();
const blocks      = [];
const leafKernels = new Set();

let insideSgemm   = 0;
let callDepth     = 0;
let monitoringCount = 0;

// ── Initialization ───────────────────────────────────────────────────────────

function initializeHeist() {
    console.log('[*] Initializing heist hooks...');
    
    Process.enumerateModules().forEach(function(m) {
        const targets = ['blas', 'linearalgebra', 'vdsp', 'bnns'];
        if (!targets.some(function(t) { return m.name.toLowerCase().indexOf(t) !== -1; })) {
            return;
        }

        console.log('[*] Monitoring ' + m.name);

        m.enumerateExports().forEach(function(exp) {
            if (exp.name === 'cblas_sgemm') {
                try {
                    monitoringCount++;
                    Interceptor.attach(exp.address, {
                        onEnter: function(args) {
                            insideSgemm++;
                            callDepth = 0;
                            console.log('[*] cblas_sgemm called (insideSgemm=' + insideSgemm + ')');
                        },
                        onLeave: function(retval) {
                            insideSgemm = Math.max(0, insideSgemm - 1);
                            callDepth   = 0;
                            
                            console.log('[*] Math complete. Flushing data...');
                            const payload = {
                                type:         'heist_data',
                                opcodes:      Array.from(opcodes),
                                blocks:       blocks,
                                leafKernels:  Array.from(leafKernels)
                            };
                            send(payload);
                        }
                    });
                } catch (e) {
                    console.log('[!] Failed to hook cblas_sgemm: ' + e);
                }
            }

            if (m.name.indexOf('BLAS') !== -1) {
                try {
                    Interceptor.attach(exp.address, {
                        onEnter: function(args) {
                            if (insideSgemm === 0) return;

                            const indent = '  '.repeat(callDepth);
                            console.log(indent + '-> ' + exp.name + ' @ ' + exp.address);
                            callDepth++;
                            this.capturedDepth = callDepth;

                            const abi = captureAbi(this.context);
                            const extracted = extractFunction(exp.address, 50000, 3);
                            extracted.block.forEach(function(h) { opcodes.add(h); });

                            if (extracted.stores.length > 0) {
                                console.log(indent + '  [STORES] ' + exp.name
                                    + ' has ' + extracted.stores.length + ' store(s)');
                            }

                            blocks.push({
                                name:    exp.name,
                                address: exp.address.toString(),
                                abi:     abi,
                                block:   extracted.block,
                                stores:  extracted.stores
                            });

                            if (extracted.stores.length > 0) {
                                leafKernels.add(exp.address.toString());
                            }
                        },

                        onLeave: function(retval) {
                            if (insideSgemm === 0) return;
                            if (this.capturedDepth !== undefined) {
                                callDepth = this.capturedDepth - 1;
                            }
                        }
                    });
                } catch (e) {}
            }
        });
    });
    console.log('[*] Total top-level hooks installed: ' + monitoringCount);
}

// Start immediately
initializeHeist();
"""

def on_message(message, data):
    if message['type'] == 'send':
        payload = message['payload']
        if payload.get('type') == 'heist_data':
            print(f"[*] Total unique opcodes captured : {len(payload['opcodes'])}")
            print(f"[*] Total blocks captured         : {len(payload['blocks'])}")
            print(f"[*] Leaf kernels with stores      : {len(payload['leafKernels'])}")

            with open('all_opcodes.json', 'w') as f:
                json.dump(payload['opcodes'], f, indent=2)
            print("[*] Written: all_opcodes.json")

            with open('stolen_blocks.json', 'w') as f:
                json.dump(payload['blocks'], f, indent=2)
            print("[*] Written: stolen_blocks.json")

            sys.exit(0)
    elif message['type'] == 'error':
        print(f"[!] JS error: {message['description']}")
        if 'stack' in message:
            print(message['stack'])
    else:
        print(message)

def main():
    try:
        # Start amx_runner — but don't capture stdout/stderr to avoid pipe buffering issues
        print("[*] Starting amx_runner...")
        process = subprocess.Popen(
            ["heist/amx_runner"],
            stdin=subprocess.PIPE,
        )

        # Give it a tiny bit to boot
        time.sleep(1)

        device = frida.get_local_device()
        print(f"[*] Attaching to PID {process.pid}...")
        session = device.attach(process.pid)
        
        script = session.create_script(JS_CODE)
        script.on('message', on_message)
        script.load()
        
        print(f"[*] Attached and script loaded. Triggering...")
        
        # Trigger the math
        process.stdin.write(b"\n")
        process.stdin.flush()
        
        # Keep alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[*] Interrupted.")
    except Exception as e:
        print(f"[!] Error: {e}")
        raise

if __name__ == '__main__':
    main()
