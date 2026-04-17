import frida
import sys
import json
import time

# Script to find EVERY instruction in libBLAS.dylib
JS_CODE = """
'use strict';

const opcodes = new Set();

function extract(address, name) {
    let curr = address;
    let count = 0;
    const limit = 10000;

    while (count < limit) {
        try {
            const op = curr.readU32();
            const instr = Instruction.parse(curr);
            
            // Just capture EVERYTHING for analysis
            const hex = '0x' + op.toString(16).padStart(8, '0');
            if (!opcodes.has(hex)) {
                opcodes.add(hex);
            }

            if (op === 0xD65F03C0) break; // RET
            curr = instr.next;
            count++;
        } catch (e) { break; }
    }
}

Process.enumerateModules().forEach(m => {
    if (m.name.toLowerCase().includes('blas')) {
        console.log('[*] Extracting all from ' + m.name);
        m.enumerateExports().forEach(exp => {
            if (exp.name.includes('sgemm')) {
                extract(exp.address, exp.name);
            }
        });
    }
});

send({ type: 'opcodes', data: Array.from(opcodes) });
"""

def on_message(message, data):
    if message['type'] == 'send':
        payload = message['payload']
        if payload['type'] == 'opcodes':
            print(f"[*] Total unique opcodes captured: {len(payload['data'])}")
            with open('all_opcodes.json', 'w') as f:
                json.dump(payload['data'], f, indent=2)
            sys.exit(0)
    else:
        print(message)

def main():
    try:
        session = frida.attach("amx_runner")
        script = session.create_script(JS_CODE)
        script.on('message', on_message)
        script.load()
        sys.stdin.read()
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == '__main__':
    main()
