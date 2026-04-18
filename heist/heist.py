import frida
import sys
import json
import time

# Static Module Sweep: Brute-force math libraries for AMX/SME patterns
JS_CODE = """
'use strict';

const TARGET_MODULES = [
    "libLinearAlgebra.dylib",
    "libvDSP.dylib",
    "libBNNS.dylib"
];

function staticSweep() {
    console.log("[*] Commencing Static Sweep for AMX/SME Opcodes...");
    let foundOpcodes = new Set();
    let details = [];

    TARGET_MODULES.forEach(modName => {
        let mod = null;
        try {
            mod = Process.getModuleByName(modName);
        } catch (e) {
            console.log(`[!] Module ${modName} not found.`);
            return;
        }

        console.log(`[*] Sweeping ${modName} @ ${mod.base}...`);
        
        let ranges = mod.enumerateRanges('r-x');
        ranges.forEach(range => {
            let address = range.base;
            let end = range.base.add(range.size);

            while (address.compare(end) < 0) {
                let opcode = address.readU32();

                // Mask for Legacy AMX (0x0020xxxx) or ARM SME (0x8080xxxx)
                // Broadened masks to catch variants
                if ((opcode & 0xFFF00000) === 0x00200000 || 
                    (opcode & 0xFFF00000) === 0x80800000 ||
                    (opcode & 0xFF000000) === 0x80000000) { // Catch more SME
                    
                    let hex = "0x" + (opcode >>> 0).toString(16).padStart(8, '0');
                    if (!foundOpcodes.has(hex)) {
                        foundOpcodes.add(hex);
                        details.push({
                            module: modName,
                            address: address,
                            opcode: hex
                        });
                    }
                }
                address = address.add(4);
            }
        });
    });

    console.log(`[+] Sweep complete. Found ${foundOpcodes.size} unique candidate opcodes.`);
    send({ type: 'opcodes', data: Array.from(foundOpcodes) });
}

staticSweep();
"""

def on_message(message, data):
    if message['type'] == 'send':
        payload = message['payload']
        if payload['type'] == 'opcodes':
            print(f"[*] Total unique opcodes captured: {len(payload['data'])}")
            if len(payload['data']) > 0:
                with open('amx_opcodes.json', 'w') as f:
                    json.dump(payload['data'], f, indent=2)
                print("[*] Results saved to amx_opcodes.json")
            else:
                print("[!] No opcodes were captured.")
            sys.exit(0)
    else:
        print(message)

def main():
    try:
        print("[*] Attaching to amx_runner for static sweep...")
        session = frida.attach("amx_runner")
        script = session.create_script(JS_CODE)
        script.on('message', on_message)
        script.load()
        # The script executes immediately on load
        # We just need to wait for it to finish and send the message
        time.sleep(2)
        session.detach()
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == '__main__':
    main()
