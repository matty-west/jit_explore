import frida
import sys
import json
import time

# Script to find the hot microkernel address by hooking EVERYTHING in vecLib
JS_CODE = """
'use strict';

function startTracing() {
    const threadId = Process.getCurrentThreadId();
    Stalker.follow(threadId, {
        events: {
            call: true,
            ret: false
        },
        onReceive(events) {
            const parsed = Stalker.parse(events, { annotate: true, stringify: false });
            for (const event of parsed) {
                if (event[0] === 'call') {
                    const target = event[2];
                    const targetPtr = ptr(target);
                    try {
                        const m = Process.getModuleByAddress(targetPtr);
                        console.log('[>] CALL to ' + m.name + ' @ ' + targetPtr);
                    } catch (e) {
                        // console.log('[>] CALL to ' + targetPtr);
                    }
                }
            }
        }
    });
}

let target = null;
try {
    target = Module.findExportByName(null, 'cblas_sgemm') || Module.findExportByName(null, '_cblas_sgemm');
} catch (e) {}

if (target) {
    Interceptor.attach(target, {
        onEnter(args) {
            console.log('[*] cblas_sgemm entered. Starting call trace...');
            startTracing();
        },
        onLeave(retval) {
            Stalker.unfollow(Process.getCurrentThreadId());
            console.log('[*] cblas_sgemm left.');
        }
    });
}
"""

def on_message(message, data):
    print(message)

def main(pid):
    try:
        session = frida.attach(pid)
        script = session.create_script(JS_CODE)
        script.on('message', on_message)
        script.load()
        print(f"[*] Attached to PID {pid}. Ready. PRESS ENTER in amx_runner...")
        sys.stdin.read()
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 heist.py <pid>")
        sys.exit(1)
    main(int(sys.argv[1]))
