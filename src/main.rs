use std::arch::asm;

fn main() {
    // Execute a NOP via inline assembly to prove the toolchain works
    unsafe {
        asm!("nop");
    }
    println!("alive");
}
