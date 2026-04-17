# jit_explore

A project for exploring Apple's AMX (Apple Matrix) instructions through JIT and targeted exploration.

The core of this project uses a "heist" strategy to extract and verify AMX opcodes by running them in a controlled environment.

## Structure

- `src/`: Rust core for JIT emission and state management.
- `heist/`: Scripts and runners for extracting AMX instruction data.
- `planning/`: Documentation on research and strategy.
- `output/`: (Ignored) Generated instruction data and binaries.
