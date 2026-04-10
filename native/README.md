# llama.cpp Vendored Source

This directory contains the vendored llama.cpp source for native mobile compilation.

## Setup

```bash
# Clone llama.cpp as a submodule (or direct clone)
git submodule add https://github.com/ggerganov/llama.cpp.git native/llama-cpp

# Or download a specific release
wget https://github.com/ggerganov/llama.cpp/archive/refs/tags/b4728.tar.gz
tar xf b4728.tar.gz -C native/llama-cpp --strip-components=1
```

## Build for Mobile

The build script `scripts/build-mobile.sh` handles cross-compilation:

```bash
# Build for Android (requires NDK)
./scripts/build-mobile.sh android

# Build for iOS (requires Xcode)
./scripts/build-mobile.sh ios

# Build both
./scripts/build-mobile.sh all
```

## Architecture

When `cargo build --features mobile-native` is used:

1. `build.rs` detects the target platform
2. CMake compiles llama.cpp with appropriate GPU backend:
   - **iOS**: Metal + NEON
   - **Android**: OpenCL (Adreno) + NEON
3. The static library is linked into the Rust binary
4. `src/mobile/llama_ffi.rs` provides safe Rust wrappers

## Required Dependencies

- **Android**: Android NDK 27+, CMake 3.24+
- **iOS**: Xcode 15+, CMake 3.24+
- **Both**: Rust nightly (for `aarch64-*` targets)
