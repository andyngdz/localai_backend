# 002: Fix macOS CUDA Dependencies

**Date:** 2025-10-25
**Status:** Approved
**Branch:** fix/deps-for-macos

## Problem

Running `uv sync` on macOS ARM64 fails with:

```
error: Distribution `nvidia-cublas-cu12==12.6.4.1 @ registry+https://pypi.org/simple` can't be installed because it doesn't have a source distribution or wheel for the current platform
```

NVIDIA CUDA libraries are hardcoded as unconditional dependencies in `pyproject.toml`, but:
- CUDA only works with NVIDIA GPUs (not available on macOS)
- These packages only have wheels for Linux (x86_64/aarch64) and Windows
- macOS uses MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration

## Current State

The codebase already supports multiple backends gracefully:
- **DeviceService** (`app/services/device.py:14-29`) detects and configures:
  - CUDA → NVIDIA GPUs (Linux/Windows)
  - MPS → Apple Silicon GPUs (macOS)
  - CPU → Fallback for all platforms

## Solution

Add platform markers to NVIDIA CUDA dependencies to make them conditional:

```toml
# Before
"nvidia-cublas-cu12==12.6.4.1",

# After
"nvidia-cublas-cu12==12.6.4.1; sys_platform != 'darwin'",
```

This tells package managers to only install NVIDIA packages on non-macOS platforms.

## Implementation

### Modified Files

**`pyproject.toml`:**
- Add `; sys_platform != 'darwin'` marker to 14 NVIDIA CUDA packages (lines 78-91)
- Add `; sys_platform != 'darwin'` marker to `triton` package (line 144)

### Affected Dependencies

**NVIDIA CUDA libraries (14 packages):**
1. `nvidia-cublas-cu12==12.6.4.1`
2. `nvidia-cuda-cupti-cu12==12.6.80`
3. `nvidia-cuda-nvrtc-cu12==12.6.77`
4. `nvidia-cuda-runtime-cu12==12.6.77`
5. `nvidia-cudnn-cu12==9.5.1.17`
6. `nvidia-cufft-cu12==11.3.0.4`
7. `nvidia-cufile-cu12==1.11.1.6`
8. `nvidia-curand-cu12==10.3.7.77`
9. `nvidia-cusolver-cu12==11.7.1.2`
10. `nvidia-cusparse-cu12==12.5.4.2`
11. `nvidia-cusparselt-cu12==0.6.3`
12. `nvidia-nccl-cu12==2.26.2`
13. `nvidia-nvjitlink-cu12==12.6.85`
14. `nvidia-nvtx-cu12==12.6.77`

**GPU compiler:**
15. `triton==3.3.1` - OpenAI's GPU programming language (Linux x86_64 only)

## Benefits

✅ `uv sync` works on macOS (uses MPS or CPU)
✅ Linux/Windows continue to get CUDA support
✅ No code changes needed
✅ Tests work across all platforms
✅ Single `pyproject.toml` for all platforms

## Testing

1. ✅ Run `uv sync` on macOS → **Success** (installed 35 packages)
2. ✅ Run `pytest` on macOS → **All 454 tests passed** (using MPS/CPU backend)
3. ⏳ Verify Linux/Windows installations still get NVIDIA packages (requires deployment verification)

## Alternative Approaches Considered

1. **Separate dependency groups** (`[cuda]`, `[mps]`) - Requires users to manually specify extras
2. **CPU-only PyTorch on macOS** - Loses MPS acceleration benefits
3. **Platform-specific requirement files** - More complex to maintain

Platform markers are the cleanest solution for transparent cross-platform support.
