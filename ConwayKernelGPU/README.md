# Metal Packed Life (Zoom/Pan)

Minimal macOS Cocoa (Objective-C/ObjC++) app.

## Features
- 1 bit per cell packed into `uint32` words
- W/H are powers of two (wraparound via bitmask)
- Custom N×N kernel (N ≤ 16) via 16× `uint16` row masks (shareable as 64 hex chars)
- Neighbor count via bit ops + `popcount`
- Packed output via `simdgroup_ballot` (no atomics)
- Viewport decode compute shader with zoom + pan

## Build (Xcode)
1. New Project: macOS → App, Language: Objective-C.
2. Add all files in this folder to the target.
3. Ensure `LifeKernels.metal` is included in the target (Build Phases → Compile Sources / Metal).
4. Run.

## Controls
- Drag: pan
- Scroll: zoom (anchored under cursor)

## Kernel hex
64 hex chars (16 rows × 4 hex chars). Whitespace/newlines allowed.

Example: full 16×16 window (includes center):
`FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF`

If you want "neighbors only" with that, pass `subtractCenter=YES`.
