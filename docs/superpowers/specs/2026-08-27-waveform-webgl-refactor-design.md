# Waveform WebGL Refactor Design

## Status

Approved for implementation on branch `feature/waveform-webgl-refactor`.

This document is based on the current `develop` implementation at commit
`debb97a`. Earlier waveform design documents are historical only and are not
inputs to this design.

## Goal

Replace the current per-track Canvas 2D waveform path with a shared WebGL2
rendering system that:

- eliminates waveform flicker during horizontal scrolling and zooming;
- bounds frontend, backend, and GPU waveform memory;
- prevents stale asynchronous responses from repopulating invalid caches;
- uses GPU acceleration on Windows, macOS, and Linux;
- falls back to Canvas 2D when WebGL2 is unavailable or loses its context;
- uses the same scene, data, and rendering rules in the timeline and parameter
  editor;
- preserves current clip semantics and visual behavior.

## Success Criteria

The target workload is 80 tracks and 5000 clips.

- Horizontal scrolling and zooming target 60 FPS on supported hardware.
- A presented frame must never contain a partially cleared waveform surface.
- Mipmap refinement may improve detail, but must not make an already visible
  waveform disappear while a finer level loads.
- After a 30-minute scroll, zoom, edit, project-switch, and source-replacement
  stress run, waveform CPU and GPU memory must settle within configured byte
  budgets instead of growing with interaction count.
- A WebGL context loss must not crash the application. Rendering must continue
  through Canvas 2D or recover after context restoration.
- Timeline and parameter-editor waveforms must agree for normal, reversed,
  stretched, looped, faded, muted, and overlapping clips.

## Scope

### In Scope

- Timeline waveform rendering.
- Parameter-editor background waveform rendering.
- Shared clip-to-waveform scene construction.
- Frontend mipmap loading and cache ownership.
- New manifest and tiled waveform commands.
- Bounded backend in-memory waveform cache.
- WebGL2 renderer and Canvas 2D fallback renderer.
- Viewport scheduling and frame presentation.
- Diagnostics, unit tests, browser tests, and stress-test instrumentation.

### Out of Scope

- ONNX execution-provider selection and inference acceleration.
- Changes to audio playback or synthesis.
- Changes to the HFSPeaks peak calculation algorithm or its three division
  factors unless a correctness bug is found while adding tiled reads.
- Moving all timeline clip chrome from Canvas 2D to WebGL.
- Browser WebGPU. The frontend renderer intentionally uses WebGL2.

## Existing System

The current waveform path is:

```text
Rust HfsPeakFile
  -> Base64 whole-level Tauri response
  -> WaveformMipmapStore full-level Float32Array cache
  -> per-visible-track WaveformTrackCanvas
  -> interleaved scratch buffer
  -> CPU downsample scratch buffer
  -> CPU gain/fade scratch buffer
  -> Canvas 2D fillRect calls
```

The parameter editor duplicates the clip-window, loop-segmentation, mipmap,
gain, and Canvas 2D rendering logic in `pianoRoll/render.ts`.

The project does contain WebGPU support, but only in the Rust ONNX Runtime
execution-provider path. That backend path is platform-specific:

- Windows uses DirectML.
- Linux x86_64 can use WebGPU through Dawn/Vulkan.
- Apple Silicon can use CoreML and WebGPU through Dawn/Metal.

There is no frontend WebGL or WebGPU renderer today. The waveform renderer
therefore uses WebGL2 as a new frontend capability and leaves backend inference
configuration unchanged.

## Root-Cause Analysis

### 1. Two Viewport Timelines

`WaveformTrackCanvas` receives viewport props from React while also updating
refs and its DOM transform through `timelineViewportBus`. Its memo comparator
intentionally ignores `pxPerSec`, `viewportStartSec`, and `viewportEndSec`.
The bus stores no current snapshot, so a virtualized track mounted after a
horizontal scroll cannot initialize itself from the current scroll transform.
It can remain misplaced until another bus event arrives.

Zooming is committed through React and scroll correction is applied later in a
layout effect. During that interval, mounted canvases can observe different
combinations of scale, scroll, dimensions, and clip slices.

### 2. Non-Atomic Multi-Canvas Presentation

Every visible track owns a canvas, a store listener, and a requestAnimationFrame
queue. A scroll or zoom invalidates those canvases independently. Each canvas
clears before rebuilding its waveform. Slow tracks, cache misses, and browser
compositing can expose cleared or temporally inconsistent track surfaces.

### 3. Loading Can Remove Existing Detail

The store requests a preferred mipmap level during drawing. It can fall back to
a loaded level, but cache eviction, asynchronous level changes, and per-track
redraw timing are not coordinated with frame presentation. A draw can clear the
surface before discovering that required data is not currently available.

### 4. Allocation Churn and Weak Ownership

The CPU path creates or borrows interleaved, downsample, and gain-applied arrays
for visible clip segments. Ownership is spread across nested loops and local
maps. The current code contains careful release calls, but adding an early
return or exception can strand a buffer. Pool limits are entry counts, so a few
large backing buffers can retain much more memory than intended.

### 5. Cache Limits Do Not Bound Bytes End to End

The frontend full-level cache uses a file-count LRU. The backend
`waveform_cache_v2` is an unbounded `HashMap<String, Arc<HfsPeakFile>>`.
Backend batch preload serializes all three full levels although the frontend
only consumes L2. Long media files and long-running sessions can therefore
retain or transiently allocate large amounts of memory.

### 6. Stale Async Work Has No Generation Barrier

Cache invalidation removes current entries but does not invalidate the identity
of already running requests. A response from an old project or old file
revision can complete later and write data back into the current store.

## Architecture

The new system has five units with narrow interfaces.

### 1. Waveform Scene Builder

`WaveformSceneBuilder` is pure TypeScript with no DOM, Canvas, or WebGL
dependency. It consumes clip metadata and one immutable viewport snapshot and
produces a `WaveformFrame`.

It owns all shared waveform semantics:

- visible clip intersection;
- source playback windows;
- forward and reversed mapping;
- playback-rate mapping;
- loop head and repeated-body segmentation;
- media-domain silence outside non-loop source bounds;
- effective manual or automatic fades;
- mute and overlap alpha;
- track-row placement;
- source tile requirements;
- loop and media-boundary marker positions.

The timeline and parameter editor provide different vertical layouts and
colors, but use the same scene builder. Neither renderer re-derives clip time
math.

The core output types are:

```ts
interface WaveformViewportSnapshot {
    revision: number;
    scrollLeftPx: number;
    pxPerSec: number;
    widthPx: number;
    heightPx: number;
    devicePixelRatio: number;
}

interface WaveformDrawSegment {
    clipId: string;
    sourceRevision: string;
    level: 0 | 1 | 2;
    sourceStartSec: number;
    sourceEndSec: number;
    timelineStartSec: number;
    timelineDurationSec: number;
    trackTopPx: number;
    trackHeightPx: number;
    playbackRate: number;
    reversed: boolean;
    gain: number;
    fadeInSec: number;
    fadeOutSec: number;
    fadeInCurve: FadeCurveType;
    fadeOutCurve: FadeCurveType;
    alpha: number;
    color: string;
}

interface WaveformFrame {
    viewport: WaveformViewportSnapshot;
    segments: readonly WaveformDrawSegment[];
    requiredTiles: readonly WaveformTileKey[];
    markers: readonly WaveformMarker[];
}
```

All frame arrays are treated as immutable after construction.

### 2. Waveform Viewport Store

Each waveform surface receives a viewport source with:

```ts
interface WaveformViewportSource {
    getSnapshot(): WaveformViewportSnapshot;
    subscribe(listener: () => void): () => void;
}
```

The timeline scroll controller writes scale, scroll, viewport size, DPR, and a
monotonic revision in one operation. New surfaces synchronously read the latest
snapshot on mount, which removes the current event-bus initialization hole.

The parameter editor owns a separate viewport source. Optional timeline/editor
viewport synchronization writes complete snapshots to both sources; it does
not make either renderer read a mixture of the two views.

React remains responsible for clip structure and layout. High-frequency
viewport updates go directly through the viewport source and one surface-level
rAF scheduler.

### 3. Waveform Data Store

`WaveformDataStore` owns manifests, peak tiles, request de-duplication,
generation checks, weighted LRU accounting, and data-ready notifications.

It exposes read-only leases rather than raw cache ownership:

```ts
interface WaveformTileLease {
    key: WaveformTileKey;
    minMax: Float32Array;
    release(): void;
}
```

A frame pins the tiles it uses until presentation completes. Eviction cannot
remove pinned or in-flight-upload tiles.

### 4. Renderer Interface

Both backends consume the same frame:

```ts
interface WaveformRenderer {
    readonly kind: "webgl2" | "canvas2d";
    resize(viewport: WaveformViewportSnapshot): void;
    render(frame: WaveformFrame, data: WaveformFrameData): RenderResult;
    releaseSource(sourceRevision: string): void;
    dispose(): void;
}
```

`WebGL2WaveformRenderer` is the preferred implementation.
`Canvas2DWaveformRenderer` is the fallback and visual reference.

### 5. Waveform Surface

A `WaveformSurface` coordinates one viewport source, scene builder, data store,
renderer, and rAF scheduler.

- The timeline has one surface covering all virtualized visible rows.
- The parameter editor has one surface beneath its existing editing canvas.
- The global data store is shared.
- WebGL contexts and framebuffers are surface-local.
- GPU textures are context-local, while a shared GPU budget coordinator counts
  and evicts texture copies across both contexts.
- A surface has one store subscription and one rAF, regardless of track count.

## WebGL2 Rendering Pipeline

### Context

The preferred context is created with:

```ts
canvas.getContext("webgl2", {
    alpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: true,
    preserveDrawingBuffer: false,
    powerPreference: "high-performance",
});
```

WebGL1 is not a secondary path. If WebGL2 initialization or shader compilation
fails, the surface selects Canvas 2D.

### Peak Textures

Peak min/max pairs are uploaded in fixed tiles of 4096 peak pairs. GPU tiles use
two-channel signed normalized 16-bit textures (`RG16_SNORM`) after finite-value
validation and clamping to `[-1, 1]`. Gain and fades remain shader parameters,
so amplified values are not clipped before display.

The existing division factors remain `16`, `512`, and `4096`. With the current
selection thresholds, one output pixel needs at most 32 source peak pairs.
Shader aggregation therefore uses a fixed upper bound and a per-instance count.

### Drawing

Each visible waveform column is an instanced quad. Per-segment instance data
contains screen bounds, source mapping, tile offset, direction, gain, fades,
alpha, and color. The vertex shader maps a column to a bounded peak range,
aggregates min/max, applies gain and fade functions, and emits the vertical
quad. The fragment shader applies color and alpha.

Segments are grouped by texture tile/page. One frame uses a bounded number of
buffer updates and draw calls instead of one draw loop per track or clip.

Loop markers and media-boundary markers may remain on the existing overlay
Canvas initially. Their positions come from `WaveformFrame.markers`, so the
time semantics are still shared.

### Atomic Presentation

The WebGL renderer draws into a surface-sized framebuffer texture. Only after
all available batches complete does it blit that framebuffer to the default
framebuffer. The visible framebuffer is never cleared before a replacement
frame is complete.

The Canvas 2D backend uses two surface-sized canvases and swaps presentation
only after drawing completes. It does not use one offscreen canvas per clip or
per track.

### Missing and Refining Data

The data store resolves requested levels in this order:

1. requested tile at the selected level;
2. an already resident coarser tile covering the same source interval;
3. an already resident finer tile;
4. the last valid texture for the same source revision;
5. no waveform for only that segment.

L2 is prefetched for project sources. Refining from L2 to L1/L0 changes detail
only after the replacement tile is resident. A temporary miss never clears the
whole surface or removes unrelated tracks.

## Backend Data API

The existing full-level Base64 commands remain temporarily for compatibility
while consumers migrate. New commands are:

```text
get_waveform_manifest(source_path) -> WaveformManifest
get_waveform_tiles_binary(source_path, revision, requests[]) -> String
```

`WaveformManifest` contains:

- canonical source identity and revision fingerprint;
- sample rate, total frames, channels, and duration;
- each mipmap level's division factor and peak count;
- tile size and tile count.

`get_waveform_tiles_binary` returns only requested tile ranges in one binary
envelope, Base64 encoded because current Tauri invocation serializes `Vec<u8>`
as JSON numbers. Requests are batched per animation turn.

The initial implementation can slice tiles from a bounded cached
`HfsPeakFile`. Disk format and peak generation remain compatible. A later
memory-mapped disk reader is unnecessary unless profiling shows backend tile
reads are still a bottleneck.

## Memory Management

### Default Budgets

- Frontend CPU peak tiles: 192 MiB.
- WebGL peak textures across all surfaces: 128 MiB.
- Canvas 2D scratch storage: 16 MiB.
- Rust in-memory `HfsPeakFile` cache: 256 MiB.

Budgets are based on actual backing-store or resource byte estimates, not entry
counts. Constants are centralized and visible in diagnostics.

### Weighted LRU

Frontend and backend caches track:

- key;
- byte cost;
- last-used sequence;
- pin/reference count;
- load state.

Eviction runs after insert and after lease release. Pinned entries are skipped.
If one entry exceeds a budget, it may be the sole resident entry while in use
and is evicted as soon as its final lease ends.

WebGL objects cannot be shared between the timeline and parameter-editor
contexts. Each context therefore owns its texture objects, registers their byte
cost with one global GPU budget coordinator, and accepts eviction callbacks
from that coordinator. The shared CPU tile remains the upload source when a
surface needs to recreate an evicted texture.

### Scratch Memory

The WebGL path removes per-frame interleaved copies, CPU downsampling buffers,
and gain buffers. Upload conversion buffers use a byte-budgeted allocator and
are released in `finally` blocks.

Canvas 2D scratch arrays are also byte-budgeted. Buffers larger than half the
scratch budget are never pooled after use.

### Project and Source Lifecycle

Opening or creating a project starts a new project generation. Source
replacement starts a new source revision. Clearing the waveform cache starts a
new generation and clears all unpinned entries. Surface unmount releases all
frame leases and renderer resources.

## Async Correctness

Every request carries:

```text
projectGeneration + sourceRevision + level + tileRange
```

The store validates all four fields before insertion. Stale responses are
counted and discarded without notifying surfaces.

The request scheduler:

- de-duplicates identical manifest and tile requests;
- caps backend IPC concurrency;
- prioritizes visible tiles over overscan and refinement;
- batches adjacent tile requests;
- stops scheduling work for generations with no live surfaces;
- uses `AbortController` for frontend work where supported, while generation
  checks remain the final correctness barrier for non-cancelable Tauri calls.

All load-state cleanup occurs in `finally`, including failure and stale-result
paths.

## Integration

### Timeline

The timeline adds one `TimelineWaveformSurface` aligned with the existing
virtualized track-canvas viewport. It receives the visible track window and
visible clip slices already produced by `buildTimelineRenderModel`.

`WaveformTrackCanvas` is removed after parity is established. TrackLane no
longer creates waveform canvases or subscribes to global waveform events.

The surface's horizontal position is the viewport's `scrollLeft`, while all
waveform x coordinates are local viewport pixels. Vertical placement uses the
visible-track index. Mounting after horizontal or vertical virtualization is
correct because the surface reads the current viewport snapshot synchronously.

### Parameter Editor

The parameter editor inserts a transparent `PianoRollWaveformSurface` below its
existing note/curve editing canvas. The old background-waveform block is
removed from `pianoRoll/render.ts`; all other parameter-editor drawing remains
Canvas 2D.

Both surfaces share source manifests, CPU peak tiles, mipmap selection, loop
segmentation, fade semantics, and data-ready notifications.

## Failure Handling

### WebGL Initialization Failure

Shader compilation, program linking, framebuffer completeness, and required
format support are checked during renderer creation. Any failure is recorded
once and selects Canvas 2D for that surface.

### Context Loss

On `webglcontextlost`:

- call `preventDefault()`;
- stop issuing WebGL commands;
- invalidate the surface GPU resource ledger;
- render subsequent frames with Canvas 2D.

On `webglcontextrestored`, the surface attempts one clean WebGL recreation.
Resident CPU tiles are uploaded lazily. Repeated restore failures keep Canvas
2D active for the remainder of the surface lifetime.

### Data and Decode Failure

A failed source revision is negatively cached for the current generation.
Failures affect only that source. Other segments continue rendering. Replacing
the source or changing its fingerprint clears the negative entry naturally.

## Diagnostics

The existing waveform debug switch is extended to report:

- selected renderer per surface;
- frame build, upload, GPU draw, Canvas 2D draw, and present times;
- requested, resolved, fallback, and missing tiles;
- CPU tile bytes, GPU texture bytes, scratch bytes, pins, and evictions;
- pending, de-duplicated, failed, canceled, and stale requests;
- WebGL context-loss and restore counts;
- viewport revision and dropped invalidations.

Diagnostics use no recurring timer when disabled and are disposed with their
surface/store owner.

## Testing Strategy

### Pure Unit Tests

- Weighted LRU byte accounting, pinning, oversize entries, and eviction order.
- Generation/revision rejection of stale asynchronous responses.
- Request de-duplication and priority ordering.
- Mipmap selection and coarser/finer fallback.
- Viewport snapshot initialization and atomic updates.
- Scene construction for normal, reversed, stretched, looped, silent-tail,
  faded, muted, and overlapping clips.
- Timeline and parameter-editor scene parity for the same clip data.

### Renderer Tests

- Shader compilation and framebuffer completeness in a real browser.
- Deterministic pixel comparisons between WebGL2 and Canvas 2D for compact
  fixtures, with a small antialiasing tolerance.
- Canvas remains nonblank during repeated scroll and zoom snapshots.
- Forced `WEBGL_lose_context` switches to fallback and later recovers.
- DPR and resize changes do not accumulate stale resources.

### Integration and Stress Tests

- 80 tracks and 5000 clips with deterministic source fixtures.
- Continuous horizontal scroll and anchored zoom.
- Vertical virtualization while horizontally scrolled.
- Project switching and repeated source replacement during in-flight loads.
- 30-minute scripted stress run with periodic memory samples.

The stress test passes when cache ledgers remain within budget, pending work
returns to zero after settling, stale responses do not change a current frame,
and no sustained upward memory trend remains after warm-up and garbage
collection opportunities.

## Migration Sequence

1. Add pure weighted caches, generation identity, viewport snapshots, and scene
   builder with tests.
2. Add manifest/tile backend commands and bounded backend cache.
3. Add Canvas 2D surface renderer as the parity reference.
4. Replace timeline per-track waveform canvases with one Canvas 2D surface.
5. Move parameter-editor background waveforms to the shared surface.
6. Add WebGL2 resource cache, shaders, framebuffer presentation, and fallback.
7. Add context-loss handling and diagnostics.
8. Remove migrated full-level loading, per-track listeners, buffer pools, and
   duplicated piano-roll waveform rendering.
9. Run parity, browser, build, and stress verification on supported platforms.

Each migration step leaves a working fallback path. The old per-track renderer
is removed only after the shared Canvas 2D surface matches current behavior;
the WebGL2 renderer is enabled only after pixel and context-loss tests pass.

## Final Decisions

- Frontend waveform GPU API: WebGL2.
- Frontend fallback: Canvas 2D.
- Backend inference GPU APIs: unchanged.
- Rendering surfaces: one timeline surface and one parameter-editor surface.
- Shared logic: scene builder, data store, cache policy, and renderer contract.
- Frame presentation: offscreen framebuffer/double-buffer, then atomic present.
- Cache policy: weighted LRU with explicit byte budgets and pinning.
- Async policy: project generation plus source revision validation.
- Primary performance target: 80 tracks, 5000 clips, near 60 FPS.
