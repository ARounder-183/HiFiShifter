# Waveform WebGL Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-track CPU waveform canvases with shared timeline and parameter-editor waveform surfaces using WebGL2, Canvas 2D fallback, tiled peak loading, atomic presentation, and bounded memory.

**Architecture:** Pure scene construction converts clips plus immutable viewport snapshots into renderer-neutral frames. A generation-aware tiled data store supplies byte-budgeted CPU leases, while each surface owns either a WebGL2 renderer with context-local textures or an atomic Canvas 2D fallback. Rust exposes fingerprinted manifests and tile batches from a bounded in-memory HFSPeaks cache.

**Tech Stack:** React 19, TypeScript 5.9, WebGL2, Canvas 2D, Tauri 2, Rust, self-executing TypeScript tests, Rust unit tests

**Spec:** `docs/superpowers/specs/2026-08-27-waveform-webgl-refactor-design.md`

## Global Constraints

- Frontend waveform GPU rendering uses WebGL2, not browser WebGPU.
- Existing Rust ONNX WebGPU, CoreML, and DirectML behavior remains unchanged.
- WebGL2 failure or context loss must keep rendering available through Canvas 2D.
- Timeline and parameter editor share scene semantics, data store, cache policy, and renderer contract.
- Timeline uses one waveform surface for all visible rows; parameter editor uses one waveform surface.
- Visible frames are presented atomically and are not cleared while replacement data is missing.
- Peak tiles contain 4096 min/max pairs.
- Default budgets are 192 MiB frontend CPU tiles, 128 MiB GPU textures, 16 MiB Canvas scratch, and 256 MiB Rust waveform cache.
- Cache limits use actual byte costs and pinning, not entry counts.
- Async insertion requires matching project generation, source revision, level, and tile range.
- Preserve existing normal, reversed, stretched, looped, silent-tail, fade, mute, overlap, and marker semantics.
- Target workload is 80 tracks and 5000 clips with near-60-FPS horizontal scroll and zoom.
- A 30-minute stress run must settle within cache budgets and leave no pending waveform requests.

---

## File Structure

### New Frontend Runtime

- `frontend/src/waveform/types.ts`: manifests, tile keys, segments, frames, leases, and renderer interfaces.
- `frontend/src/waveform/weightedLru.ts`: reusable byte-budgeted pinned LRU.
- `frontend/src/waveform/viewportStore.ts`: immutable current-value viewport source.
- `frontend/src/waveform/sceneBuilder.ts`: clip window, loop segmentation, marker, and tile requirement logic.
- `frontend/src/waveform/tileCodec.ts`: tiled binary envelope decoder.
- `frontend/src/waveform/dataStore.ts`: generation-aware request scheduler and CPU cache.
- `frontend/src/waveform/frameData.ts`: tile-backed draw-piece resolution.
- `frontend/src/waveform/renderers/canvas2dRenderer.ts`: atomic fallback and visual reference.
- `frontend/src/waveform/renderers/webgl2Renderer.ts`: shaders, textures, framebuffer presentation, and disposal.
- `frontend/src/waveform/renderers/gpuBudget.ts`: cross-context GPU byte accounting.
- `frontend/src/waveform/rendererFactory.ts`: WebGL2 selection and fallback.
- `frontend/src/waveform/WaveformSurface.tsx`: scene/data/render coordination.
- `frontend/src/waveform/diagnostics.ts`: opt-in counters and memory snapshots.

### Modified Consumers

- `frontend/src/components/layout/TimelinePanel.tsx`: mount one timeline surface.
- `frontend/src/components/layout/timeline/TrackLane.tsx`: remove per-track waveform surface.
- `frontend/src/components/layout/timeline/TimelineScrollArea.tsx`: publish complete viewport snapshots.
- `frontend/src/components/layout/timeline/hooks/useTimelineState.ts`: own current-value timeline viewport.
- `frontend/src/components/layout/PianoRollPanel.tsx`: mount one parameter-editor waveform surface.
- `frontend/src/components/layout/pianoRoll/render.ts`: remove duplicated waveform drawing.
- `frontend/src/components/layout/pianoRoll/useClipsPeaksForPianoRoll.ts`: stop owning peak loading.
- `frontend/src/services/api/waveform.ts` and `frontend/src/services/invoke.ts`: manifest/tile calls.
- Project, runtime, and source-replacement thunks: generation/revision lifecycle.

### Modified Backend

- `backend/src-tauri/src/audio/hfspeaks_v2.rs`: revision, manifest, byte size, and tile serialization.
- `backend/src-tauri/src/state.rs`: weighted 256 MiB waveform cache.
- `backend/src-tauri/src/commands/waveform.rs`: manifest and tile commands.
- `backend/src-tauri/src/commands.rs` and `backend/src-tauri/src/lib.rs`: command registration.

---

### Task 1: Byte-Budgeted Cache And Current-Value Viewport

**Files:**
- Create: `frontend/src/waveform/types.ts`
- Create: `frontend/src/waveform/weightedLru.ts`
- Create: `frontend/src/waveform/weightedLru.test.ts`
- Create: `frontend/src/waveform/viewportStore.ts`
- Create: `frontend/src/waveform/viewportStore.test.ts`

**Interfaces:**
- Produces: `WeightedLru<K, V>`, `createWaveformViewportStore(initial)`, `WaveformViewportSource`, and shared type declarations.
- Consumes: no later-task code.

- [ ] **Step 1: Write failing weighted-LRU tests**

Use literal costs and real leases:

```ts
const cache = new WeightedLru<string, string>(10);
cache.set("a", "A", 6);
cache.set("b", "B", 6);
assertEqual(cache.has("a"), false, "oldest unpinned entry is evicted by bytes");

cache.clear();
cache.set("pinned", "P", 8);
const lease = cache.acquire("pinned");
cache.set("new", "N", 8);
assertEqual(cache.has("pinned"), true, "pinned entry survives eviction");
lease?.release();
assertEqual(cache.totalBytes <= 10, true, "release enforces budget");

cache.clear();
cache.set("oversize", "O", 30);
const oversize = cache.acquire("oversize");
assertEqual(cache.totalBytes, 30, "sole pinned oversize entry may remain");
oversize?.release();
assertEqual(cache.totalBytes, 0, "released oversize entry is removed");
```

These catch entry-count eviction, eviction of leased values, and permanent
retention of released oversize values.

- [ ] **Step 2: Run the test and verify RED**

Run from `frontend/`:

```bash
node --experimental-strip-types src/waveform/weightedLru.test.ts
```

Expected: module/export missing.

- [ ] **Step 3: Implement `WeightedLru`**

```ts
export class WeightedLru<K, V> {
    constructor(readonly budgetBytes: number);
    get totalBytes(): number;
    get size(): number;
    has(key: K): boolean;
    set(key: K, value: V, bytes: number, onEvict?: (value: V) => void): void;
    peek(key: K): V | undefined;
    acquire(key: K): { value: V; release(): void } | null;
    delete(key: K): boolean;
    clear(): void;
}
```

Store monotonic access sequence, byte cost, pin count, and an idempotent
eviction callback. Evict least-recent unpinned entries until within budget.

- [ ] **Step 4: Write failing viewport-store tests**

```ts
const store = createWaveformViewportStore({
    revision: 0,
    scrollLeftPx: 0,
    pxPerSec: 150,
    widthPx: 1200,
    heightPx: 600,
    devicePixelRatio: 2,
});
store.set({ scrollLeftPx: 450, pxPerSec: 300, widthPx: 1000 });
assertDeepEqual(store.getSnapshot(), {
    revision: 1,
    scrollLeftPx: 450,
    pxPerSec: 300,
    widthPx: 1000,
    heightPx: 600,
    devicePixelRatio: 2,
}, "new subscribers see one complete latest snapshot");
```

Also assert one notification for one changed patch and no notification for an
identical patch. This catches the current event-only initialization hole.

- [ ] **Step 5: Verify RED, implement types/store, then verify GREEN**

Define `WaveformLevel`, `WaveformTileKey`, `WaveformViewportSnapshot`, and
`WaveformViewportSource`. `set()` merges a patch, normalizes finite positive
scale/dimensions/DPR, and increments revision only for a real change.

```bash
node --experimental-strip-types src/waveform/viewportStore.test.ts
node --experimental-strip-types src/waveform/weightedLru.test.ts
node --experimental-strip-types src/waveform/viewportStore.test.ts
npm run build
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/waveform
git commit -m "feat(waveform): add bounded cache and viewport store"
```

### Task 2: Shared Waveform Scene Builder

**Files:**
- Create: `frontend/src/waveform/sceneBuilder.ts`
- Create: `frontend/src/waveform/sceneBuilder.test.ts`
- Modify: `frontend/src/waveform/types.ts`
- Reuse: `frontend/src/utils/loopRender.ts`

**Interfaces:**
- Consumes: Task 1 viewport/types, `ClipInfo`, and existing loop helpers.
- Produces: `buildWaveformFrame(args): WaveformFrame`.

- [ ] **Step 1: Write failing literal scene tests**

Use hand-derived fixtures:

```ts
// Viewport [10s,20s] at 100 px/s; clip [8s,13s] -> screen x=[0,300].
assertDeepEqual(
    frame.segments[0].screenRect,
    { x: 0, y: 18, width: 300, height: 72 },
    "visible clip uses viewport-local coordinates",
);
// Forward: start=2,length=4,rate=2 -> source [2,10].
// Reversed: sourceEnd=9,length=3,rate=2 -> source [3,9].
// Loop: media=10,start=8,rate=1,length=15 -> [8,10],[0,10],[0,3].
```

Also assert automatic fade overrides manual fade, leading overlap splits alpha
at the exact boundary, media-domain silence requests no tile, and mipmap
hysteresis remains stable. These catch wrong reverse anchors, loop-window
semantics, and viewport/world coordinate mixing.

- [ ] **Step 2: Run scene test and verify RED**

```bash
node --experimental-strip-types src/waveform/sceneBuilder.test.ts
```

- [ ] **Step 3: Implement frame construction**

```ts
export interface BuildWaveformFrameArgs {
    viewport: WaveformViewportSnapshot;
    rows: readonly WaveformSceneRow[];
    manifests: ReadonlyMap<string, WaveformManifest>;
    strokeColor: string;
    previousLevelByClip: Map<string, WaveformLevel>;
}
```

Generate visible loop periods by direct indexing, cap malformed-period work at
4096 segments per visible clip, and derive 4096-pair tile indices from manifest
division factors. Frame arrays become immutable after construction.

- [ ] **Step 4: Verify GREEN and loop regressions**

```bash
node --experimental-strip-types src/waveform/sceneBuilder.test.ts
node --experimental-strip-types src/utils/loopRender.test.ts
```

If no standalone `loopRender.test.ts` exists, run every existing test that
imports `loopRender.ts` and record the exact commands.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/waveform/sceneBuilder.ts frontend/src/waveform/sceneBuilder.test.ts frontend/src/waveform/types.ts
git commit -m "feat(waveform): build shared renderer-neutral scenes"
```

### Task 3: Fingerprinted Manifest, Tile Protocol, And Bounded Rust Cache

**Files:**
- Modify: `backend/src-tauri/src/audio/hfspeaks_v2.rs`
- Modify: `backend/src-tauri/src/state.rs`
- Modify: `backend/src-tauri/src/commands/waveform.rs`
- Modify: `backend/src-tauri/src/commands.rs`
- Modify: `backend/src-tauri/src/lib.rs`

**Interfaces:**
- Produces: `WaveformManifestPayload`, `WaveformTileRequest`, `get_waveform_manifest`, `get_waveform_tiles_binary`, and `WaveformPeakCache`.
- Consumes: existing HFSPeaks disk format/computation.

- [ ] **Step 1: Add failing HFSPeaks tile tests**

Build an in-memory level with `min=[-1,-.5,0,.5,1]` and
`max=[1,.5,0,-.5,-1]`. Assert manifest fingerprint changes with source
fingerprint fields, tile size is 4096, tile 0 contains exactly five interleaved
pairs, and out-of-range tile indices emit no record.

- [ ] **Step 2: Run RED test**

```bash
cargo test waveform_tile_envelope_contains_only_requested_peaks --quiet
```

- [ ] **Step 3: Implement exact protocol**

```rust
pub const WAVEFORM_TILE_PEAKS: usize = 4096;
pub const WAVEFORM_TILE_MAGIC: &[u8; 4] = b"WFTL";
pub const WAVEFORM_TILE_VERSION: u16 = 1;
```

```text
magic[4], version:u16, tile_count:u16
repeat:
  level:u32, tile_index:u32, peak_start:u32, peak_count:u32,
  division_factor:u32, sample_rate:u32,
  interleaved min/max f32 little-endian[peak_count*2]
```

Revision is BLAKE3 of canonical path, source length, mtime nanoseconds, and
HFSPeaks version.

- [ ] **Step 4: Add failing cache tests**

Instantiate `WaveformPeakCache::new(10)`, insert synthetic 6-byte entries, and
assert byte eviction, refreshed LRU order, and `Arc::strong_count > 1` pinning.

- [ ] **Step 5: Run RED cache test**

```bash
cargo test waveform_peak_cache_evicts_by_bytes_and_respects_pins --quiet
```

- [ ] **Step 6: Implement 256 MiB weighted backend cache**

Replace the unbounded map. Track estimated bytes and last-use sequence. `get`
refreshes LRU; `insert` evicts least-recent entries whose Arc has no external
owner. Expose current bytes/entry count for diagnostics.

- [ ] **Step 7: Add commands**

```rust
pub fn get_waveform_manifest(state: State<'_, AppState>, source_path: String)
    -> Result<WaveformManifestPayload, String>;
pub fn get_waveform_tiles_binary(
    state: State<'_, AppState>, source_path: String, revision: String,
    requests: Vec<WaveformTileRequest>,
) -> Result<String, String>;
```

Reject revision mismatch before Base64 serialization. Keep legacy commands
until Task 9.

- [ ] **Step 8: Verify and commit**

```bash
cargo test waveform_tile --quiet
cargo test waveform_peak_cache --quiet
cargo check
git add backend/src-tauri/src/audio/hfspeaks_v2.rs backend/src-tauri/src/state.rs backend/src-tauri/src/commands/waveform.rs backend/src-tauri/src/commands.rs backend/src-tauri/src/lib.rs
git commit -m "feat(waveform): expose bounded tiled peak data"
```

### Task 4: Frontend Tile Codec And Generation-Aware Data Store

**Files:**
- Create: `frontend/src/waveform/tileCodec.ts`
- Create: `frontend/src/waveform/tileCodec.test.ts`
- Create: `frontend/src/waveform/dataStore.ts`
- Create: `frontend/src/waveform/dataStore.test.ts`
- Modify: `frontend/src/services/api/waveform.ts`
- Modify: `frontend/src/services/invoke.ts`
- Modify: `frontend/src/waveform/types.ts`

**Interfaces:**
- Consumes: Task 1 LRU and Task 3 protocol.
- Produces: strict decoder, `WaveformDataStore`, manifest lookup, tile leases, prefetch, generations, and invalidation.

- [ ] **Step 1: Write failing codec tests**

Build a literal one-tile envelope with two min/max pairs. Assert exact metadata
and values. Bad magic, truncation, count overflow, and malformed trailing
records must return typed errors rather than partial data.

- [ ] **Step 2: Run RED, implement strict decoder, verify GREEN**

Validate every offset with `DataView`, cap counts before allocation, and return
interleaved `Float32Array` views over the decoded buffer.

```bash
node --experimental-strip-types src/waveform/tileCodec.test.ts
node --experimental-strip-types src/waveform/tileCodec.test.ts
```

- [ ] **Step 3: Write failing store tests**

Inject deferred Promise transport and verify identical request de-duplication,
stale rejection after `beginGeneration()`, old-revision rejection, visible
priority ahead of overscan/refinement at concurrency 4, lease pinning, and CPU
bytes returning within 192 MiB after release.

- [ ] **Step 4: Run store test and verify RED**

```bash
node --experimental-strip-types src/waveform/dataStore.test.ts
```

- [ ] **Step 5: Implement store**

```ts
export class WaveformDataStore {
    constructor(args: { transport: WaveformTransport; cpuBudgetBytes?: number; maxConcurrent?: number });
    beginGeneration(): number;
    get generation(): number;
    ensureManifest(sourcePath: string): Promise<WaveformManifest | null>;
    requestTiles(requests: readonly WaveformTileNeed[]): void;
    acquireTile(key: WaveformTileKey): WaveformTileLease | null;
    invalidateSource(sourcePath: string): void;
    subscribe(listener: () => void): () => void;
    getStats(): WaveformDataStats;
    dispose(): void;
}
```

Coalesce adjacent requests per source/revision. Validate generation and
revision before insertion or notification. Cleanup every queue/load state in
`finally`.

- [ ] **Step 6: Add API mapping, verify, and commit**

```bash
node --experimental-strip-types src/waveform/dataStore.test.ts
npm run build
git add frontend/src/waveform frontend/src/services/api/waveform.ts frontend/src/services/invoke.ts
git commit -m "feat(waveform): load generation-safe peak tiles"
```

### Task 5: Frame Data And Atomic Canvas 2D Surface

**Files:**
- Create: `frontend/src/waveform/frameData.ts`
- Create: `frontend/src/waveform/frameData.test.ts`
- Create: `frontend/src/waveform/renderers/canvas2dRenderer.ts`
- Create: `frontend/src/waveform/renderers/canvas2dRenderer.browser.test.ts`
- Create: `frontend/src/waveform/browserTestMain.ts`
- Create: `frontend/waveform-test.html`
- Create: `frontend/src/waveform/rendererFactory.ts`
- Create: `frontend/src/waveform/WaveformSurface.tsx`
- Modify: `frontend/src/waveform/types.ts`

**Interfaces:**
- Consumes: frames, store, tile leases.
- Produces: tile-contained draw pieces, atomic fallback renderer, generic surface.

- [ ] **Step 1: Write failing draw-piece tests**

For peak indices 4090 through 4102, assert two pieces: tile 0 `[4090,4096)`
and tile 1 `[4096,4102)`. Their screen rectangles touch exactly. Reversal swaps
screen order while preserving clip-local fade time.

- [ ] **Step 2: Run RED, implement, verify GREEN**

`resolveWaveformFrameData(frame, store)` acquires exact or fallback tiles,
splits at boundaries, and returns one idempotent `release()`.

```bash
node --experimental-strip-types src/waveform/frameData.test.ts
node --experimental-strip-types src/waveform/frameData.test.ts
```

- [ ] **Step 3: Write failing browser atomic-presentation tests**

Assert rendering happens on a back canvas before one visible copy; missing data
returns `deferred` without clearing the visible canvas; a later complete frame
presents once; DPR resize disposes old backing surfaces.

- [ ] **Step 4: Implement Canvas 2D renderer**

Use one back canvas per surface. Aggregate directly from tile views per output
column, apply shared fades, draw markers, then copy once. Scratch storage is
byte-budgeted at 16 MiB and buffers over 8 MiB are not pooled.

- [ ] **Step 5: Implement `WaveformSurface` lifecycle**

Own one canvas, one data-store subscription, and one rAF. Read current viewport
synchronously, request missing tiles, preserve the last frame when deferred,
release leases in `finally`, and cancel/unsubscribe/dispose on unmount.

- [ ] **Step 6: Browser/build verification and commit**

Run the Vite diagnostic route at DPR 1 and 2, then:

```bash
npm run build
git add frontend/src/waveform
git commit -m "feat(waveform): add atomic shared canvas surface"
```

### Task 6: Replace Timeline Per-Track Canvases

**Files:**
- Create: `frontend/src/components/layout/timeline/TimelineWaveformSurface.tsx`
- Create: `frontend/src/components/layout/timeline/runtime/timelineWaveformRows.ts`
- Create: `frontend/src/components/layout/timeline/runtime/timelineWaveformRows.test.ts`
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`
- Modify: `frontend/src/components/layout/timeline/TrackLane.tsx`
- Modify: `frontend/src/components/layout/timeline/TimelineScrollArea.tsx`
- Modify: `frontend/src/components/layout/timeline/hooks/useTimelineState.ts`
- Modify: `frontend/src/components/layout/timeline/index.ts`

**Interfaces:**
- Consumes: shared surface, current viewport, virtual rows, visible clips.
- Produces: one timeline waveform surface.

- [ ] **Step 1: Write failing timeline adapter tests**

An 80-track model scrolled to row 40 must emit only visible/overscan rows with
local y starting at zero. A viewport set to horizontal scroll 900 before model
construction must immediately build with `scrollLeftPx=900`.

- [ ] **Step 2: Run RED and implement row adapter/wrapper**

```bash
node --experimental-strip-types src/components/layout/timeline/runtime/timelineWaveformRows.test.ts
```

The wrapper fills the existing virtual track viewport, uses local row y, theme
stroke color, leading-overlap maps, and `pointerEvents: none`.

- [ ] **Step 3: Publish complete viewport snapshots**

`TimelineScrollArea` writes scroll, scale, width, height, and DPR together
before React scheduling. New surfaces read current state at mount. Retain the
old bus only for the MIDI canvas compatibility consumer.

- [ ] **Step 4: Mount one surface and remove per-track mounts**

Place the surface beside `TimelineCanvasViewport`, below interactive DOM
overlays and above clip body backgrounds. Remove `WaveformTrackCanvas` from
`TrackLane` and remove now-unused props/imports.

- [ ] **Step 5: Verify and commit**

```bash
node --experimental-strip-types src/components/layout/timeline/runtime/timelineWaveformRows.test.ts
node --experimental-strip-types src/components/layout/timeline/runtime/timelineViewportDispatch.test.ts
node --experimental-strip-types src/components/layout/timeline/runtime/timelineWindowing.test.ts
npm run build
git add frontend/src/components/layout/TimelinePanel.tsx frontend/src/components/layout/timeline frontend/src/waveform
git commit -m "refactor(waveform): use one timeline rendering surface"
```

- [ ] **Step 6: Manual flicker regression**

Start horizontally scrolled, vertically mount new rows, and verify immediate
waveforms. Continuously zoom around pointer/playhead anchors and verify no blank
intermediate frame.

### Task 7: Share The Parameter-Editor Waveform Surface

**Files:**
- Create: `frontend/src/components/layout/pianoRoll/PianoRollWaveformSurface.tsx`
- Create: `frontend/src/components/layout/pianoRoll/pianoRollWaveformRows.ts`
- Create: `frontend/src/components/layout/pianoRoll/pianoRollWaveformRows.test.ts`
- Modify: `frontend/src/components/layout/PianoRollPanel.tsx`
- Modify: `frontend/src/components/layout/pianoRoll/render.ts`
- Modify: `frontend/src/components/layout/pianoRoll/useClipsPeaksForPianoRoll.ts`

**Interfaces:**
- Consumes: same scene/data/surface as Task 6.
- Produces: one noninteractive layer below the editing canvas.

- [ ] **Step 1: Write failing parity tests**

Normal, reversed, and looped fixtures must produce identical source ranges,
tile needs, fades, and marker timeline positions through timeline and piano
adapters. Only y, height, alpha, and color may differ.

- [ ] **Step 2: Run RED and implement wrapper**

```bash
node --experimental-strip-types src/components/layout/pianoRoll/pianoRollWaveformRows.test.ts
```

Mount the waveform canvas in the sticky viewport immediately before the editing
canvas. Keep all pointer handlers on the foreground editing canvas. Publish a
complete current viewport from existing scroll/zoom code.

- [ ] **Step 3: Remove duplicated drawing and loading**

Delete only the background waveform block and its waveform-specific module
state/imports from `render.ts`. Preserve every other piano-roll drawing path.
Remove waveform listeners/preload ownership from `useClipsPeaksForPianoRoll`.

- [ ] **Step 4: Verify visual parity and commit**

```bash
node --experimental-strip-types src/components/layout/pianoRoll/pianoRollWaveformRows.test.ts
npm run build
git add frontend/src/components/layout/PianoRollPanel.tsx frontend/src/components/layout/pianoRoll frontend/src/waveform
git commit -m "refactor(waveform): share parameter editor waveform surface"
```

Compare timeline/editor normal, reverse, stretch, loop, fade, crossfade, mute,
and out-of-media silence; verify editing still hits the foreground canvas.

### Task 8: WebGL2 Renderer, GPU Budget, And Context Recovery

**Files:**
- Create: `frontend/src/waveform/renderers/gpuBudget.ts`
- Create: `frontend/src/waveform/renderers/gpuBudget.test.ts`
- Create: `frontend/src/waveform/renderers/webgl2Renderer.ts`
- Create: `frontend/src/waveform/renderers/webgl2Shaders.ts`
- Create: `frontend/src/waveform/renderers/webgl2Renderer.browser.test.ts`
- Modify: `frontend/src/waveform/browserTestMain.ts`
- Modify: `frontend/src/waveform/rendererFactory.ts`
- Modify: `frontend/src/waveform/WaveformSurface.tsx`

**Interfaces:**
- Consumes: draw pieces and CPU leases.
- Produces: WebGL2 rendering, shared 128 MiB accounting, framebuffer presentation, and fallback/recovery.

- [ ] **Step 1: Write failing GPU budget tests**

Register resources from two fake context owners. Assert cross-owner LRU
eviction calls the correct owner-local delete callback, pinned resources
survive, and disposing one owner removes its bytes.

- [ ] **Step 2: Run RED, implement coordinator, verify GREEN**

```bash
node --experimental-strip-types src/waveform/renderers/gpuBudget.test.ts
node --experimental-strip-types src/waveform/renderers/gpuBudget.test.ts
```

Use owner-qualified keys with Task 1 LRU. Never share WebGL objects across
contexts.

- [ ] **Step 3: Write browser RED tests**

Render a literal four-column tile into 64x32. Assert nontransparent pixels at
hand-derived bounds and WebGL/Canvas masks within one edge pixel. Force
`WEBGL_lose_context`, assert fallback renders nonblank, restore, and assert the
surface either returns to WebGL2 or remains a functioning fallback.

- [ ] **Step 4: Implement shaders and renderer**

Use context attributes from the spec. Upload finite clamped peaks to
`RG16_SNORM`. Group pieces by texture and draw instanced unit quads. Aggregate
at most 32 pairs per output column, apply gain/fade/alpha in shader, render into
an RGBA framebuffer, and blit only after all batches complete. Check compile,
link, format, and framebuffer status.

- [ ] **Step 5: Implement context fallback/recovery**

On loss, prevent default, stop GL commands, clear GPU accounting, and switch to
Canvas 2D. On restore, attempt one clean renderer recreation and lazy uploads;
repeated failure keeps fallback.

- [ ] **Step 6: Verify pixels, layouts, build, and commit**

```bash
npm run build
git add frontend/src/waveform
git commit -m "feat(waveform): render shared surfaces with WebGL2"
```

Capture desktop/compact screenshots and canvas pixel counts. Verify nonblank,
aligned surfaces with no overlap over interactive controls.

### Task 9: Lifecycle, Diagnostics, Cleanup, And Stress Verification

**Files:**
- Create: `frontend/src/waveform/diagnostics.ts`
- Create: `frontend/src/waveform/stressScenario.ts`
- Create: `frontend/src/waveform/stressScenario.test.ts`
- Modify: `frontend/src/features/session/thunks/projectThunks.ts`
- Modify: `frontend/src/features/session/thunks/runtimeThunks.ts`
- Modify: `frontend/src/features/session/thunks/timelineThunks.ts`
- Modify: `frontend/src/utils/waveformDebug.ts`
- Delete after zero-reference check: `frontend/src/components/waveform/WaveformTrackCanvas.tsx`
- Remove obsolete full-level cache and scratch-pool paths only after consumers migrate.

**Interfaces:**
- Consumes: Tasks 1-8.
- Produces: lifecycle transitions, resource diagnostics, deterministic stress checks, and legacy cleanup.

- [ ] **Step 1: Write failing lifecycle/stress tests**

Start generation 1 with deferred transport, advance generation for project
open, resolve the old request, and assert no insertion/redraw. Replace the same
path with a new revision and assert only that source is invalidated.

Generate deterministic 80 tracks/5000 clips, simulate 10,000 scroll/zoom
snapshots plus project/source changes, release all frames, settle the scheduler,
and assert literal limits:

```ts
assert(stats.cpuBytes <= 192 * 1024 * 1024);
assert(stats.gpuBytes <= 128 * 1024 * 1024);
assert(stats.scratchBytes <= 16 * 1024 * 1024);
assertEqual(stats.pendingRequests, 0, "all requests settle");
assertEqual(stats.subscriberCount, baseline, "dispose releases subscriptions");
```

- [ ] **Step 2: Run RED and wire lifecycle**

```bash
node --experimental-strip-types src/waveform/stressScenario.test.ts
```

Advance generation after successful new/open/forced-open transitions before
new UI requests. Cache clear advances generation and clears unpinned entries.
Source replacement invalidates affected revisions without flushing unrelated
sources.

- [ ] **Step 3: Extend diagnostics**

Under `hifishifter.debugWaveform=1`, report renderer kind, viewport revision,
build/upload/draw/present times, exact/fallback/missing tiles, CPU/GPU/scratch
bytes, pins, evictions, pending/deduplicated/failed/stale requests, and context
loss/restoration. Disabled diagnostics create no timer.

- [ ] **Step 4: Remove obsolete paths after reference scan**

```bash
rg -n "WaveformTrackCanvas|applyGainsToPeaks|renderWaveform|getWaveformMipmapBinary|batchGetWaveformMipmap" frontend/src
```

Delete only zero-reference rendering paths. Normalization consumers may keep a
small compatibility accessor, but it must use the bounded new store rather
than a second cache.

- [ ] **Step 5: Full frontend verification**

Run all new waveform tests and existing affected timeline/piano tests, then:

```bash
npm run lint
npm run format:check
npm run build
```

- [ ] **Step 6: Full backend verification**

```bash
cargo test waveform_tile --quiet
cargo test waveform_peak_cache --quiet
cargo test --quiet
cargo check
```

- [ ] **Step 7: Browser/Tauri stress verification**

Verify WebGL2 and forced Canvas modes at desktop and compact widths. Exercise
horizontal scroll/zoom, vertical virtualization while scrolled, project
switching, source replacement, and context loss. Run the 30-minute script and
retain its final diagnostics in the task report.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "test(waveform): verify bounded flicker-free GPU rendering"
```

---

## Final Verification

- [ ] `git diff --check` is clean.
- [ ] No generated dependency, build, screenshot, cache, or log artifacts are tracked.
- [ ] WebGL2 is selected on capable browsers.
- [ ] Forced WebGL failure and context loss render through Canvas 2D.
- [ ] Timeline and parameter-editor frames match shared scene fixtures.
- [ ] No per-track waveform canvas, listener, or rAF remains.
- [ ] Frontend/backend cache ledgers stay within exact budgets.
- [ ] Old-generation responses increment stale counters without changing frames.
- [ ] The 80-track/5000-clip workload remains responsive.
- [ ] The 30-minute run shows no sustained post-warm-up memory growth.
