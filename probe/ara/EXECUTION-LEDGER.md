# 执行 ledger（交接快照）

> 这是切换 agent 时从 `.superpowers/sdd/2026-10-04-ara-bridge-probe.md/progress.md`
> 复制过来的**冻结快照**。原文件是 SDD 工作区里的活文件、且被 gitignore；
> 这份进版本控制，确保过程记录与其中的 Ruling 不随会话切换而丢失。
>
> **权威顺序**：spec > plan > 本文件。本文件记录的是"当时怎么判断的"，
> 而不是"现在应该怎么做"。若与 spec 冲突，以 spec 为准。

---
# SDD ledger — plan: docs/superpowers/plans/2026-10-04-ara-bridge-probe.md

Executor: inline (superpowers:executing-plans), agent session, Windows.
Worktree: `E:\code\HiFiShifter\.worktrees\ara-bridge-probe`
Branch: `feature/ara-bridge-probe`
Base commit at start: `8b93e2ce`

## Setup notes

- **Spec read**: `docs/superpowers/specs/2026-10-04-ara-bridge-design.md`. Authority for rulings.
- **Helper scripts unusable**: `sdd-workspace` / `task-brief` / `review-package` are bash
  and this host's `bash` resolves to Git Bash 5.1 invoked as `sh`, where `set -o pipefail`
  fails. Workspace, ledger and briefs are therefore created and maintained by hand with
  identical layout. Cost if wrong: none to the deliverable; the scripts are convenience only.
- **Workspace location ruling**: the helper puts the workspace at
  `<repo-root>/.superpowers/sdd/<plan>/`, which would leave `git status` dirty (the repo has
  no such entry, and a per-worktree `info/exclude` is not honoured — verified, it needs
  `extensions.worktreeConfig` and that setting is shared across worktrees, so it does not
  isolate). Ruling: ignore `.superpowers/` on **this branch only** instead of on `develop`
  or adding a repo-wide scratch entry. Cost if wrong: one `.gitignore` line to relocate.

## Environment prerequisites (plan section, verified)

| Item | State |
| --- | --- |
| MSVC env bootstrap | `tools/msvc-env.ps1` present. Verified: without it `cl.exe` fails `D8050`; with it a direct `cl.exe` compile exits 0. |
| `frontend/dist` | Built (23 files). `build.rs` will not re-run npm. |
| `frontend/node_modules` | Installed (`--ignore-scripts`; esbuild postinstall spawn is sandbox-blocked). |
| REAPER | 7.81 at `D:\Softwares\REAPER (x64)\reaper.exe`. `REAPER.ini` line 341 `ara=2` (ARA enabled). `vstpath64` includes `C:\Program Files\Common Files\VST3`. |
| Melodyne | 5, VST3 at `C:\Program Files\Common Files\VST3\Celemony\Melodyne\Melodyne.vst3` |
| MSVC / CMake | VS 2022 Community, VCTools 14.44.35207; CMake 4.4.0 |
| `cargo test` baseline | **NOT OBTAINED** — see blocker below |

### Known blocker: baseline tests cannot be run from this session

`cargo test` fails in the native C/C++ build steps. Symptoms rotate across runs
(`cl : D8050` / `MSB6003: Failed to create a temporary file` / `UnauthorizedAccessException`
on `%TEMP%\MSBuild*`), and the failing crate rotates with cargo's retries. Ruled out:
`TEMP`/`TMP` exist and are writable; `cl.exe` compiles standalone successfully; no stale
MSBuild temp files. Two `%TEMP%\esbuild-*` dirs left by the earlier confined runs carried
`DESKTOP-L3EU8NQ\CodexSandboxUsers` ACLs and were removed.

Assessment: this session's sandbox interferes with cargo's grandchild compiler processes.
Not a repo defect — the main tree's same test binary built at 02:08 today. Baseline must be
taken by a human in a normal terminal (instructions are in the plan's environment section).

**Ruling**: proceed with the plan's non-code / non-baseline steps while the baseline is
outstanding, because Tasks 1 and 2 produce artifacts (a captured ARA model, an SDK
build) whose correctness does not depend on the baseline. Task 3 does depend on it and
will not be started until it exists. Cost if wrong: Task 3 blocked, no work wasted.

## Pre-flight scan

Plan task interfaces:

| Row | Produces → Consumes | Finding |
| --- | --- | --- |
| 1 → 2 | `captures/ara-model.json` → "confirm the binding can express these fields" | OK. Task 2 only needs the field list. |
| 1 → 3 | `captures/ara-model.json` → fixture for the mapping test | OK, and Task 3 already states `AraDocument` is a probe-local type, so the mapping does not depend on Task 2's binding being stable. This is what makes the risk ordering safe. |
| 2 → 3 | binding path → (implicitly) the ability to run Task 3 at all | **Conflict, resolved by the plan itself**: Task 2's kill criterion already says do not start Task 3 without a working binding. No ruling needed. |

Pre-flight: no unresolved shared-interface conflicts.

## Progress

### Task 1: 取得宿主侧 ARA 模型的真实样本 — IN PROGRESS

Steps 2 (clone + build) **done**. Steps 1, 3, 4, 5 need REAPER GUI interaction or
edits to SDK example code; status below.

**Step 2 evidence (the SDK toolchain works):**

- `probe/ara/ARA_SDK` cloned with all submodules (`ARA_API`, `ARA_Library`, `ARA_Examples`
  and its 3 submodules).
- VST3 companion SDK installed by the SDK's own script → `probe/ara/ARA_SDK/vst3sdk`,
  pinned `v3.7.11_build_10`.
- CMake configured: `cmake -B build-vs2022 -G "Visual Studio 17 2022" -A x64 -D ARA_SETUP_DEBUGGING=OFF`.
- Built: `probe/ara/ARA_SDK/ARA_Examples/build-vs2022/bin/Release/ARATestPlugIn.vst3`,
  284672 bytes, **ARA SDK version 2.3.0**.

**Task 1: Ruling: build `-D ARA_SETUP_DEBUGGING=OFF`** — with it ON, the SDK's post-build step
copies the plug-in into `$ENV{CommonProgramW6432}\VST3\` (`C:\Program Files\Common Files\VST3`),
a write outside this worktree. Rule: keep the build self-contained and ask before installing
system-wide. Consequence: REAPER must be pointed at the build output, or the user runs the
install. Cost if wrong: one extra step for the user.

**Task 1: Ruling: build recipe** — the MSBuild build failed repeatedly with
`MSB6003 ... UnauthorizedAccessException ... MSBuildTemp\tmp*.rsp`. Investigated and ruled out:
`%TEMP%\MSBuildTemp` exists, is writable from this shell, and carries benign ACLs (including the
sandbox group, alongside `ARounder FullControl`); no `TEMP`/`TMP` misconfiguration.
The combination that **succeeded**:

1. MSVC env loaded from `vcvars64.bat` so `cl.exe` is on `PATH` (the same root cause as the
   repo's own `D8050`, see `tools/msvc-env.ps1`), **and**
2. `/m:1` (serial — parallel compilation appears to be what the sandbox interrupts), **and**
3. `/p:TrackFileAccess=false` (MSBuild then skips the temp response file entirely).

Cost if wrong: none — this is the recipe that produced the artifact.

### Task 1: Step 1 and Step 3 — DONE

**Step 1 (ARA is live in REAPER): PASSED.** Human check — Melodyne 5 on a track
with audio shows that audio's content. This is the precondition for everything
else; without it every inference about "what the host gives us" would rest on a
false premise.

**Step 3 (dump instrumentation): built and staged.** Hooks
`willNotifyModelUpdates()` and `didEndEditing()`, serializing the full model
graph. Verified by artifact size: plug-in rebuilt at **320000 bytes** (was
284672), so the dump is linked in. Staged to
`D:\VST\ARATestPlugIn.vst3\ARATestPlugIn.vst3` — `D:\VST` is already in
`REAPER.ini`'s `vstpath64`, so no config change is needed, only a VST rescan.

**Task 1: Ruling: instrumentation source files must live inside `ARA_Examples/`.**
The SDK's `ara_group_target_files()` assumes every source is under the project
directory. Placing them at `probe/ara/instrumentation/` made CMake configuration
fail with "is not a prefix of file", which does not name the real constraint.
Moved to `ARA_Examples/instrumentation/`; the authoritative, reviewable copy is
committed at `probe/ara/instrumentation-reference/`. Cost if wrong: the
instrumentation must be re-copied from the reference dir before rebuilding —
documented in the README.

**Task 1: Ruling: `/utf-8` is required on the instrumented target.**
The instrumentation is UTF-8 with Chinese comments; this machine's MSVC defaults
to code page 936, under which the compiler reported `C2447: '{': missing function
header` — a syntax error with no relation to its cause. The SDK's own sources are
pure ASCII, so the official build never exposes this. Cost if wrong: none; the
flag is additive and the build is verified.

**Task 1: Ruling: colour omitted, `std::filesystem` avoided.**
The target compiles as C++11, so `std::filesystem` is unavailable, and `ARAColor`
is an ARA 2.0 draft addendum not visible in this configuration. Colour is
irrelevant to the render mapping, so the field is dropped rather than worked
around. Cost if wrong: one field to add later if a host turns out to distinguish
regions by colour.

**Task 1: Ruling: tempo is NOT in the object model.**
`ARA::PlugIn::MusicalContext` exposes only name/orderIndex/colour — there is no
tempo or bar-signature member. Tempo (`kARAContentTypeTempoEntries`) and bars
(`kARAContentTypeBarSignatures`) arrive through a *content reader*. The dump
therefore records musical contexts and their region sequences, but not tempo
points; adding those needs a content-reader call. Cost if wrong: Task 3 loses
tempo-map comparison, which the plan does not require for the region mapping.

### Task 1 remaining: Steps 4, 5 — need human

Step 4 (awkward fixture: same source placed multiple times, stretched, reversed,
faded, plus a non-44.1 kHz project) and Step 5 (`FINDINGS.md`, above all the list
of fields ARA does not provide but rendering needs) require REAPER GUI work.

### Task 1 Step 3 — DONE. Real REAPER ARA capture obtained.

`probe/ara/captures/ara-model.reaper.json` (1447 bytes) is a real REAPER-produced ARA
model, captured in a fully isolated instance (`-cfgfile` + own vstpath64 + own plugin
copy). Findings that de-risk the spec:

| ARA field | REAPER's actual value | Consequence for the mapping |
| --- | --- | --- |
| `audioSource.persistentID` | the **absolute file path** (`E:\...\tone44100.wav`) | This is the direct counterpart of `Clip.source_path` — the single most uncertain row of the spec's §5.2 table now has evidence. |
| `audioSource.sampleAccessEnabled` | `true` | The plug-in can read source PCM. |
| `merits64BitSamples` | `true` | REAPER prefers 64-bit sample access. |
| `regionSequence.name` | `probe-44k` (the REAPER track name) | regionSequence ↔ track. |
| `audioModification` | exists; persistentID equals the source's | modification ↔ Take is a real relationship, not an assumption. |
| `startInPlaybackTime` / `durationInPlaybackTime` | `0` / `2` | Seconds as `double` — no sample↔second conversion needed for placement. |
| `sampleRate` / `sampleCount` | `44100` / `88200` | Matches the 2 s fixture. |
| `documentName` | `""` | REAPER sends no document name. |
| `audioModification.name` | `null` | Unnamed when the host did not set one. |

**Task 1: Ruling: live REAPER capture runs in an isolated instance only.**
`-cfgfile` does NOT fork a second instance when REAPER is already running — REAPER is
single-instance, so the script lands in the running one. That mistake wrote two probe
tracks into `D:\音MAD\...\test\test.rpp` (confirmed by the user to be a scratch project,
no loss). Rule now: kill all REAPER processes, launch with `-cfgfile` + an isolated
`vstpath64`, and never send scripts to a running instance.
Cost if wrong: contaminating a project the user cares about.

**Task 1: Ruling: the plug-in's dump path must not rely on the environment.**
`ARA_PROBE_OUT` is inherited by REAPER when launched from a shell but was `nil` in the
earlier job-based launches, and the plug-in process cannot be assumed to see it either.
Added a fallback chain ending in `.\captures\ara-model.auto.json` relative to the
launch working directory, plus an absolute path into this worktree.
Cost if wrong: a silent no-op instrumentation — which is exactly what cost several
rounds here.

### Task 1 Step 4 — PARTIAL: one clean region captured; the awkward fixture is next

Captured: a single unstretched region on one track. **Not yet captured:** the same source
placed multiple times, a stretched region, a reversed region, fades, and a non-44.1 kHz
project. Those are the cases that expose the transformation flags.

**Blocker found for the stretch case specifically:** the SDK reports at load time that
this test plug-in "does not support time-stretching" and "does not support content-based
fades", so REAPER may never set `kARAPlaybackTransformationTimestretch` for it. If that
holds, the stretch flag can only be observed with a plug-in that advertises support —
i.e. Melodyne, whose model this instrumentation cannot dump. Resolution options are
recorded in the next ledger entry once tried.

### Task 1 Step 4 — DONE. The awkward fixture produced four hard results.

`probe/ara/captures/ara-model.awkward.json` (4272 bytes), from an isolated instance:
4 regions of one source at 0/3/6/9 s on one track, one of them stretched to
`D_PLAYRATE = 2.0`, one with 0.5 s fades, one reverse attempt, plus a 48 kHz source
on a second track.

**1. Duplicate placements collapse onto one source — verified.**
One `audioSource` + one `audioModification` + **four** `playbackRegion`s, all sharing
the same absolute-path persistentID. This is exactly the shape `Clip.source_path`
needs: many clips, one source.

**2. Time-stretch is expressed as a duration discrepancy, NOT a flag.**
The stretched region reports `durationInModificationTime: 1` against a 2 s source,
`durationInPlaybackTime: 1`. So `playback_rate` must be derived as
`durationInModificationTime / durationInPlaybackTime`. This is lossless and it is
precisely the spec §5.2 row "playback transformation ↔ playback_rate".

**This also retires the blocker I recorded two entries ago.** The SDK's
"plug-in does not support time-stretching" notice does not obstruct the mapping,
because the stretch is not carried by
`kARAPlaybackTransformationTimestretch` at all. My earlier worry was wrong.

**3. Sample rate is per source and reported exactly.**
`tone44100.wav` → 44100 / 88200 samples; `tone48000.wav` → 48000 / 96000 samples.
Sources keep their own rate independent of the project, so the spec's "model domain
is fixed at 44.1 kHz" concern does not cause misalignment on the ARA path.

**4. Fades are a real gap.**
The faded region reports `hasContentBasedFadeAtHead/Tail: false`, and
`isTimestretchEnabled` is likewise false — because this plug-in declares at load
time that it supports neither time-stretching nor content-based fades, so REAPER
never offers them. Consequence: fade **shape and curvature**
(`fade_in_shape` / `fade_in_dir`) are unobtainable via ARA for this plug-in and
must be owned by HiFiShifter's own model. This is the first concrete entry for
Step 5's missing-fields list.

**Task 1: Ruling: the `sampleAccessEnabled: false` observation is left OPEN.**
Both sources in the awkward run report `sampleAccessEnabled: false`, whereas the
clean run reported `true`. The plug-in cannot analyse pitch without sample access,
so this may matter. Most likely the dump fired before REAPER granted access; it
could also be real. Not explained, not papered over — recorded as an open question
to settle before Task 3 relies on reading source PCM.

**Still FAILED**, at `fdk-aac-sys` (`cmake` crate → MSBuild `CL.exe` task,
same `MSBuildTemp` temp-file denial). So the recipe is necessary but not sufficient here.

**Ruling**: stop trying to obtain the baseline from this session. The ARA SDK build succeeded
because that project's own CMakeLists controls `TrackFileAccess`; the `cmake` *crate* (used by
`fdk-aac-sys` / `opusic-sys`) builds its project files itself and gives no such hook, so three
of HiFiShifter's native deps remain unreachable from inside this sandbox. Baseline must come
from a normal terminal. Cost if wrong: Task 3 stays blocked; nothing else is affected.

---

## Task 2 Step 3 — DONE. The Rust cdylib loads in REAPER as an ARA plug-in.

Evidence: `probe/ara/rust-path/FINDINGS.md` §3, `probe/ara/captures/task2-{capture,plugin}.log`.
REAPER inserted `VST3: HiFiShifter ARA Probe (HiFiShifter)`, queried
`IPlugInEntryPoint`/`IPlugInEntryPoint2`, created the ARA document controller
(`apiGeneration=V2Final`), bound it, and the plug-in logged
`sources=1 modifications=1 regionSequences=1 playbackRegions=2` for one source placed twice.

**Task 2: Ruling: path A (`ara2-bridge` + a self-written VST3 shell) is the selected path.**
`ara2-bridge` 0.3.0 (with companion) supplies the ARA↔VST3 COM adapters
(`ARA::IMainFactory`, `ARA::IPlugInEntryPoint2`) and the document-controller runtime; the only
missing piece was the VST3 module shell (factory + component + processor + minimal controller),
which is now written in Rust. Path B (hand-writing the whole ARA binding) is unnecessary.
Cost if wrong: the shell is ~900 lines of one-off probe code that a real implementation would
replace with a maintained VST3 binding once one exists.

**Task 2: Ruling: VST3 IID bytes must use the GUID layout on Windows.** `INLINE_UID(l1,l2,l3,l4)`
expands to `l1` little-endian, `l2` split into two little-endian u16s, `l3`/`l4` big-endian.
Established empirically: REAPER's `IPluginFactory2` request emitted
`50B607004BF20B4CA464EDB9F00B2ABB`, exactly the GUID layout of
`{0007B650-F24B-4C0B-A464-EDB9F00B2ABB}`. My first implementation used little-endian per word;
the symptom was not an error but "0 classes" (the cache kept only a bare timestamp), because the
host's `countClasses` landed on the wrong vtable slot. Cost if wrong: silent misdetection that
looks like an incompatibility rather than a bug.

**Task 2: Ruling: `IPluginFactory` derives from `FUnknown`, not `IPluginBase`.** Inserting
`initialize`/`terminate` slots between `release` and `getFactoryInfo` shifts the whole factory
vtable by two, so `countClasses` reads `getFactoryInfo`'s return value. Cost if wrong: "0 classes".

**Task 2: Ruling: REAPER requires an edit controller for an ARA plug-in.** With only
`kVstAudioEffectClass` + `kARAMainFactoryClass` registered, REAPER completed initialize and the
ARA bind, then asked for `IEditController`, called `getControllerClassId`, gave up, unloaded the
module, and then called an ARA callback through the stale controller pointer — a real crash
(`0xc0000005`, module `HiFiShifterARAProbe.vst3_unloaded`, offset resolving to
`ara2_bridge_plugin::ffi::generated_callbacks::begin_editing`). Adding a minimal
`kVstComponentControllerClass` (no parameters, no GUI, `createView` returns null) made the insert
succeed and the crash disappear. Cost if wrong: the probe reports "REAPER refuses ARA without a
controller", which would itself be a finding, but a real implementation must ship a controller
anyway.

**Task 2: Ruling: one companion binding per processor instance.** REAPER creates **three**
`IAudioProcessor` instances for one track (roles `0x6` = editor renderer + editor view, and two
`0x1` = playback renderer). The companion's binding is one-shot, so each instance needs its own
`CompanionProcessorBinding` + `Vst3PluginEntryAdapter`; sharing one would make the second bind
fail. Cost if wrong: only the first role would bind and ARA rendering would break.

**Task 2: Ruling: `sampleAccessEnabled` is granted by the host, not withheld.**
`enableAudioSourceSamplesAccess(source, enable=true)` is called after the ARA bind (log line
`[0009]`), and `enable=false` later on deactivation (`[0014]`). So the `false` in Task 1's captures
is a revoke, not a refusal; it is not evidence that "host-supplied source + local synthesis" is
broken. The still-open part is only whether access is stable across repeated calls during
playback. Cost if wrong: none to the path decision; it downgrades an open question to a narrower one.

**Task 2: Ruling: the `cargo test` baseline is still outstanding, so Task 3 stays closed.**
Nothing in Task 2 changed any file under `backend/` or `frontend/`, so the missing baseline does
not affect Task 2's conclusions. Task 3 still needs the human-run baseline from a normal terminal.
Cost if wrong: Task 3 remains blocked; no work is wasted.

---

## Baseline — OBTAINED. Blocker 1 is cleared.

The sandbox that used to block the native build steps is gone, so the run completed in this
session with the documented recipe (MSVC env, clean `TEMP`/`TMP` set *after* vcvars, `--jobs 1`).

`cargo test --no-fail-fast` in `backend/src-tauri`:

| Target | Result |
| --- | --- |
| `backend_lib` unittests | 777 run → **772 passed, 4 failed, 1 ignored** |
| `main.rs` unittests | 0 tests |
| `tests/loop_semantics.rs` | 10 passed |
| `tests/track_duplicate.rs` | 5 passed |
| `tests/smoke.rs` | 1 passed |
| `tests/reaper_export_rates.rs` | 1 passed |
| doc-tests | 0 tests |
| **total** | **789 passed, 4 failed, 1 ignored** |

**Task 3: Ruling: the 4 pre-existing failures are environmental and must not be fixed here.**
All four are `audio_engine::snapshot::tests::{build_snapshot_attaches_volume_curve_to_rendered_and_raw_clips,
build_snapshot_pads_from_previous_render_for_mid_playback_miss,
build_snapshot_releases_pad_suppression_when_current_render_hits,
build_snapshot_suppresses_pad_at_transport_arm}`. They share one cause: the tests write to the
hardcoded POSIX path `/tmp/hifishifter-*.aiff`, which on Windows resolves to `E:\tmp\…`
(`E:\tmp` does not exist), so `std::fs::write` returns `Os { code: 3, kind: NotFound }`.
This is a Windows path assumption in the tests, not a product defect and not caused by the probe.
Recorded as the baseline so any later change can be distinguished from "already broken".

**Task 3: Ruling: Task 3's prerequisite is now satisfied.** The baseline is no longer empty, so
edits to `backend/` can be evaluated against it. The probe is still acting on the plan's ordering:
Task 3 is only started on the user's go-ahead.

---

## Task 3 — DONE (field level). Evidence: `probe/ara/captures/roundtrip/FINDINGS.md`.

11 tests in `probe/ara/mapping/` pass. The converter lands on the **real** `Clip` /
`TimelineState`, not on a probe-local mirror.

**Task 3: Ruling: consume the real types through the existing `__test_internals` hook.**
`backend/src-tauri/src/lib.rs` keeps almost every module private (`mod state;`, `mod mixdown;`),
so an external crate cannot see `TimelineState`. But there is already a
`#[doc(hidden)] pub mod __test_internals` that re-exports `state::{Clip, TimelineState}`.
Using it means the probe adds **no** `pub` to `backend/`, so the "don't touch backend/" rule holds
and the mapping still targets the genuine product structs. Cost if wrong: none; the hook is
`#[doc(hidden)]` and pre-existing.

**Task 3: Ruling: the probe crate must seed its `Cargo.lock` from the backend's.** A fresh
resolution pulls different `windows-core` versions and `backend_lib` then fails to compile as a
dependency (`cast()` not found in `webview2_accelerators.rs`, trait from a different
`windows-core`). This is a feature-unification difference between "built inside its own package"
and "built as a dependency" — not a backend defect. Copy `backend/src-tauri/Cargo.lock` first.
Cost if wrong: the probe cannot build against the product kernel at all.

**Task 3: Ruling: Task 1's "time-stretch = duration difference, proven by the awkward fixture"
does not hold — the fixture carries no stretch at all.** In the awkward capture every one of the
five regions has `durationInModificationTime == durationInPlaybackTime` (including the region that
was supposed to be stretched: 1 s of modification over 1 s of playback). Task 1 read the "1 s
region against a 2 s source" as a stretch, but that is just a **trimmed region**; equal durations
mean **no** stretch. The likely cause is that the ARATestPlugIn declared no time-stretch support,
so REAPER never wrote a stretch into the ARA model. The formula itself still holds and is now
verified on a synthetic document (durMod 2 / durPlay 1 → `playback_rate == 2.0`).
Cost if wrong: a mapper written against that fixture would "pass" a stretch test that never
exercised stretch. Follow-up: re-capture with the Task 2 plug-in advertising
`Timestretch | ReflectTempo | ContentFades`; that is the only mapping branch still lacking a
host-level observation.

**Task 3: Ruling: ARA has no reverse bit.** Checked `ARAInterface.h`: the playback-transformation
flags are only Timestretch, TimestretchReflectingTempo, ContentBasedFadeAtHead/Tail. So reversal
cannot come from the region model; either the host reverses the source it feeds (fine) or the
plugin cannot know (not fine). Recorded as an open, non-degradable-in-principle item rather than
solved.

**Task 3: Ruling: plan Step 6 (sample-level audio comparison) cannot be done as written.**
`render_mixdown_interleaved` and `MixdownOptions` are not exposed through `__test_internals`, and
adding `pub` to `backend/` is outside the probe's rules. Minimal fix, requiring approval because
it edits `backend/`: add
`pub use crate::audio::mixdown::{render_mixdown_interleaved, MixdownOptions};`
to the existing `__test_internals` block. Until then the Task 3 conclusion covers **field-level
losslessness only, not render-output equality** — R4 (renderer produces audio inside the ARA
window) remains unverified. Cost if wrong: R2's audio half is asserted rather than measured.

**Task 3: Ruling: the lost-field list contains nothing that blocks v1.** Fades, loop, item gain
and reverse already belong to HiFiShifter's own model per the spec; the source-content fingerprint
is replaced by ARA's content-change notification. The one item to keep watching is reverse
(see above). Cost if wrong: a reversed region would render forward; it is flagged, not hidden.

---

# 开发阶段（分支 `codex/ara-plugin`，2026-10-04）

> 这一段是**实现设计**产生的 Ruling，不属于探针计划。方案见
> `docs/superpowers/specs/2026-10-04-ara-plugin-v1-design.md`，
> 计划见 `docs/superpowers/plans/2026-10-04-ara-plugin-v1-phase1-2.md`。

**Dev 1: Ruling: 依赖闭包 ≠ 内核边界，而且必须**先拆 `state.rs` 再算闭包**。**
实测：从 `mixdown` / `{mixdown, state}` 出发的闭包都是 **43 模块 / 2.57 MB**，其中
`project` / `notebook_assets` / `hfspeaks_v2` / `temp_manager` / `media` / `recording`
都不是内核 —— 它们被卷进来，是因为 `state.rs`（11826 行）把纯模型与运行时容器 `AppState`
放在同一个模块里，而 `AppState` 引用这些。决定：先把 `state.rs` 拆成
`model` / `app`（纯搬运，路径零改写），**拆完之后重算的闭包才是施工清单**。
代价：若照拆分前的 43 模块施工，会白搬一批非内核模块并让插件二进制白白变大。

**Dev 2: Ruling: 把 workspace 根提到 `backend/`，三个 crate 共用一份 `Cargo.lock`。**
现状是三个各自带 `[workspace]` 的 crate，插件 crate 靠"记得复制 app 的 lock"避免
`windows-core` 版本漂移（不复制就编译失败，已实测）。决定：合并成一个 workspace，
让这个问题在结构上消失。代价：`target/` 从 `backend/src-tauri/target` 变为
`backend/target`，本地要重建一次依赖；回退是 `git mv` 回来，可逆。

**Dev 3: Ruling: 插件 crate 只依赖 `hifishifter-kernel`，并用测试钉住。**
"不把 WebView2 / Tauri 带进 DAW 进程"必须用 `cargo metadata` 的依赖树断言来守，
而不是靠代码评审记得。代价：若靠人评审，某次图省事的 `use backend_lib::…` 会到
DAW 里才显形。

**Dev 4: Ruling: 插件侧用 `ara2-bridge-plugin` 的 `PluginModel` 高层 trait，不手写回调委托。**
探针用的是低层适配器 + 手工收集（`model.rs`，236 行）。实测发现同仓库的 `ara2-bridge`
默认 feature `plugin` 已提供 `DocumentLifecycle` / `MusicalContexts` / `RegionSequences` /
`AudioSources` / `AudioModifications` / `PlaybackRegions` 这些**语义化 trait**，
以及 `PluginBuilder` / `RealtimeHeadTailAdapter` / `Persistence`。决定：产品实现这些 trait，
只复用探针的 VST3 模块外壳。代价：若高层框架在 REAPER 下不可用，回退到探针已验证的
低层路径 —— 那是一条**已知可行**的路，所以这是低风险决定。

**Dev 5: Ruling: 参数编辑通道（产品设计 §5.4）收敛为"插件持权威 + VST3 state 持久化 +
本体作客户端 + 单写者乐观并发"。**
权威归插件的理由不是偏好：v1 的成功判据要求"工程重开后不需要重新合成"，而工程文件由宿主
保存，所以曲线必须落在与工程同行的地方 —— 选 VST3 组件 state（不是 ARA 文档归档，
后者在语义上属于模型对象图）。并发用 `revision` 乐观并发而不是"最后写者胜"，
因为后者会**静默丢编辑**。代价：乐观并发在冲突时让用户重做一次，体验略差，但不会损坏数据。

**Dev 6: Ruling: 源内容版本折进**既有的** `Clip.source_file_fingerprint`，`render_key.rs` 不改。**
产品设计 §4.2 要求"源内容变更必须使渲染缓存键失效"。现有渲染键已含该字段且已有测试
逐项钉住它必须影响哈希，所以只需在 ARA 映射时把"宿主内容版本 + 几何字段"折成该指纹，
改动面从"给管线加一个新类型"缩小为"映射时多算一个指纹"。残余窗口（宿主在插件未运行期间
改内容且版本号不递增、几何不变）**未验证**，列入 Phase 3 实验。代价：若窗口存在，
会造成过期缓存被复用 —— 属于静默出错，所以宁可多失效。

**Dev 7: Ruling: `EngineCommand` 进内核并删掉 `SetAppHandle`；`audio_engine` 整体留在 app 层。**
`SetAppHandle { handle: tauri::AppHandle }` 是该枚举无法离开 app 层的唯一原因，
而 worker 需要的句柄已由 `app_events::app_handle()` 提供（进程级出口，注册早于引擎启动）。
边界必须写死：`audio_engine` 是 cpal 设备边界，**不搬**，所以 `engine.rs` / `snapshot.rs`
里那 10 处 `tauri::` 与约 35 处 `emit` 不动 —— 否则搬迁会失控。代价：句柄改道有时序风险，
用"手工启动 app 并播放一次"兜底。

**Dev 8: Ruling: 实现计划只覆盖 Phase 1 + Phase 2。**
理由是"每个计划必须独立产出可工作软件"：Phase 1 的产出是"插件 crate 只依赖内核"，
Phase 2 的产出是"插件在 REAPER 里加载并呈现真实时间线"。Phase 3（渲染闭环，含 R4）与
Phase 4（参数通道）在 Phase 2 验收后另写计划。代价：交付被拉长成多轮，
但每一轮都有可验收的实物，避免了"一个大计划写到一半发现前提不成立"。

**Dev 9: Ruling: workspace 合并时，非根包的 `[profile.*]` 会被静默忽略，必须搬到根。**
合并 `backend/` 成一个 workspace 后，cargo 对 `src-tauri/Cargo.toml` 里的
`[profile.release]` / `[profile.dist]` / `[profile.dev-opt]` 只发一条 warning
（`profiles for the non root package will be ignored`）就忽略 —— 也就是 release 构建会
丢掉 `strip` / `opt-level = 3` / `dist` 的 fat LTO，而症状只是"产物变大变慢"，不报错。
决定：把三段 profile 整体搬到 `backend/Cargo.toml`（同时作用于 kernel 与 plugin，是期望行为）。
代价：若不搬，发布产物的优化与体积悄悄退化，且不会有人发现。

**Dev 10: Ruling: `state/model` 不是叶模块，`time_stretch` 拖着原生构建 —— 原计划的
搬迁顺序是反的。**
实测（拆分 `state.rs` 之后重算闭包）：内核目标集是 **40 模块 / 2.15 MB**，
只剩 `pitch_clip` / `pitch_analysis` 碰 `tauri::`（拆分前 43 模块 / 2.57 MB、5 个碰）。
两处推翻计划假设：① `state/model` 引用 `project` / `models` / `midi_import` /
`audio_utils` / `time_stretch`，必须先搬依赖 —— 机械搬迁要排在它**之前**；
② `time_stretch` 的两个后端（`sstretch` 静态链接、`soundtouch` DLL）由 **app 的 `build.rs`**
编译，`git mv` 会在 app 里假性通过、在插件里链接失败。
决定：Task 3 暂停，等"原生依赖构建归属"（设计文档 §4.9）定了再重写顺序。
代价：若照原顺序硬搬，会得到"app 能跑、插件链接失败"的假成功。

**Dev 11: Ruling: 设计文档 §4.2 关于闭包缩小的预测被部分证伪，按实测改写。**
原预测 `hfspeaks_v2` / `notebook_assets` / `temp_manager` / `recording` 会因拆分
`state` 而离开闭包。实测：**只有 `recording` 离开了**，另外三个仍在内核闭包里
（模型侧 `project` / `models` 也引用它们）。当时该说法已标注为推断并写明"以重算为准"，
所以没有误施工。决定：结论按实测改写，并把实测清单固化成
`probe/ara/kernel-closure-measured.md` 作为施工的唯一权威来源。
代价：若无这份实测，施工会按 43 模块的旧清单走，多搬或少搬都无从察觉。

**Dev 12: Ruling: 闭包测量脚本有 bug，闭包被低估；修正后发现唯一的"生产代码越界"
正是 HostServices 要修的那条边。**
脚本的正则 `^\s*(?:pub )?mod\s+` 漏掉了 `pub(crate) mod`，于是 `commands` /
`channel_policy` / `channel_mode` / `channel_decision` / `stereo_detect` 没进候选表，
经它们扩散的依赖全部丢失。修正后：含测试 46 模块、仅生产代码 44 模块。
多出来的 6 个（`commands` / `recording` / `search` / `system_clipboard` /
`linux_clipboard` 及 `commands` 子模块）**全部由一条边拉进来**：
`pitch_analysis/schedule.rs` 直接调用 `crate::commands::playback::request_background_render`
与两个全局开关。
**这条边就是设计 §4.3 表里第 2 类"向音频引擎投递命令"，也就是 `HostServices` 的职责。**
决定：**执行顺序改为先做 `EngineCommand` + `HostServices`，再做模块大搬迁** ——
反过来做会一路撞同一面墙。另记一处较小的越界：`project.rs` 的 `#[cfg(test)]` 里有
`crate::commands::channel_scan` 的集成测试，搬迁那一步再决定挪回 app 还是把
`channel_scan` 的纯函数拉进内核。
代价：若不先改这条边就硬搬，`pitch_analysis` 会带着整条 `commands` 链进内核，
而 `commands` 有 196 处 `tauri::` —— 内核"不认识 Tauri"这条不变量当场破产。

**Dev 13: Ruling: 内核的 39 模块生产代码闭包已经**完全不含 Tauri**。**
做法是两条出口：
① `HostServices`（后台渲染开关/消费标志/请求渲染）改掉了
`pitch_analysis -> commands::playback` 那条边；
② 新增内核的**进程级事件出口** `hifishifter_kernel::events::events()`，
把 `pitch_analysis/{dyn_analysis,schedule}.rs` 与 `pitch_clip.rs` 里 5 处
`state.app_handle` + `tauri::Emitter` 换掉（`pitch_clip` 的 `app_handle` 参数
随之从签名里消失，调用点改由出口自己判断"宿主是否在线"）。
实测（仅生产代码，从 `mixdown` 出发、排除 `audio_engine`）：**39 模块，碰 `tauri::` 的 0 个**。
代价：若继续把 `AppHandle` 当参数往下传，内核模块会永远拖着 `tauri::`，
"插件不把 WebView2 带进 DAW 进程"这条不变量就没有成立的一天。

**Dev 14: Ruling: 大搬迁只剩两处机械阻塞，且都不需要新的设计决定。**
① `formant_cache.rs` / `pitch_clip.rs` / `synth_clip_cache.rs` 里
`crate::audio_engine::byte_budget_cache::…` → 改成 `crate::byte_budget_cache::…`
（该模块早已在内核，app 根补一个再导出即可）；
② `pitch_clip.rs` 的 3 处 `crate::audio_engine::types::EngineCommand` →
把 `EngineCommand`（含 `StretchKey`/`AudioKey`）搬进内核；它依赖的 `TimelineState`
与 `MetronomeConfig` 本来就在那 39 个模块里，随大搬迁一起走。
另外确认 `renderer/chain.rs` 那条 `audio_engine::mix::sample_automation_curve`
引用**在 `#[cfg(test)]` 里**（该文件 724 行起），生产代码不受影响。
代价：无 —— 这两处都是纯机械改写，做错会当场编译失败。

**Dev 15: Ruling: 大搬迁已完成，内核含 41 个模块，插件 crate 只依赖内核。**
做法与结果：
- `state/model` + 其余 40 个模块整体搬进 `hifishifter-kernel`；app 侧用
  `pub(crate) use hifishifter_kernel::…` 接回 crate 根，**调用点一个都没改**；
- `EngineCommand` / `StretchKey` / `AudioKey` 搬进内核，`SetAppHandle` 删除
  （worker 改为惰性从 `app_events::app_handle()` 取句柄）；
- 内核的 `#[cfg(test)]` 对 app 的测试构建不可见，所以三处测试按"谁能同时看见两边"
  重新安置：`project.rs` 的扫描集成测试与渲染一致性测试搬进 app
  （`project_tests.rs` / `renderer_cross_checks.rs`），`streaming_pitch` 的分配计量
  测试留在内核（内核自建 `alloc_probe`，`#[global_allocator]` 是按 crate 生效的）；
- 原生构建全部跟模块走：WORLD、Signalsmith、SoundTouch、vslib 四段都搬进内核的
  `build.rs`。三条都是**实测**出来的链接失败（LNK1181 / LNK2019）逼出来的，
  不是预防性设计；
- ONNX 模型路径的"开发树兜底"补了一条回看 app `resources/` 的分支 ——
  否则内核自己的测试会因为"模型不在内核目录下"而失败。

实测：内核 **501 passed / 0 failed / 1 ignored**；app lib **279 passed / 4 failed**
（4 条仍是既有的 `/tmp` 路径环境性失败）；插件 **11 + 1 passed**（1 条是 A5 守卫）。
集成测试 17 条全通过。**`cargo tree -p hifishifter-plugin` 里没有 tauri / wry /
webview2-com / HiFiShifter，共 219 个包。**
代价：无 —— 整个过程每一步都有测试计数兜底；唯一"看起来会通过"的陷阱
（原生链接）在插件单独构建时就会暴露。

**Dev 16: Ruling: A5 守卫必须用 `cargo tree`，不能用 `cargo metadata`。**
第一版守卫读 `cargo metadata`，结果被自己的工具否定：metadata 列的是**整个
workspace 的成员**，app 本体必然在里面。改成 `cargo tree -p hifishifter-plugin
--edges normal --prefix none` 后按行首包名匹配。
代价：若沿用 metadata，守卫要么永远失败、要么被迫放宽到形同虚设。

---

# Phase 2（插件骨架产品化）

**Dev 17: Ruling: `ara2-bridge-plugin` 0.3.0 的高层 `PluginModel` **丢掉了整个模型图**，
不能用来建时间线。**
实测（对着锁定的 crate 源码逐项核对）：

| 映射需要的输入 | 高层 trait 给不给 | 证据 |
| --- | --- | --- |
| audioSource 的 id/采样率/样本数/声道数 | **给** | `AudioSourceProperties` 有 getter |
| audioModification 的 id | **给** | `AudioModificationProperties::persistent_id()` |
| regionSequence 的名字/序号 | **给** | `RegionSequenceProperties` |
| **playbackRegion 的起点/长度/名字** | **不给** | `PlaybackRegionProperties` 只有 `transformation_flags()` 一个公开读访问器（字段全私有） |
| **audioModification → audioSource 的边** | **不给** | `AudioModifications::create_audio_modification(context, properties)` 没有 source 参数 |
| **playbackRegion → audioModification 的边** | **不给** | `PlaybackRegions::create_playback_region(context, properties)` 没有 modification 参数 |
| **playbackRegion → regionSequence 的边** | **不给** | 同上（框架内部的 `runtime.create_playback_region(modification, sequence, properties)` 两个参数都**没有往委托层传**） |

即：这些边在框架**内部是有的**（`runtime.rs` 的节点上存着 `modification` / `sequence`），
只是**没有出现在要实现的 trait 上**。探针之所以没撞上，是因为它只用高层接口**计数**，
从不读边；而 Task 1 那份真实模型样本来自 C++ 插桩，不是这条链路。

**处置（第一步已做）**：把 `ara2-bridge-core` 本地化并补上五个读访问器
（`start/duration_in_{modification,playback}_time` 与 `name`），见
`backend/third-party/ara2-bridge-core/PATCHED.md`。补丁**纯增量**，
`[patch.crates-io]` 之后其余 ara2-bridge crate 行为不变。

**仍未解决**：三条边拿不到。当前的兜底是"文档里只有一个 source / modification 时
全部挂到它上面"（v1 单实例语义下通常正确），多源文档下是**近似**，已在
`ara/model.rs` 里显式告警并注明。

**下一步的两个选项**（推荐 a）：
- **(a) 同样把 `ara2-bridge-plugin` 本地化，给三个 trait 方法补上边参数** ——
  框架内部本来就有这些值，改动是"多传两个参数"，规模可控；
- (b) 自己手写一层 ARA 文档控制器委托（直接对 `ara2-bridge-sys`）——
  等于重写上游整个回调派发层，规模大得多。

代价：若不做，插件的模型只有"一堆没有连线的对象"，无法建时间线 ——
Phase 2 的 A1/A2 判据只能停留在"计数一致"。

**Dev 18: Ruling: 选了 (a) —— 把 `ara2-bridge-plugin` 一并本地化，给三个 trait 方法补上边参数。**
`AudioModifications::{create,clone}_audio_modification` 收 `source`，
`PlaybackRegions::create_playback_region` 收 `modification` + `sequence`；
`PluginModel` 上加等式约束把补出来的关联类型钉回"正主" trait 同名类型，
所以实现方（我们的 `ModelHandle`）只需给一套类型。
补丁纯增量，细节与撤销步骤见 `backend/third-party/ara2-bridge-plugin/PATCHED.md`。
实测：插件 crate 编译通过，11 条映射测试 + A5 守卫 + 导出符号守卫全绿；
内核 501 / app 279（4 条既有环境性失败）。**这一步让"插件能建出真实时间线"从
不可能变成可能** —— 之前拿到的模型只有计数，没有连线。
代价：仓库里多两个 vendored crate（~630 KB）。上游补上后按 PATCHED.md 撤销即可。

**Dev 19: Ruling: 产品 DLL 需要自己的可控日志出口，才能在 DAW 进程里核对 ARA 模型。**
实测：REAPER 不提供插件 log 后端；加入 HIFISHIFTER_ARA_LOG 环境变量驱动的
轻量 file logger 后，隔离实例原样记录了 ARA 绑定、对象数量与 clip 起点。
决定：日志器只在插件 DLL 内启用，宿主已有 logger 时不覆盖，未设置路径时静默。
代价：若删除，产品仍可能加载，但无法复核宿主回调与 A1/A2 现场证据。

**Task 10: Ruling: A1 与真实 UI 移动、切片均有端到端证据 — 补齐更新/销毁回调，按存活区域映射 — 若仍只累积创建事件，会留下旧位置和幽灵 clip。**
实测：隔离 REAPER 7.81 加载 HiFiShifter.vst3，同一素材两次摆放，日志原样出现
'ara: sources=1 modifications=1 regionSequences=1 playbackRegions=2 clips=2'；
TrackFX_AddByName 返回 0，ARA bind 走到 V2Final。用 ReaScript
SetMediaItemPosition 的早期调用没有新摘要，不足以断定宿主未通知。产品委托缺少
update_playback_region，上游默认空实现会丢弃属性更新；回归测试先失败（起点仍为 0，
期望 5），实现后通过。真实 UI 拖动日志为 clipStartsSec=[1.000000,4.000000]，
单选第二项再拖动为 [1.000000,5.000000]；切片后 playbackRegions=3 clips=3，
起点为 [1.000000,5.000000,5.500000]。截图与原始日志已归档。

**Task 11: Ruling: REAPER 拉伸通过时长差表达，实际反向 take 未在映射输入里带方向 — U2 通过；保留 U1 方向缺口并在 Phase 3 验真实输出 — 若默认补正向，会静默输出错误。**
实测：工厂声明 TIMESTRETCH | REFLECT_TEMPO | CONTENT_FADES；三条 region 的
原始日志中拉伸项为 durationMod=2.000000 durationPlay=1.000000 flags=0x1，
倒放项仍与普通项相同的两个时长坐标、同一 source persistentID、flags=0x1。
正式反向实验使用 action 41051，PCM_Source_GetSectionInfo 返回 reversed=true；
早期未在文档中定义的 B_REVERSED setter 不能作为证据，已替换。
还通过 ARA reader 读取共享源首 16 样本，与文件正向 PCM 最大差 1.40624999978023e-8。
归一化 JSON 由 verify_task11_capture.ps1 自动生成，保留原始日志。
U1 只说明当前映射没有方向；宿主是否在插件输出之外处理倒放仍需 Phase 3 输出实验。

**Task 10: Ruling: 浏览器 CUA 不暴露 Windows 窗口，不能据此宣布 Computer Use 不可用 — 使用 skill 指定的 node_repl + @oai/sky 实测 — 若混用接口，会把可执行验证误判成人工阻塞。**
验证时使用新隔离配置目录，并复制旧隔离配置的扫描缓存；REAPER 会自动添加系统 VST3
目录，vstpath64 单独一项不是完整隔离扫描的保证。新 profile 的首次系统扫描曾卡在
已有 Synthesizer V 插件激活窗口，已结束该扫描子进程；用户主配置与工程均未触碰。

**Task 10: Ruling: 文件 logger 使旧 setProcessing 日志触发音频线程 I/O — 移除该调用并增加实时回调守卫 — 若保留，会在播放启停时阻塞音频线程。**
独立审查对照锁定 SDK ivstaudioprocessor.h：setProcessing 可从 processing thread 调用。
回归测试先捕获两次日志进入，移除后为 0；保留非实时生命周期诊断。logger 安装失败时
也不再更改既有日志器的级别。

**Task 11: Ruling: 验证器不能只判时长不同后硬编码 PASS — 校验身份、位置、2/1比例、stretch位、反向项与正向项的几何一致性 — 若缺检查，变异证据也会被写成可信结论。**
六条回归实测通过：有效输入、错误时长、缺stretch位、反向几何变化、错误index、反向未生效。
最新插件构建与测试：4 lib + 13 mapping + 1 A5 + 1 exports，全部通过。既有内核/app
没有本批修改，未重复其全部测试，四条既有 Windows 路径失败仍不处理。

**Task 11: Ruling: 最后一次指针及采集超时调整后必须重新验证 — 已重跑插件构建、19 条测试与 6 条采集回归并关闭隔离 REAPER，Task 10/11 一批本地提交 — 若沿用旧测试结果，最终提交可能包含未经验证的尾部改动。**
本次 `git diff --check` 无输出。日志显式 force-stage，截图/JSON 与脚本逐路径暂存；
不包含 REAPER profile、DLL、SDK 检出或产品前端。不 push。

**Task 12: Ruling: 锁定 SDK 没有 storeAudioSourceContent，head/tail adapter 也不调度预渲染 — 更正上位设计并拆出 Phase 3a，先验 process 输出/归属/倒放，再接内核 — 若继续旧假设，会实现不存在的 PCM 回写接口并把缓存空洞误当宿主兜底。**
依据：`ARA_API/ARAInterface.h` 的 archive 接口和 `ivstaudioprocessor.h` 的 process 输出；
`backend/third-party/ara2-bridge-plugin/src/realtime.rs` 只有 head/tail 查询。
用户授权按建议持续分批执行，保持产品目标，纠正接口事实；不声称 A3/A4 已完成。

**Task 12: Ruling: 空 process 不初始化输出且接受非法 setup/布局 — 以真实 vtable 的六条预期失败为 RED，补安全 f32 stereo 缓冲边界与原生 SDK 布局 oracle — 若继续空实现，旧样本可能混入输出，手写 ABI 错位会直到宿主里才崩溃。**
原生 MSVC 实测 size/align：ProcessSetup 24/8，AudioBusBuffers 24/8，ProcessData 80/8，
ProcessContext 112/8，Chord 4/2，FrameRate 8/4；全部字段 offsetof 与 Rust 一致。
已确认 zero-frame flush 与 inactive null plane 合法。kSample64/非 stereo 明确拒绝。

**Task 12: Ruling: 审查发现只校验输出形状会在非法输入时先写输出 — 新增输入 sentinel 回归，统一先校验两边再写，并扩展 process 日志守卫 — 若只看不崩溃，会把不支持的布局伪报成功。**
新增回归先实测失败（negative input channel 返回 OK），修复后 13 lib 测试通过。
复审无 Task 12 checkpoint 阻塞。实时无分配/无锁由代码审查确认，动态守卫仅覆盖日志，
不能描述成做过动态 allocation 计数。当前输出仍为安全零，Task 14 才接 PCM。

**Task 13: Ruling: 每个 renderer 的分配用 model-ref 地址键，不能用 slot 或整张文档代替 — 增加 RegionOwners 与模型线程分配观察器，拒绝未知/跨文档键 — 若错误复用全时间线，会让多个处理器重复输出或串文档。**
三条所有权测试 RED/GREEN；两个真实扩展 FFI 通知测试 RED/GREEN；索引接线覆盖 region
销毁、document 销毁及旧 Model Drop 不撤销新文档复用地址的回归。
观察器在内部锁释放后通知，音频回调不访问表；editor sequence 仍未覆盖。

**Task 13: Ruling: 实测工厂返回的 Processor 关闭后 refcount 仍为 1 — 成功 queryInterface 后消耗工厂的初始引用，并以 native entry 的 builder Arc 保留扩展 owner — 若只移除 Box::leak 而不校验 COM 生命周期，会留下泄漏或提前释放。**
两条真实工厂测试先失败（remaining=1），修复后 remaining=0；宿主持有 entry COM 引用
时 owner 保留，最终 native release 后 owner 消亡。当前测试的 entry 尚未绑定文档。

**Task 13: Ruling: 手动销毁测试 lease 不等于产品文档销毁接线已完成 — 本地提交标为部分检查点，保留实际 controller lease 撤销/分配清空、bound teardown 与 editor sequence 三项未完成 — 若越过这道门接快照，可能在文档关闭后继续播放旧区域。**
最新实际构建及测试：20 lib + 13 mapping + 1 A5 + 3 renderer FFI + 1 exports = 38 passed。
采集验证器 6 passed；diff 检查通过，独立审查同样重跑 38 条无失败。
本批新 DLL 未部署到 REAPER，不把单测声称为宿主卸载或 PCM 发声验证。

**Task 13: Ruling: ARA 绑定必须关联真实 controllerRef，不能等第一次分配才猜文档 — 增加工厂身份通知、DocumentSession索引、控制器侧独立lease与原生shim上下文 — 若用最后文档/首region猜测，空文档和多文档关闭时会释放错误对象。**
真实工厂空文档、native bound teardown、companion先释放、editor sequence与重入查询
均有回归。两个 lease owner 独立，文档close撤销访问，entry最后COM引用保持interface storage。

**Task 13: Ruling: 本次首次实现误将VST3 bind参数当instance，测试夹具也传错，实际REAPER不绑定 — 对照锁定ARAVST3.h与C++shim纠正为不透明controllerRef并重采 — 若只相信同源夹具，会在全绿测试下交付无ARA行为。**
错误采集已移到.build-tmp保留；有效采集日志有bind document=...、3 region和4 renderer分配。
本批未改SDK/registry内容。扩展API补丁均在vendored plugin并更新PATCHED.md。

**Task 14: Ruling: 宿主PCM权威与实时无阻塞必须在输出侧落实 — scope内完整读取后准备44.1/48k快照，process只读原子指针，源更新/撤权先撤销发布 — 若按persistentID读文件或在callback推理，会复用错误源或阻塞DAW。**
源PCM与所有退役快照共享512MiB硬预算；单次快照另限64MiB，超限拒绝。只支持普通/裁切；
stretch/fades明确Unsupported。本批没有 pitch edit，A3未关闭。

**Task 15: Ruling: 真倒放最终输出仍为正放 — 保留失败WAV/原始日志与反向metadata，按计划停止完整v1，不进入Phase3b/4/5 — 若假定宿主会补偿方向，用户会静默听到错误音频。**
普通/裁切maxdiff=5.960464477539063e-8，间隙0；反向输出对反向oracle差0.5003815367817879，
对正向oracle差5.960464477539063e-8。官方41051+section reversed=true。只针对本REAPER链路，
不泛化所有ARA。验证器真实采集返回FAIL，6条变异回归通过不等于倒放功能通过。

**Task 14: Ruling: 集中审查发现setup接受96k和region换sequence成员未迁移 — 两个回归先失败再修复，补PATCHED契约 — 若协商假成功或旧成员残留，会导致静音或editor输出错轨。**
用户要求减少review，本批只做一次集中审查及针对发现的修复验证；没有逐任务重复审查。
额外覆盖scope授权/撤权、采样率oracle、30秒块拼接seek与真实process分配/释放计数。

**Task 15: Ruling: 最终源码验证通过不等于宿主倒放通过 — 构建成功、55条插件测试及两套各6条验证器回归通过，真实输出验证仍FAIL，提交有限检查点并停在方向阻塞 — 若把测试全绿当产品完成，会掩盖已知错误音频。**
最终重跑：35 lib + 13 mapping + 1 A5 + 5 renderer FFI + 1 exports = 55 passed，
0 failed；Phase3a验证器6 passed、Task11采集验证器6 passed，git diff --check通过。
真实输出验证返回exit 1：Reverse output was forward, not reversed；输出/源SHA256与
实测报告一致。关闭后复核无REAPER进程。未重复内核/app全套测试，不改变既有基线失败。
按明确路径暂存源码与证据，只有两份本批命名日志force-stage；只本地提交，不push。

**Task 16: Ruling: 用户明确授权先不做倒放并继续到GUI可用 — 正向范围覆盖Task15的方向停止条件，沿用独立app客户端设计 — 若把范围缩减写成方向修复，会误导用户得到错误音频。**
新spec/plan为2026-10-04-ara-gui-forward；保留原反向失败证据。既有30秒/双采样率与
资源上限作为本轮已知限制；只减少重复review，不跳过真实宿主GUI与音高输出验收。

**Task 17: Ruling: REAPER为一轨创建editor及逐片段playback处理器 — GUI只发现editor入口，文档共享曲线权威并分别预渲染所有renderer分配 — 若逐处理器独立编辑，用户只能改到一片或产生重复输出。**
宿主日志实际assigned=0x6/0x1；原声基线导出成功。Task16真实WORLD PCM注入与Task18
GUI客户端已本地提交。GUI默认开启宿主轨道WORLD分析，仅显式commit后改变DAW音频。

**Task 17: Ruling: SDK禁用访问要求同步销毁reader但不等于销毁已采集副本 — 保留有界编辑PCM副本，源content/geometry/deactivate/destroy时失效，实时播放发布仍撤销 — 若保留未经版本失效的数据，会提交旧源；若全部删除，REAPER停播后GUI无法编辑。**
依据锁定ARAInterface.h enableAudioSourceSamplesAccess注释。reader始终在scope返回前
释放。复制缓存不延长reader/HostContentScope生命周期，不通过文件路径补读源。

**Task 19: Ruling: 接完整内核后DLL新增vslib导入导致REAPER加载失败 — 插件关闭内核默认vslib，仅开onnx，独立app默认不变 — 若只在本机补闭源DLL，用户包仍加载失败且违反不分发vslib边界。**
dumpbin最终imports无vslib/Tauri/WebView；隔离目录附SoundTouchDLL/DirectML。

**Task 19: Ruling: 本机WebView2创建报0x800700AA且build版也无主窗口 — 隔离启动脚本设置WEBVIEW2_USER_DATA_FOLDER到worktree，窗口实测正常；不改产品跨平台启动逻辑 — 若误判为代码问题，会为单机profile故障改坏其他平台。**
用户已确认其他Windows/mac正常；保留原profile，不终止其他应用。GUI build必须启用
标准custom-protocol feature嵌入frontend/dist，实际exe为HiFiShifter.exe。

**Task 18: Ruling: 新命令必须登记既有invoke参数映射 — 授权新增services/invoke.ts的ARA映射与聚焦回归 — 若绕过门面，GUI命令实参会与本地约定脱节。**

**Task 19: Ruling: 大PCM单次写入真实命名管道超时，小消息通过不足以证明链路 — 16KiB分块读写并增加超过管道缓冲的真实回归 — 若错误，GUI仍在大快照下载时挂住。**
新鲜重跑plugin60+IPC4全部通过，IPC全套0.08秒；基础checkpoint4800705，不表示集中审查已通过。

**Task 19: Ruling: 用户手绘提交截图的Conflict由samples_access开关误推进model_revision触发 — 分离播放快照撤销与真实内容/几何版本，保留正确的过期提交拒绝 — 若误判，可能接受宿主已变更的旧编辑，须正反两向回归。**
真实日志Snapshot后只有两次enable=false，没有新的begin_editing/region更新；clear_renderers却无条件revision++。
用户后来自行操作成功Commit revision1/model8，尚无输出/持久化验收，不据此跳过修复。

**Task 19: Ruling: 集中审查四项影响编辑正确性与保存恢复 — 一次修复wave连同假冲突，参数投影/局部归属/持久关联/compose门禁分别TDD — 若省略，可能无提示丢曲线或提交成功却原声输出。**
持久身份选择真实宿主modification+source完整排序集合，只接受唯一精确关联，歧义/空关联显式拒绝；
不按名称或ara-track-N猜。代价是成员身份改变的恢复需重新关联，先导能力比静默套用更保守。

**Task 19: Ruling: 用户物理Escape停止Computer Use且GUI可能有未保存手绘曲线 — 本轮只读诊断与源码修复，不刷新/关闭/强杀现有GUI或REAPER — 若忽略，会覆盖用户当前编辑并违背停止请求。**

**Task 19: Ruling: 首轮修复复审发现false-checkpoint尾块不推进版本，仍能把未发送曲线标为已保存 — 比较实际Commit支持参数，并让四个成功参数写入在timeline锁内独立标dirty，checkpoint只控制undo — 若只靠版本或手工bump回归，会漏掉真实分块/平滑并丢用户曲线。**
源码0b8f4dd4；真实非checkpoint5条RED正常exit1，恢复后ARA18/params10正常exit0。
controller独立重跑同样通过。首轮F1/F3/F4/授权误冲突及最终F2/R1均已复审通过，无新问题。

**Task 19: Ruling: 真实参数命令测试摘要通过但进程不退出，误归因音频worker的尝试不足 — 明确不把摘要当通过，隔离仅测试的设备/推理外部边界，取得正常0/1退出 — 若扩大成产品生命周期修复，会改变其他机器已正常的行为且缺少依据。**
实际fixture缺省Nsf触发ensure_params_for_root→build_root_pitch_key→FCPE后台预热。
显式cfg(test) no-device AppState/None算法fixture保留真实参数命令与dirty逻辑；Default仍真实引擎。
没有global/TLS模式，产品分析/播放生命周期未改；真实DSP另外用WORLD oracle验证。

**Task 19: Ruling: 当前用户GUI仍占用旧产物，不能为了重建强制清理 — 新GUI在独立target构建，启动脚本选最新完整本地产物并拒绝重复实例 — 若误用旧exe，新投影保护不生效；若强杀，会丢正在编辑的曲线。**
controller最新验证plugin68/IPC4/kernel28/app18/params10/project21/frontend8通过；
基线普通/裁切/gap最大差5.96e-8。GUI音高导出、保存重开、同路径源改变仍无最终证据，未关闭Task19。
最后独立target嵌入dist GUI构建正常exit0，38.55秒，92661760 bytes；启动重复实例防护已实测拒绝。

Task 19: Ruling: 原GUI手绘后真实导出四窗220.5Hz变393.75Hz，无GUI重开PCM maxdiff=0，用户确认恢复曲线 — 关闭一期核心修音/持久化验收，保留同路径改源及边界未决 — 若过度外推，会把未测声道/采样率/seek能力写成发布保证。

Task 19: Ruling: 首次重开模块加载126来自项目CWD使flat开发DLL依赖不可见 — 启动脚本只设当前进程PATH并保留失败输出，规范bundle留发行任务 — 若只补本机系统PATH，部署缺陷会转嫁给用户且污染环境。

Task 20: Ruling: 用户二期目标是REAPER内的原GUI和自动应用，原createView仍null且原事件依赖Tauri — 新增原生IPlugView/WebView2和宿主通信适配、共享编辑会话，不启动独立app；独立app入口保留 — 若复用外部窗口/手动提交当成二期，会偏离实际目标。

Task 20: Ruling: 用户要求不要反复跑测试，等最后集中执行 — 保留回归用例但不逐函数跑红绿；完整功能后集中测试与REAPER验收，必要编译只用于排错 — 若错误，缺陷可能较晚暴露，因此不提前宣称未测功能完成。

Task 21: Ruling: 宿主同步窗口消息可能重入view，HWND销毁后也可能被复用 — 自有child析构COM、异步callback仅捕获weak，attach/resize不持view锁，window token识别原窗口 — 若省略，会死锁、复活已关FX或误关闭另一实例。

Task 21: Ruling: WebView2异步环境创建无cancel，weak失效不能阻止迟到COM回调进入DLL — 首次创建时只固定本模块到REAPER退出，窗口/浏览器/worker仍正常释放，明确不支持宿主运行中热更新 — 若只靠ExitDll返回false，宿主忽略返回值时会执行已卸载代码；代价是DLL映射保留到宿主退出。

Task 22: Ruling: 原参数命令依赖AppState但仅需timeline/undo/dirty/发布四个边界 — 完整原函数体与互转测试机械迁入kernel/editor，经ParamHost薄适配保留独立app副作用，插件不复制简化曲线语义 — 若hook改变锁序或dirty记账，会丢最后一笔/破坏独立模式，留共享回归最终验证。

Task 22: Ruling: SDK允许IConnectionPoint之间存在宿主代理 — 采用宿主IMessage及UTF16属性传本进程租约令牌，不传Rust裸Arc、不查全局最后实例；取消distributable声明 — 若只query私人接口，经过代理或不同组件顺序就会丢关联/串实例。

Task 22: Ruling: 原分析/波形API仍以文件域工作，不能按宿主persistentID读取 — 从已授权PCM生成私有分析WAV并反向关联，原HfsPeaks/描述符/mix波形/history实现共享；GUI ID按实例前缀隔离异步事件 — 若复用宿主文件或全局root IDs，分析可能越权/过期或串另一个FX。

Task 24: Ruling: 自动编辑不能让每笔推理阻塞UI，也不能等推理后才进入工程state — 32任务actor、150ms合并，先接受权威参数再发布快照，getState屏障处理已收到尾块；新submitted ticket撤销旧作业发布 — 若混为一个提交，保存/关FX可能漏最后一笔或慢作业覆盖新曲线。

Task 24: Ruling: UI等待getState时worker若阻塞发响应会形成等待环 — 命令响应使用受native 32 pending上限约束的非阻塞邮箱，事件另设128有界邮箱，COM只由UI timer发送 — 若共享满队列并阻塞发送，保存或销毁可能死锁整个REAPER。

Task 24: Ruling: 编译成功未证明Host消息实现/内嵌原GUI/长期自动快照预算 — 保持任务未完成，先适配前端与规范部署再集中真实验收，并审计retired回收及doc事务等待 — 若现在报告二期可用，会掩盖尚未跑过的核心宿主路径。

Task 23: Ruling: 原GUI独立窗口/录音/文件路径不能直接进入插件 — 显式host mode保留原Dock与参数编辑，禁用宿主拥有的入口，不初始化Tauri window API，自动应用替代连接/提交栏 — 若忽略，会调用不存在的app运行时或让GUI几何与DAW分叉。

Task 25: Ruling: eager SoundTouch/DirectML导入在InitDll之前就决定能否加载 — 加Win32薄入口，用邻接绝对Engine路径和DLL_LOAD_DIR，不依赖系统PATH/CWD — 若仅在Rust InitDll添加DLL目录，时机已太晚，用户重开仍126。

Task 25: Ruling: MSVC默认CP936读UTF8中文注释造成续行/函数头语法错误，utf8后暴露CRT terminate同名 — 显式utf8/Cpp17与engine_exit命名，只重编薄入口 — 若归因架构不可行，会错误放弃可加载的原生路径。

Task 24: Ruling: 自动应用会持续生成快照，保留到owner销毁会快速耗尽预算 — 新增实时原子读区、非实时无读者回收及分配前收集，失败保留旧音频；回归最终集中执行 — 若原子顺序证明错，会use-after-free，所以未测不能宣称已安全验收。

Task 26: Ruling: 实际REAPER加载/消息绑定result0且FX内原GUI层级成立，无外部HiFiShifter进程 — 关闭内嵌显示/关联可行性疑问，继续自动pitch/持久化/资源验收，保留baseline和原日志 — 若把显示通过等同完整修音，会掩盖编辑输出仍未验证。

Task 23: Ruling: 真实UI调用暴露clipboard/transliterate缺口 — 抽原native clipboard为无Tauri共享crate、原纯搜索整体迁kernel并保留app路径，真实读写参数格式与CJK转写 — 若用空结果掩盖缺命令，原编辑器复制粘贴/索引会悄悄失效。

Task 26: Ruling: Unknown是工程前向兼容枚举，不是合法GUI算法 — 仅插件命令准入拒绝Unknown/vslib并保持patch先验证再修改，独立app反序列化不变 — 若只信Deserialize成功，错误字符串可静默改变轨道及undo状态。

Task 26: Ruling: hostEvents多一层动态导入改变独立窗口初始事件时序，新增面板违反排版门 — 独立app直达原Tauri事件模块，插件才适配；采用原字号角色，布线门同时解析实际native actor — 若放宽门/改测试等更久，会掩盖standalone回归。

Task 26: Ruling: 75插件/58共享kernel回归与定向61前端通过，不等于宿主手绘/新全量通过 — 留真实手绘/导出/重开open，保存当前一次性RPP，不猜Sky句柄或用脚本替代GUI — 若混淆证据，会虚报用户最关心的自动使用链路。

Task 26: Ruling: 插件首轮缺少模型登记和原分析调度，参数曲线虽接受但自动WORLD不能收敛 — bundle按模块相邻Resources登记原模型，插件会话持有可取消/join分析worker并消费ClipPitchReady；若沿用游离全局worker，关闭FX会泄漏或串实例。真实actor回归输出329.104Hz（目标329.63Hz），缺原线时保持pending；代价是关闭FX会等待当前分析安全退出。

Task 26: Ruling: Computer Use对REAPER父窗口的坐标drag不能命中WebView子窗口 — 保留失败返回与截图证据，不猜HWND、不用PowerShell UIA或Lua代替用户手绘；若强行绕过会把脚本提交误报成真实GUI验收。

Task 26: Ruling: WebView accessibility树可读但控件click仍不能从REAPER父窗口执行，secondary action也没有Invoke；同进程输入代理实验进一步造成插件IPC超时 — 撤回代理，不牺牲已验证通信链路，保留该工具边界证据；若继续叠加代理会把真实插件故障误判成输入问题。

Task 27: Ruling: 用户真实试用确认基本可编辑，并报告停播发声、游标不一致、多轨道不可用；日志实际出现共享身份歧义 — 不再将问题归为Computer Use，先修三个高优项，播放控制和自动载入留后续批次 — 若只相信先前单实例绿测，会漏掉宿主真实生命周期错误。

Task 27: Ruling: realtime process未检查kPlaying，宿主停播仍会重复请求相同项目帧；base_sec与position_sec重复传同一绝对时间 — 停播实时回调只保留清零输出，offline模式继续供音；绝对时间只放position_sec，host_authoritative允许前端跟随停播seek — 若只用processing开关或忽略offline，会仍循环发声或令导出静音。

Task 27: Ruling: REAPER复制轨道合法共享modification/source，旧setState把单组件恢复直接写进整张文档且每个getState保存所有轨道 — 组件先暂存恢复，按真实assigned regions限定候选集合后合入共享权威，每组件只保存自己轨道；live共享身份合法，未限定的restore歧义仍拒绝 — 若按名称/临时序号猜归属，会串轨或抹掉另一轨曲线。

Task 27: Ruling: 各轨共同使用文档revision导致另一轨正常编辑触发本轨Conflict — 比较实例参数投影，只有本轨权威或宿主模型实际变化才拒绝旧写入/发布 — 若无条件忽略revision，会接受同轨旧GUI覆盖；共享与非共享身份回归均保留。

Task 27: Ruling: 用户正在修改隔离REAPER工程，不能热覆盖模块；首轮冷查询测试预热线程不退出锁住默认测试exe — 不关闭用户REAPER，编译同源别名hifishifter_plugin-feedback-check.exe验证（62 passed、exit0），新bundle生成到全新embedded-feedback-01目录 — 若覆盖正在加载的DLL或只信测试摘要，会丢编辑/虚报通过。

Task 28: Ruling: 原映射BPM固定120，宿主tempo从未进入GUI — 读取有效ProcessContext tempo，actor版本通知同步原GUI，真实REAPER120→150后GUI=150 — 若外推完整Tempo Map或time stretch，会把数字同步误当音频时间变换支持。

Task 28: Ruling: REAPER默认Beats时间基准改BPM会隐式将倍率变1.25 — 仅在独立副本用Time基准隔离验证，保留原工程默认值与time stretch未支持边界 — 若替用户改项目Timebase，会改变其全部素材节奏。

Task 28: Ruling: ARA播放服务可选且回调受模型线程和文档生命周期约束 — 已验证HostClients产生可撤销请求租约，只在native UI发送标准请求；确认包不伪造实际播放，暂停后定位停止点 — 若在actor调用或只存裸ref，会越线程或在关闭后调用宿主悬空地址。

Task 28: Ruling: 用户指明先点窗口；实测点标题栏后确已进入FX焦点，但网页按钮点击仍被本机工具拒绝 — 记录实际错误及BPM已实测/按钮未实测的区别，不重建输入代理 — 若忽略焦点证据或泛化为插件无法操作，会误诊本机工具限制。

Task 28: Ruling: command.txt残留save可能在新采集实例重放 — 启动时清空控制文件、支持独立ScratchName，验证始终在工程副本 — 若保留残留命令，会覆盖用户此前的测试编辑。

Task 28: Ruling: 集中审查发现首次Unsupported状态未显示，几何加载失败会连带阻断时钟查询 — 首载保存错误，get_playback_state提前分发，真实停播/暂停观察不依赖可编辑几何 — 若不修，GUI会永远等待音频或无法确认宿主已经停止。

Task 28: Ruling: 新增首载拉伸回归首轮64通过/1失败，Snapshot只序列化take权威而扁平倍率反序列化默认1 — 检查前normalize_takes重建真实投影，完整65/65 exit0再构建新版bundle — 若仅改测试期望，真实Beats隐式拉伸会漏报为可编辑。

Task 28: Ruling: 原GUI保持打开时，宿主BPM150→180及单素材起点0→1秒均自动跟随，无手工重载，日志clipStartsSec=[1,3] — 关闭无pending场景的连续同步疑问，保留pending冲突和网页按钮手绘验收open — 若外推所有冲突/Tempo Map/音频验收，仍会夸大证据范围。

Task 28: Ruling: 原生Duplicate tracks后共享源1/sequence2/region4，两个GUI载入已应用，模型过渡仍记两条unknown host track — 记录实测加载进展和残余日志，不宣称双轨曲线/PCM全验收；最终前端311文件/2715测试exit0 — 若仅凭窗口正常忽略过渡请求，会遗漏刷新时旧轨道请求的时序问题。

Task 29: Ruling: 原GUI初始store含独立app的track_main，首个真实快照前就取参数，插件从未授权该轨道 — 插件lazy初始化空轨道/空选择，独立app保持默认Main；19定向回归/tsc exit0，双轨重开新日志无旧错误 — 若在后端忽略未知身份，会削弱跨实例隔离并隐藏虚构首帧。

Task 29: Ruling: 实测Tab只循环宿主原生控件，自有编辑器无Tab停留且onFocus空实现 — 仅补自有窗口WS_TABSTOP、标准MoveFocus和受线程/token校验的焦点转交，不改父窗口、不重建输入代理 — 若只解决工具点击或吞宿主按键，真实键盘用户仍无法进入编辑器且可能损坏REAPER快捷键。

Task 29: Ruling: GetNextDlgTabItem在零Tab stop时也返回首child，初版回归假通过 — 加真实HWND停留能力校验后旧源码正确失败，再测实际浏览器焦点；不把Windows首child回退当可操作证据 — 若只看返回句柄，测试会掩盖已实测的不可达性。

Task 29: Ruling: 标准焦点修复后Shift+Tab实际进入HTML，原Space请求宿主Start及循环中的主动Pause，宿主和两GUI秒位置停在2.414 — 关闭键盘播放控制可行性疑问，仍区别于鼠标按钮点击；REAPER时间/tick格式不据截图强行等同 — 若把短素材自然结束当主动暂停，会形成假验收，故仅在隔离副本临时开循环后恢复Off。

Task 29: Ruling: 逻辑pianoRoll表面已切换但DOM仍在顶栏，画布局部Ctrl+0/Ctrl+I收不到事件 — 素材范围转参数选区时聚焦自有scroller，批量与单素材入口一致；真实原对话框输入64、键盘激活确定后自动1/1 — 若改测试为直接IPC参数提交，会绕过用户需要的原GUI路径。

Task 29: Ruling: 第二轨独奏输出四窗口220.5→329.104Hz、gap0，关闭全部FX窗口后输出PCM maxdiff0 — 归档真实GUI编辑/自动渲染/无UI供音证据，验证器显式接受第一段起点1秒且拒绝错误布局，不重排PCM — 若仅比较hash或波形变化，会把gain-only/错位基线误报为修音。

Task 29: Ruling: 一次集中review确认native线程/token/重入边界，无其它Critical/Important，仅单素材焦点漏接 — 补齐同契约分支，不追加review轮次；多参数面板广播时最后监听器获焦点暂记Minor残余 — 若对未验组合泛化通过，会掩盖多面板交互时序问题。

Task 29: Ruling: 正常退出REAPER冷重开测试RPP，第二轨GUI尚未打开时恢复供音PCM maxdiff0，之后原FX恢复MIDI64曲线 — 关闭键盘正向编辑/自动合成/保存重开这条实际链路疑问；最终前端312文件/2718测试和bundle exit0，鼠标/双轨均编辑/资源矩阵仍open — 若把新会话代次0/0当丢曲线，或把单条链路外推完整二期，都会误报状态。

Task 30: Ruling: 单轨已编辑而另一轨0/0只证明隔离的部分场景 — 用新scratch复制归档RPP，原GUI两轨设不同音高并逐轨Solo验证撤销/重做/冷恢复，不重试同一鼠标工具拒绝、不覆盖已有证据 — 若只凭两个窗口存在，可能漏掉第二个actor写入对第一轨的覆盖。

Task 30: Ruling: 第一轨原GUI60的宿主Solo输出四窗口260.94674556213016Hz/gap0，但第一次进程缺失时RPP仍与旧归档相同 — 保留真实输出，确认无进程后才重开/重新原GUI编辑并保存，第二轨67及本批冷恢复不标通过 — 若把导出或save完成当作恢复证据，会遗漏未保存/未冷启的最后一笔。

Task 31: Ruling: 保存后REAPER数次未响应与状态查询超时，后来恢复True；源授权callback同步调用edited render_edits并持文档事务 — 不误称死锁，新增独立release构建对照并将模型线程重复合成纳入前置任务；检测到用户输入即停发键鼠，不热替换/强杀 — 若只换优化构建或把全部延迟归于工具，会掩盖真实线程/生命周期风险。

Task 32: Ruling: 用户希望同一原GUI显示多轨，现EditorSession确实按组件缩窄而DocumentSession已有共享模型/权威 — 采用真实文档级唯一actor/授权区域并集，保持逐renderer输出和逐组件state，完整双轨验收转Task35；宿主多个FX面板是同步视图，不强改父窗口 — 若只拼接客户端或让master混整工程，会破坏撤销一致性、权限与宿主音频路由。

Task 31: Ruling: release构建6m00s/exit0，bundle引擎与实际release源SHA一致、thin三exports齐全、邻接依赖存在；原加载模块hash未变 — 归档构建事实和独立产物，不宣称宿主优化验收或主线程风险已修 — 若仅看build绿灯就关闭耗时门，会把未测的用户卡顿留在正式使用链路。

Task 31: Ruling: 当前批后台准备首轮lib70与收尾render26通过，但一次集中审查发现冷恢复单轨过期无重排和assignment校验/发布竞态 — 统一先合并全部恢复，分配写入/撤销与发布共用短事务，补真实无GUI冷恢复和native observer回归；用户随后要求只在最后review，后续不再逐任务派review — 若只相信绿测，可能冷启静音或重新播已移除区域。

Task 36: Ruling: 用户报告播放头跳动，源码多个renderer写共享clock、prefetch仍更新、任一实例停处理会改共享playing — 先采集实际时钟来源与mode，不把候选当实测根因；采用工程多轨GUI并保留逐轨输出，时钟也必须收敛明确权威 — 若只让GUI取最后写入或取最大时间，会继续抖动或破坏seek/loop。

Task 37: Ruling: SDK明确纯editor renderer必须透传输入，当前两角色都快照替换；普通fade曲线不是content-based fade标志，拉伸比例已映射但主动拒绝 — 先核实editor透传/宿主最终fade，再分别评估GUI fade数据与线性拉伸开放，不只移除拒绝逻辑 — 若把这些都叫自动同步完成，会漏音频覆盖和参数时间映射错误。

Task 31: Ruling: review修正后的定向bound_tests11正常exit0，冷恢复双轨无GUI实际PCM/模型线程assignment序列化可证；总73未重复全跑，现preparation-01为修正前先导 — 只提交源码/准确记录验证范围，native最终包/性能仍open；不继续派review，按用户要求到最终集中一次 — 若把旧bundle加载日志当作最新源码通过，会漏掉修正未部署。

Task 36: Ruling: editor-only按SDK透传，后台不再合成其歌曲快照，完整lib76 exit0；原生Playing首段82次回跳且主要mode1 — 保留真实诊断，不能排除所有prefetch，官方typed API在UI/model线程读所属project实际位置；getter/时钟回归通过不是宿主修复通过 — 若滤掉mode1或读当前活动project，会冻结游标或跨工程串时钟。

Task 38: Ruling: 用户再次要求时间拉伸必须，原kernel已有保调管线而plugin在GUI/几何校验挡住 — 正向线性比率进入真实kernel，0.5秒220Hz→1秒保调/非静音及GUI倍率3回归通过，plain resampler仍拒绝拉伸；继续完成marker/tempo/坐标重投影，不缩小完整目标 — 若只显示长度或去掉错误但不验证音频，会截尾/移调/曲线错位。

Task 36: Ruling: 用户物理Esc停止Computer Use，随后续目标并非明确重启界面授权 — 本轮继续源码，不重开CU；host-stretch-01构建exit0只记产物，最终native门保留 — 若自动抢回窗口，会违背用户中断并把未测源码当宿主已通过。

Task 36: Ruling: host-stretch-01前端/Rust/native构建exit0，最后plugin lib80/0失败/exit0 — 归档源批并转工程工作区开发，保留真实新模块/拉伸/时钟验收open；之后只在最终review，不重跑旧前端全量 — 若仅以库测试/构建结束替代原GUI宿主验收，会过早关闭新四项目标。

Task 32: Ruling: 文档workspace仅取活组件真实assignment并集，同名/同源轨道保留，空scope零轨；本批2个定向回归exit0 — 提交范围基础并推进共享actor，GUI尚未接入不宣称多轨已完成 — 若把只读投影当交付，会遗漏跨轨历史/组件保存与独立音频隔离。

Task 33: Ruling: 用户坚持时间拉伸必须、倒放暂缓且最终一次review — 按完整集成spec继续，单implementer处理共享actor并在批末测试，不恢复此前Esc停止的Computer Use — 若用旧unsupported或本地绿测缩小门，会虚报完整拉伸/宿主GUI行为。

Task 38: Ruling: 官方锁定REAPER头提供parent take但未给ARA hostRef转换opcode；7.81淡化又新增DIR_NEW/DIR2_NEW，原kernel markers只持久化不渲染 — 以直接所属take+真实assignment建立绑定，先记录接口/契约缺口，不按名字/位置猜对应；后续将参数坐标与marker管线一起接入 — 若沿用旧字段或把时长比/字段保存当完整支持，会继续曲线漂移或错误fade/拉伸。

Task 33: Ruling: 真实文档唯一原actor/跨轨history、scope指纹、queued路线代次与事件授权、global solo逐轨PCM在2cdf48ff；批末lib89/前端74/tsc及实际document apply回归exit0 — 继续Task34详细v2/冷恢复，保留native多轨GUI和完整拉伸门；只最后review，不部署旧bundle冒充新版 — 若仅以共享Arc或绿测结束，会漏逐组件保存、宿主窗口及音频链路。

Task 33: Ruling: 短actor filter曾打印摘要后退出挂住，full89及无actor对照正常exit；无证据把FCPE预热确定为根因 — 保留本轮自有PID/输出调查，未改kernel禁模型、未用强杀当Green，显式doc.close三秒内join/释放/撤销通过；后续继续寿命门 — 若把摘要当退出码或随意禁预热，会掩盖本机退出异常或损坏真实分析功能。

Task 34: Ruling: 三个真实RED暴露组件终止未即时撤销输出/view、已排队关窗尾笔被丢、doc.close后旧actor Arc滞留分析缓存 — a5dfa654最小修正并保留route代次/同名view身份检查，12定向回归exit0；组件/文档关闭不混为一谈，v2形状不变 — 若直接忽略closed或清整个共享actor，会越权写入或把其它轨道编辑停掉。

Task 34: Ruling: 旧归档原始v2 481/22516bytes、实际COM/Weak/IBStream、尾笔/undo范围保存和未开GUI后台publisher WORLD冷恢复已有源码证据 — A60≈260.947/B67≈390.265Hz，undo后A60不变/B64≈329.104Hz；继续完整宿主变换，不以这些fake host/source测试冒充REAPER原窗口/冷重开 — 若把库oracle当真实宿主验收，会漏host state/window时序及变换链。

Task 38: Ruling: 普通fade改变未必推进ARA model revision，强host interface引用也不保活project/take — typed几何批次前后校验官方GetProjectStateChangeCount，并在各getter前后重查doc/owner/scope以容许重入；缺direct take保持不可用，撤回虚假REFLECT_TEMPO/CONTENT_FADES能力但保留线性TIMESTRETCH — 若只检查开始时活性或广告未实现能力，会有悬空查询或宿主停止自行淡化。

Task 38: Ruling: kernel ClipStretchMarker已含秒/速度变化语义，原生marker单位/坡度还未实测 — 38a用独立HostStretchMarker raw字段保存，最终mapper拿到证据后显式转换，不伪造kernel参数 — 错了只需后续转换/接口调整；若提前套字段名，错误会扩散到渲染/缓存。

Task 38: Ruling: 已确认秒域的源↔项目时间需要统一正逆映射，普通总倍率不能表达分段变化 — 新ara/time_map.rs提供仿射/分段与新项目时刻回旧曲线坐标，先4项真实RED再GREEN/exit0；不接未验证raw marker、不用该数学测试冒充音频/曲线持久化通过 — 若映射各处自行推倍率，会在裁切/拆分和局部拉伸时漂移；下一步必须接编辑权威与保存而非只留helper。

Task 38: Ruling: 6b8fb67c typed直接take/逐getter授权/普通fade版本检查和诚实TIMESTRETCH广告、b5572a4e短COM初始化引用与重入终止清理已提交；19新定向/6相关旧回归自然exit0 — 源安全门通过，实际REAPER QI/parent/marker单位未验证，完整Task38不关闭 — 若以fake ABI合同当native证据，会在错误take绑定上开发整个变换链。

Task 38: Ruling: 只采GUI入口的元数据无法覆盖无GUI隐藏playback实例，且逐UI tick全量读marker会带来新的UI重任务 — 单一入口驱动同真实文档playback采集，只入口发布clock；模型/scope与project counter相同则两次轻量检查后复用成功数据，fade/model改变完整刷新 — 3新回归先RED后GREEN，连同4映射/4旧安全回归共11/exit0；若counter合同或采集性能与真实宿主不同，native门必须补测，不外推成完整同步。

Task 38: Ruling: 用户最新明确“不考虑非线性拉伸，只做线性拉伸” — spec/plan最终门改为固定倍率和tempo-timebase整段线性变化，非线性marker/坡度/段内warp不实现且明确不支持；其余四项目标不变 — 若错误排除线性曲线/音频/持久化验收会缩小用户真正要求；本裁定来自用户，不是为绿测自行降门。

Task 38: Ruling: 项目整轨数组不能作为移动/拉伸后的曲线权威，重叠区域也不能共用单一音频参数 — 新ParameterAtlas以真实region/modification/source边存不可变源basis，接接受事务/模型稳定投影/每clip独立原kernel参数/限定范围恢复；源曲线Arc共享并计入512MiB、临时渲染缓冲收费 — 6个新用例先RED后GREEN，末次8定向/旧失败5/旧归档v2各正常exit0；若GUI投影再被误当源数据，会累积插值损失或覆盖重叠区域，最终门仍需补此交互。

Task 38: Ruling: 源basis是新的持久化语义，旧v2只有绝对项目数组 — 有atlas的新保存采用v3、显式key不落盘且在组件实际范围重绑定，继续接受旧v2，不含atlas的旧数据仍编码v2 — 若版本号仍伪称旧语义，旧引擎会静默丢basis；代价是v3工程须用新插件打开，不覆盖用户旧RPP。

Task 38: Ruling: 批末128回归123通过/5失败，外部IPC clip扁平投影缺失、旧fixtures无modification边及JSON f64一ULP误判造成尾线截断 — 外部参数源basis只取宿主几何、fixture补真实边，不弱化验证；以8ε×max(1,绝对秒坐标)数值容差保留同布局整轨数组，定向5修复exit0 — 失败assert持Mutex使cleanup二次panic并0xc0000409，未当绿；若只改期望或当内存溢出忽略，会隐藏实际曲线保存损坏。

Task 40: Ruling: 用户明确 HNSEP 不分块、HiFiGAN 分块即可 — 保留 HNSEP 整段处理，停止其分块研究；长源神经分块门限定 HiFiGAN，HNSEP 参数/缓存/资源保护仍执行 — 若将此误解为取消 HNSEP 验证，会留下错源缓存或失效参数；整段成本须诚实报告。

Task 38: Ruling: 重叠region的整轨可见数组不能重新捕获给不可见素材，选择变化也不应推进编辑代次 — capture_changes只接相对当前选择投影的delta，select_clip/select_track从doc只读切换源投影；本批10定向回归在进程级CPU设置下正常exit0，不重复全量/不追加review — 若沿用整数组回捕会串曲线；这是源码actor/音频合同证据，仍非原生GUI验收，外部IPC旧提交路径另留残余。

Task 39: Ruling: HNSEP旧键只含clip/长度/64位源指纹，缓存仅按条数；真实并发请求也可能重复整段分离 — 改实际完整PCM+已加载模型/EP权威、128MiB字节LRU和worker单飞；真实CPU两owner1次推理、等长中间换源新增1次，气声/张力/共振峰下游修改复用stem — 若把源分离与目标参数合并键，会白跑分离；该缓存上限是kernel额外占用，不伪称已纳入plugin512MiB。

Task 40: Ruling: HiFiGAN虽512帧分块，却整段收集未命中输入/输出，chunk HashMap无字节上限 — 限每批4块立即拼接、128MiB chunk LRU，pipeline版本8明确失效旧PCM；35秒48k真实CPU冷7935ms/暖1078ms、6块尾454帧/完整非静音/暖新增推理0 — 若称常量总内存或ARA长源已完成，会忽略完整mel/HNSEP激活与宿主30秒cap；资源与native门仍open。

Task 39: Ruling: 插件先后44.1k/48k独立运行kernel会重复重推理 — 含HiFiGAN工作区仅合成44.1k，48k从就绪PCM派生并为交错源副本收费；真实RenderInput双输出798ms/HNSEP1次/派生逐样本一致，相关3回归exit0 — 若外推独立App所有采样率/跨owner缓存，会夸大局部改动；App完整回归、内容层和冷恢复仍待最终批。

Task 39: Ruling: 项目网格参数与clip名字会让纯摆放改变音频/缓存，模型/config版本也未进入HiFiGAN内存键 — atlas直接投影region局部零点、逐region局部kernel合成后摆放；宿主DSP完整内容不带clip名，共享内容worker单飞可取消，HiFiGAN键接实际已加载模型/config完整摘要/EP/块大小；2新kernel合同、12插件回归和1真实模型诊断均exit0 — 实测移动到512.013秒/换region名PCM逐样本一致且新增HiFiGAN推理0，修改目标音高新增1/HNSEP仍1；若外推旧v2移动或磁盘冷复用仍会误报，后者尚未接入。

Task 39: Ruling: 宿主PCM仍被file-only整clip磁盘门禁排除，不能直接删除门禁 — 新隔离ARA mono合成内容cache，完整PCM/模型/有效参数键、96-byte有界头/完整payload摘要、64MiB条目/2GiB配额、唯一临时文件原子替换，只在worker读写且取消后不写；新进程574ms命中/HiFiGAN与HNSEP合成run均0/PCM摘要一致，故意损坏后拒绝并重建，3格式/键/配额合同exit0 — 若外推REAPER保存冷重开或长素材512MiB方案会夸大证据；App文件缓存/用户工程未动，缓存管理GUI和长源峰值仍待收尾。

Task 41: Ruling: 用户明确恢复Computer Use，先确认无REAPER后构建新release bundle A42E55C2…/exit0并在隔离副本启动 — 实测一个原GUI两轨、旧v2第二轨WORLD输出与归档PCM相同/关GUI输出maxdiff0，原RPP hash不变；不追加review、不把旧bundle或库测试冒充本轮native — 若外推双轨新编辑/HiFiGAN参数，会夸大仅旧曲线恢复的证据。

Task 40: Ruling: 冷启不打开GUI立即离线导出全零，后台ready后同实例/同参数定向再导出maxdiff0 — 把offline后台准备竞态列为必须修复，保存恢复本身仍有效；不将第一次静音当成功或归为用户没开App，不在实时process加等待/IO — 若只等一会再导出并宣称通过，正常用户仍可得到静音成品；先补真正offline就绪契约。

Task 36: Ruling: 真实REAPER日志扩展available=false/position_authority=false，两次播放backwards计数265；网页click仍被Computer Use安全边界拒绝 — 保留宿主初始化分阶段诊断待办及GUI交互未测，明确游标/fade元数据门失败，不猜parent或绕过工具注入 — 若以typed fake ABI/可见两轨当全部可用，会漏掉真正高优时钟问题；隔离实例均正常退出，无强杀。

Task 40: Ruling: 官方VST3头明确kOffline切换经过UI线程setupProcessing，后台就绪后导出才正确 — 只在离线setup等待worker并核对model/edit/epoch/scope/keys，两率完整发布后登记；offline缺快照/上下文明确失败，实时process/setProcessing仍无等待/IO/分配 — 8定向exit0；新7E37包真实冷启不打开GUI立即首导PCM maxdiff0/RMS0.08688376956，旧全零RED已关闭此夹具门；若外推全部长源/设备/算法仍会夸大。

Task 36: Ruling: 实际QI成功但initialize parent(project)为null，旧代码把可用扩展整个丢弃 — 保留拥有引用的接口，稍后仅在原model/UI线程取同一直接parent(3)非空绑定，不随活动tab重绑、不以null猜当前项目；2新合同及旧初始化/几何/离线共21回归exit0 — 真实新时钟/geometry仍须新包验证，不能把源码绿测当游标已修。

Task 41: Ruling: Computer Use报告用户输入，42280实例随后标题modified — 按技能停止自动键鼠，保留该实例与用户编辑；新延迟绑定包只在全新目录构建，既有7E37加载DLL不替换，不追加review — 若擅自关闭/重开或发送脚本，会再侵入用户资产；当前任务继续源码，不将这一暂时native边界当技术不可行。

Task 42: Ruling: 用户确认已关闭REAPER，并要求四BUG集中修完后再由其一次验收，Computer Use逐项操作太慢 — 停止自动UI操作，新增四BUG为完整目标的必达项，源码/回归/构建集中收尾后才启动隔离实例交用户测 — 若中途反复叫用户验证，会违背新批次要求；五项目标和HNSEP不分块边界不变。

Task 42: Ruling: actor仅在队列超时执行应用，source投影清orig key而clip cache命中不重发完成通知 — 当前源码把到期调度放到循环顶部、核对已处理写入票据，并由actor组装缺key根的已有分析cache，不依赖GUI轮询原线；本批尚未验证，不标修复完成 — 若只去掉pending门或将非空数组当分析完成，会在早落笔/清音/换源时发布错误参数。

Task 42: Ruling: 自动render失败旧路径被合并为authority error，后续apply无法恢复；气声等只改效果也需要原线 — 分离render_error并自动重试，复用原processor需求门禁与完整缓存组装，Conflict仍保留曲线；actor首轮28/29，修正张力夹具漏开分离开关后该项及host合同14项exit0 — 若把所有错误清掉或先发布raw，会静默绕过冲突/遗漏气声；尚无本批native结论。

Task 36: Ruling: 前端将宿主位置加上完整RTT，合成排队越久越前漂；actor重合成也阻塞播放轮询 — 同route/view准入后原子快查播放态，取消宿主RTT外推并将插件视觉补帧限制100ms，真后退seek/loop仍接受；前端28项与tsc exit0，backend持timeline锁快查合同通过 — 若用最大时间或忽略全部后退掩盖跳动会破坏seek/loop；最终实机播放头仍待用户集中测。

Task 38: Ruling: 官方7.81新曲率有c/S双轴，旧shape/dir已非当前形状权威 — 用独立UI版本投影真实长度/auto长度和原始新轴，GetAppVersion选择语义；参数提交清普通fade避免二次烘焙；新非零轴只显示范围/数值不画猜测曲线 — 若称完整fade形状已支持会误导；任意新轴精确曲线oracle仍open，本批不启动REAPER或中途叫用户验收。

Task 40: Ruling: 内嵌GUI把整份PCM经IPC形态复制，源/分析30秒与快照64MiB是旧探针限制 — 改同进程授权Arc冻结、流式分析WAV及固定512MiB整批PCM预检，reader保持4096帧窗口；单轨180秒GUI/两率尾部/seek通过，显式额度峰值265,248,000字节 — 若简单提常量或把额度计数叫RSS，会漏掉多轨新旧快照/NN工作内存；稀疏/共享快照、真实NN长宿主与RAM/GPU门仍open。

Task 42: Ruling: 从未读过参数面板的初载没有根参数条目，分析完成不能写回；同URI换PCM也不能沿旧orig就绪标记 — 首载创建实际根条目，清项目key后主动组缓存，gen0/缓存全命中也明确请求重新准备；真实220→440Hz原线57→69且目标60保持，无reload/getter推进 — 若仅修已存在key条目，会把首次初始化或冷恢复继续锁成手工重载；32项首轮31过，此新增门修后1项及7护栏exit0。

Task 39: Ruling: 私有分析目录带model/edit代次且每次重写mtime，布局改变会制造新的原线cache身份 — 分析路径按完整授权内容命名，逐样本校验已有WAV后复用，损坏以临时文件原子重建；无gen宿主移位路径/mtime稳定合同通过 — 若凭路径名或只验长度信任旧文件，会用损坏/同ID旧PCM作为F0基线。

Task 41: Ruling: 单独kernel禁用vslib的测试编译发现6个缺失名字，导入错误地受vslib门控 — 仅修测试import，使WORLD/HiFiGAN共通混音护栏仍在无vslib配置可编译，3定向exit0；不启用闭源vslib、不更改产品DSP — 若把插件依赖lib能编译当kernel全部测试能运行，会在最终统一门再次漏掉配置错误；完整基线仍待最后一次统一回归。

Task 40: Ruling: 双轨长素材原子更新因重复mono平面与单region额外全长mixed浪费预算 — worker逐bit归并相同声道并在每renderer准备后立即退额度，单region直用原kernel结果；实际双轨180秒修改第二轨两率不串且完整尾部，额度峰值497,088,000<536,870,912 — 若做downmix或在音频线程回收会改声音/破坏实时；首次预算断言补入1,152,000 atlas曲线字节后通过，不虚报模型资源门。

Task 42: Ruling: actor成功发布后闲置后台邮箱可能仍报旧失败，同代重复准备也会争长源预算 — 只在已校验成功发布时清空闲历史error，运行任务不伪成功；PreparedVersion相同且两率已ready的后台任务复用 — 若无版本/快照就绪检查就清错误，会掩盖真实失败；mailbox运行失败/缓存正确与跨轨输出合同通过。

Task 40: Ruling: 三分钟真实HiFiGAN多批/整段HNSEP功能与暖缓存正常exit0，但旧CPU会话结束驻留17.5GB/峰值18.3GB、提交23.2GB — 排查并关闭原生ORT Separator的arena/pattern，不切块HNSEP，其它模型和Intel macOS原策略保持；短真实mask旧新逐bit相同，三分钟新诊断exit0结束驻留1.3GB、峰值仍13.3GB — 若把功能通过或PCM额度331MB说成资源完成，会让普通机器OOM；后续还需产品级资源保护及峰值优化。停止检查前旧诊断已正常结束，未实际杀进程，未碰49800或REAPER。

Task 45: Ruling: 用户新增最终产物为正式构建流/文档、App插件易同步、条件性Linux/macOS使用 — 纳入Task45-47验收，继续共享kernel与同一原GUI权威，并把Windows-only视图/模块与平台实测缺口明确列出，不替代五目标或四BUG — 若只交付probe命令或把内核跨平台当ARA GUI可用，会遗漏最新交付要求。

Task 47: Ruling: 用户随后将Linux/macOS插件标为“本次不作要求” — 移入后续事项，本轮继续Windows完整产品/构建与同步交付，不开新的跨平台插件实现支线；独立App原跨平台功能边界保持 — 若继续为跨平台嵌入扩张本轮，会违背最新范围并延迟交付；不把未实现平台声称已支持。

Task 40: Ruling: 10秒真实ORT profile末级Concat输出343,277,568字节，输入65+32通道与输出97通道同时存活，三分钟该下界约12.3GB — 推理前加入整段资源估计和Windows物理/提交空间、Linux MemAvailable保护，cache成功命中仍复用，低资源明确失败不忽略效果；3合同及短真实模型/cache链exit0 — 若用PCM额度或线程调小承诺消除大图下界会误导；估计不是自定义模型严格上界，低峰值完整图与GPU/macOS资源边界仍open，HNSEP不切块。

Task 45: Ruling: 用户要求后续迭代新功能时App/插件可同步，现有业务/DSP与原GUI已有唯一共享源码但构建命令分散 — 新统一入口一次frontend、分开App/插件Cargo特性、原生loader/resources打包与全新目录，构建文档列共享修改点/宿主能力例外/同批Verify；语法与PlanOnly exit0 — 若将All构建合成一次Cargo调用会把vslib特性带入插件；真实All构建和native验收尚未通过，不报交付完成。

Task 45: Ruling: 真All构建暴露MSVC环境脚本$name覆盖带校验的Name参数；随后manifest的false未写成PowerShell $false — 内部参数改BuildName保留别名，补manifest表达式检查；All Debug产物及45文件摘要验证exit0，记录源码/model指纹与nativeAcceptance=false，目录不覆盖 — 若只跑PlanOnly/语法就称构建流可用会漏实际变量作用域/执行语义；Release、集中Verify与用户GUI验收仍未完成。

Task 45: Ruling: 用户再追加单独简短中文VST使用说明，并明确同步要求面向后续新功能迭代 — 新增docs/hifishifter-vst-使用说明.md，按当前实际UI/轨道FX接入方式写安装、多轨、焦点、自动应用、保存与限制；开发/构建同步文档独立保留 — 若把长篇开发文档当使用说明或将未校准fade/未实机四BUG写成已过，会误导用户；最终验收后还需更新能力结论。


