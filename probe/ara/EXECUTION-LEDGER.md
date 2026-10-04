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


