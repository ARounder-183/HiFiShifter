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


