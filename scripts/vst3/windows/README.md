# Windows VST3 load probe

This directory contains the black-box load gate for the Windows x64 VST3 bundle.

## Why

A DLL can load successfully and still be unusable: GetPluginFactory may deadlock,
return null, or expose a broken factory. backend/hifishifter-plugin/tests/vst3_exports.rs
only checks the Rust engine DLL; it cannot see the native HiFiShifter.vst3 shim.
This probe exercises the same entry points a host does, in a separate process, with a
hard timeout.

## Files

- load_probe.cpp — minimal VST3 host-side probe. It does not include the VST3 SDK;
  it defines only the few vtable slots it needs.
- verify.ps1 — static export/dependency checks, compiles the probe, runs it with a
  timeout, writes .build-tmp/vst3-load-gate.log, .build-tmp/vst3-load-gate.err.log
  and native-load-gate.json.

## Usage

From the repository root, with the MSVC environment loaded:

```powershell
. .\scripts\env\msvc-env.ps1
.\scripts\vst3\windows\verify.ps1 -BundleDirectory .\.build-tmp\loadgate-test\HiFiShifter.vst3 -TimeoutSeconds 20
```

A passing run prints:

```text
countClasses: 3
class[0]: Audio Module Class / HiFiShifter
createInstance(component): ok
PASS
VST3 load gate passed: ...\HiFiShifter.vst3
```

## Exit codes

- 0 — probe passed.
- non-zero — probe failed or timed out; verify.ps1 throws with the log paths.

## Limits

The probe verifies module entry points, factory enumeration and component creation.
It does not test the editor, audio processing or ARA behavior; those still require
REAPER/VEGAS acceptance.
