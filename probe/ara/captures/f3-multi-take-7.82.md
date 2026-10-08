# F-3 evidence: what REAPER 7.82 sends to ARA for a three-take item

Captured 2026-10-09 with `probe/ara/run_probe_headless.ps1 -SetupEel probe/ara/fixtures/ara_multi_take_setup.eel -SettleSeconds 20`.

Fixture self-check (`.build-tmp/f3-fixture.log`): `takes=3`, `curtake=1` — one item, three takes pointing at three different files, active take is index 1 (`tone48000.wav`).

Read against that: **three** `audio_source` callbacks (one per take, all three files) each followed by `host PCM ready`, which is only logged after `render::source::read_source_pcm` succeeds — so the inactive takes' audio is reachable. And exactly **one** `playback_region`, pointing at `tone48000.wav`, the active take.

Plugin log excerpt:

```text
[INFO] hifishifter_plugin::ara::model: [ara] document controller created: apiGeneration=V2Final
[INFO] hifishifter_plugin::ara::model: [ara] begin_editing
[INFO] hifishifter_plugin::ara_entry: [ara] bind document=1 known=0x7 assigned=0x6
[INFO] hifishifter_plugin::render::extension: [ara] GUI channel ready instance=8de22180-5c20-456c-8af8-b620862c883c
[INFO] hifishifter_plugin::ara::model: [ara] begin_editing
[INFO] hifishifter_plugin::render::extension: [ara] renderer assignment role=2 regions=0
[INFO] hifishifter_plugin::ara::model: [ara] audio_source #0: persistentID=D:/Projects/HiFiShifter/probe/ara/fixtures/tone44100.wav sampleRate=44100 sampleCount=88200
[INFO] hifishifter_plugin::ara::model: [ara] audio_source samples_access enable=true
[INFO] hifishifter_plugin::ara::model: [ara] host PCM ready source=0 frames=88200 version=0
[INFO] hifishifter_plugin::ara_entry: [ara] bind document=1 known=0x7 assigned=0x1
[INFO] hifishifter_plugin::ara::model: [ara] audio_source #1: persistentID=D:/Projects/HiFiShifter/probe/ara/fixtures/tone48000.wav sampleRate=48000 sampleCount=96000
[INFO] hifishifter_plugin::ara::model: [ara] audio_source samples_access enable=true
[INFO] hifishifter_plugin::ara::model: [ara] host PCM ready source=1 frames=96000 version=0
[INFO] hifishifter_plugin::ara_entry: [ara] bind document=1 known=0x7 assigned=0x1
[INFO] hifishifter_plugin::ara::model: [ara] audio_source #2: persistentID=D:/Projects/HiFiShifter/probe/ara/fixtures/embedded-editor-voice.wav sampleRate=44100 sampleCount=88200
[INFO] hifishifter_plugin::ara::model: [ara] audio_source samples_access enable=true
[INFO] hifishifter_plugin::ara::model: [ara] host PCM ready source=2 frames=88200 version=0
[INFO] hifishifter_plugin::ara_entry: [ara] bind document=1 known=0x7 assigned=0x1
[INFO] hifishifter_plugin::ara::model: [ara] playback_region #0: source=D:/Projects/HiFiShifter/probe/ara/fixtures/tone48000.wav startMod=0.000000 durationMod=2.000000 startPlay=0.000000 durationPlay=2.000000 flags=0x1
[INFO] hifishifter_plugin::render::extension: [ara] renderer assignment role=1 regions=1
[INFO] hifishifter_plugin::render::extension: [ara] host inventory: 1 track(s), 1 item(s), 1 claimed by an assigned region
[INFO] hifishifter_plugin::render::extension: [ara] background snapshot ready role=1 revision=0 model=2 regions=0
[INFO] hifishifter_plugin::render::extension: [ara] background snapshot ready role=1 revision=0 model=2 regions=0
[INFO] hifishifter_plugin::render::extension: [ara] background snapshot ready role=1 revision=0 model=2 regions=1
```
