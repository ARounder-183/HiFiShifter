# F-2 evidence: REAPER 7.82 sends no ARA musical content

Captured 2026-10-09 with `probe/ara/run_probe_headless.ps1 -SetupEel probe/ara/fixtures/ara_musical_content_setup.eel -SettleSeconds 20`.

Fixture self-check (`.build-tmp/f2-fixture.log`): `tracks=1 media_items=1 tempo_markers=2 track_fx=1` — the project really did have a two-marker tempo map and the plugin really was inserted.

The ARA document is built in full (source, region sequence, playback region, host PCM) but there is **not a single `[ara][probe]` line**: REAPER never calls `createMusicalContext`, so `probe_host_musical_content` never runs.

Plugin log excerpt:

```text
[INFO] hifishifter_plugin::ara::model: [ara] document controller created: apiGeneration=V2Final
[INFO] hifishifter_plugin::ara::model: [ara] begin_editing
[INFO] hifishifter_plugin::ara_entry: [ara] bind document=1 known=0x7 assigned=0x6
[INFO] hifishifter_plugin::render::extension: [ara] GUI channel ready instance=fe9aaef5-dad9-4864-8c31-d1594262d0f1
[INFO] hifishifter_plugin: [vst3] IAudioProcessor::setupProcessing mode=0 rate=48000
[INFO] hifishifter_plugin::ara::model: [ara] begin_editing
[INFO] hifishifter_plugin::render::extension: [ara] renderer assignment role=2 regions=0
[INFO] hifishifter_plugin::ara::model: [ara] audio_source #0: persistentID=D:/Projects/HiFiShifter/probe/ara/fixtures/tone44100.wav sampleRate=44100 sampleCount=88200
[INFO] hifishifter_plugin::ara::model: [ara] audio_source samples_access enable=true
[INFO] hifishifter_plugin::ara::model: [ara] host PCM ready source=0 frames=88200 version=0
[INFO] hifishifter_plugin::ara_entry: [ara] bind document=1 known=0x7 assigned=0x1
[INFO] hifishifter_plugin: [vst3] IAudioProcessor::setupProcessing mode=0 rate=48000
[INFO] hifishifter_plugin::ara::model: [ara] playback_region #0: source=D:/Projects/HiFiShifter/probe/ara/fixtures/tone44100.wav startMod=0.000000 durationMod=2.000000 startPlay=0.000000 durationPlay=2.000000 flags=0x1
[INFO] hifishifter_plugin::render::extension: [ara] renderer assignment role=1 regions=1
[INFO] hifishifter_plugin::render::extension: [ara] host inventory: 1 track(s), 1 item(s), 1 claimed by an assigned region
[INFO] hifishifter_plugin::render::extension: [ara] background snapshot ready role=1 revision=0 model=2 regions=1
```
