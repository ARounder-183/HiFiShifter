# ara2-bridge-plugin

Safe ARA 2.3 plug-in authoring runtime for Rust. The crate owns factory storage, negotiated document-controller vtables, all 54 controller callbacks, model identities, graph ordering, host-service wrappers, optional semantic capabilities, and companion extension-role state.

## Authoring model

Implement the six focused required trait groups on one model type:

- `DocumentLifecycle`
- `MusicalContexts`
- `RegionSequences`
- `AudioSources`
- `AudioModifications`
- `PlaybackRegions`

Compose a fresh controller with `PluginBuilder::new(model)`, then register optional providers such as `content`, `analysis`, `partial_persistence`, `processing_algorithms`, `licensing_for`, `audio_file_chunks`, `signal_preservation`, and `realtime_head_tail`.

Register that constructor on an immutable `FactoryBuilder`:

```rust,ignore
let factory = FactoryBuilder::new("com.example.plugin", "com.example.archive")
    .display("Example", "Example Audio", "https://example.com", "1.0")
    .document_controller(|| PluginBuilder::new(MyModel::default()).build())
    .build()?;
```

The constructor must return a fresh `Plugin` for each document. Factory capability declarations must agree with controller providers; mismatches are rejected during controller creation.

## Host access and lifetimes

Eligible musical-context and audio-source hooks receive `HostContentScope`. The token permits host content reads only for the current object (or any live object during `end_editing`). Audio-source management hooks may create owned `HostAudioReader` values. Readers are synchronously revoked when sample access is disabled, their source is destroyed, or the controller terminates; retained handles become inert.

All graph mutations require a balanced editing session. The runtime enforces live parent edges, deactivation order, leaf-first destruction, stale/foreign reference rejection, ARA1 synthetic region sequences, and content-reader exclusivity. Application code never owns raw ARA model references.

## Asynchronous and realtime work

Clone `UpdateEmitter` and `AnalysisEmitter` from the builder before `build()`. Worker threads queue changes; the model thread delivers them only from `notifyModelUpdates`. `RealtimeHeadTailAdapter` publishes immutable snapshots and performs allocation-free, blocking-lock-free callback queries.

## Verification

Run:

```bash
cargo test -p ara2-bridge-plugin --tests
cargo test -p ara2-bridge-testkit --test plugin_contract
MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test -p ara2-bridge-testkit --test plugin_contract
```

The testkit fixture enables every optional capability and joins raw callback coverage against the generated 54-slot manifest.
