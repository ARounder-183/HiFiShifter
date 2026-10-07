# ara2-bridge-core

Shared safety infrastructure for the `ara2-bridge` host and plug-in runtimes.

This crate validates versioned packed ARA records, copies caller-owned strings and arrays into
aligned storage, owns model identities in bounded stable registries, coordinates API generations,
enforces model-thread and lifecycle rules, and contains panics at callback boundaries. It does not
implement a host or plug-in by itself.

## Public modules

The crate exposes typed content events and readers, checked random-access archives and filters,
bounded ARA iXML plus RIFF/FORM rewriting, timeline and harmony utilities, owned channel layouts,
processing-algorithm backing, and license-request validation. See the `content-reader`,
`archive-roundtrip`, and `audio-file-chunk` examples for pointer-free usage.

## Safety boundary

Foreign pointer entry points are `unsafe` because portable Rust cannot prove an arbitrary C address
readable. Their `# Safety` sections state the required allocation, extent, lifetime, alignment, and
nested-pointer rules. Once those preconditions hold, malformed values are rejected before dependent
reads. Packed fields are copied only through generated unaligned accessors.

Application objects never become ARA references directly. `Registry<K, T>` owns stable cells;
`Handle<K>` prevents cross-kind use and is neither `Send` nor `Sync`. Removal tombstones a cell
before returning its value, and cells are never reused within a session.

## Runtime policy

`Lifecycle` checks editing, generation-specific restoration, sample/content access, rendering,
poisoning, and teardown. `RealtimeHeadTailView` is immutable and allocation-free on lookup.
Dispatch adapters catch panics, record structured diagnostics, poison the runtime, and return the
method-specific ABI sentinel.

Run focused verification with:

```bash
cargo test -p ara2-bridge-core
cargo clippy -p ara2-bridge-core --all-targets -- -D warnings
cargo +nightly miri test -p ara2-bridge-core --test registry
```
