# ara2-bridge

[![CI](https://github.com/entrepeneur4lyf/ara2-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/entrepeneur4lyf/ara2-bridge/actions/workflows/ci.yml)

Rust infrastructure for authoring ARA 2.3 hosts and plug-ins. The project targets the complete
redistributable public surface of Celemony's ARA SDK 2.3: the core ABI, generation compatibility,
host and plug-in runtimes, content and persistence utilities, and CLAP, VST3, and Audio Unit v2
companions. DSP algorithms, AUv3, and the separately licensed AAX API are outside this scope.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `ara2-bridge-sys` | Pregenerated, target-selected raw ABI and compatibility metadata |
| `ara2-bridge-core` | Shared safe types, validation, registries, and dispatch |
| `ara2-bridge-plugin` | Plug-in factory, document controller, and extension roles |
| `ara2-bridge-host` | Host services, document graph, and plug-in dispatch |
| `ara2-bridge-companion` | CLAP, VST3, and Audio Unit v2 adapters |
| `ara2-bridge-testkit` | Mock peers, fixtures, and conformance scenarios |
| `ara2-bridge` | Aggregating facade |

The current `0.3.0` implementation includes the plug-in and host runtimes, core content and
persistence utilities, CLAP/VST3/Audio Unit v2 adapters, deterministic conformance kit, native C++
interoperability, safety harnesses, and provenance-aware raw bindings. Platform-specific native
evidence is collected by the runner matrix before a release is declared conformant.

## Getting Started

The facade defaults to plug-in authoring:

```toml
[dependencies]
ara2-bridge = "0.3.0"
```

Run the public examples from a checkout:

```bash
cargo run -p ara2-bridge --example minimal-plugin
cargo run -p ara2-bridge --example minimal-host --no-default-features --features host
cargo run -p ara2-bridge --example archive-roundtrip
```

See [the build guide](docs/building.md) for the complete platform, feature, SDK, and maintainer
build process; [companion SDK setup](docs/companion-sdk-setup.md) for locked native inputs;
[the migration guide](docs/migration-0.1-to-0.2.md) for the intentional 0.1 API break; and
[the manual source map](docs/manual-source-map.md) for the 12-chapter inventory.

Projects using native companion features can install and build the complete locked SDK locally:

```bash
curl -fsSLO https://raw.githubusercontent.com/entrepeneur4lyf/ara2-bridge/v0.3.0/scripts/install-ara-sdk.sh
bash install-ara-sdk.sh
cargo build
```

The installer uses the invoking project's Git root, places sources under `.third-party/`, builds the
Celemony examples under `target/ara-sdk-build`, and records relocatable paths in
`.cargo/config.toml`. It never needs `sudo` or writes into a global SDK location.

## Development

Rust 1.82 or newer is required. Package builds without native companion features do not need Clang
or an SDK checkout. Maintainer generation requires Clang and the project-local SDK installation.
Follow the prerequisites and tiered workflow in [the build guide](docs/building.md); the short
quality gate is:

```bash
bash scripts/install-ara-sdk.sh
cargo xtask ara generate --check
cargo xtask ara probe-core --check-all
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

Do not edit files under `ara2-bridge-sys/src/generated/`; regenerate them through `xtask` and
commit the provenance and ABI evidence with the change.
