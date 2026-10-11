# 本地检出说明（HiFiShifter）

这份源码是 crates.io 上 `ara2-bridge-testkit` **0.3.0** 的逐字拷贝，外加
`src/plugin.rs` 里一处**纯适配性**的补丁（见下）。

## 来源与身份

| 项 | 值 |
| --- | --- |
| crate | `ara2-bridge-testkit` 0.3.0（2026-07-16 发布，未 yank） |
| `.crate` 的 sha256 | `211a178428af5bb88881fcdea7ddc0399ee8589ab94114cbf873f089801b584f` |
| 上游仓库 | <https://github.com/entrepeneur4lyf/ara2-bridge> |
| 上游 git sha（`path_in_vcs = ara2-bridge-testkit`） | `9caaa01e22573a3432a6c65350cf65bdd2e655cc` |
| 许可 | `MIT OR Apache-2.0`（与本仓 MIT 兼容） |

本目录随仓分发，但**不含** `Cargo.lock` 与 `target/`（构建产物，已 gitignore）。

## 为什么要 vendor

本仓用 `[patch.crates-io]` 把 `ara2-bridge-core` / `ara2-bridge-plugin` 指向
`backend/third-party/` 下打过补丁的版本。其中 **plugin 补丁改了 trait 签名**
（见 `../ara2-bridge-plugin/PATCHED.md`）：`AudioModifications` / `PlaybackRegions`
各自多声明了关联类型，且三个工厂方法各多了父子边参数。

crates.io 上发布的 testkit 是按**上游签名**写的，直接引用会编译失败
（`E0046` 缺关联类型 / `E0053` 签名不兼容）。所以必须 vendor 并同步适配。

## 补了什么

只改 `src/plugin.rs` 里 `TestPluginModel` 的两个 trait 实现，共 4 处：

| 位置 | 改动 |
| --- | --- |
| `impl AudioModifications` | 新增 `type AudioSource = u64;`（与 `AudioSources::AudioSource` 同型） |
| `create_audio_modification` | 新增 `_: &Self::AudioSource` 参数 |
| `clone_audio_modification` | 新增 `_: &Self::AudioSource` 参数 |
| `impl PlaybackRegions` | 新增 `type AudioModification = u64;` 与 `type RegionSequence = u64;` |
| `create_playback_region` | 新增 `_: &Self::AudioModification`、`_: &Self::RegionSequence` 参数 |

`src/scenarios/*` 与 `src/native.rs` **不需要改**：它们调用的是
`ara2-bridge-host` 的编辑会话 API（`edit.create_audio_modification(source, ..)`），
那是宿主侧接口，本就带 source 参数，不受 plugin 补丁影响。

## 与本仓工作区的关系

- `backend/Cargo.toml` 的 `exclude` 列入了本目录，`cargo test --workspace`
  不会连带编译它（cpp-interop 的 C++ 构建因此不污染常规测试）。
- 本目录的 `Cargo.toml` 带空 `[workspace]`，是独立工作区根，故可自带
  `[patch.crates-io]` 把 core/plugin 指回 `../ara2-bridge-core`、`../ara2-bridge-plugin`。

## 怎么跑

cpp-interop 需要 Celemony 的 `ARA_Library` / `ARA_Examples`（Apache-2.0）与
`ARA_Examples/3rdParty` 下的 cpp-base64（zlib）、pugixml（MIT）、ICST_AudioFile
（2-clause BSD）。这些**以 git submodule 形式拉取、不随本仓分发**（`third_party/sdk/`
整目录 gitignore）。准备步骤：

```powershell
. .\tools\msvc-env.ps1
. .\tools\sdk-env.ps1; Set-SdkEnvironment
.\tools\prepare-plugin-sdks.ps1 -WithTestkit     # 额外拉 ARA_Library / ARA_Examples
cd backend\third-party\ara2-bridge-testkit
cargo test --features cpp-interop --test cpp_interop
cargo test --test plugin_contract                 # 纯 Rust，不需要 C++ SDK
```

## 撤销

上游若把 trait 签名补成带父子边的形式，删掉本目录、删掉
`backend/Cargo.toml` 里对应的 `exclude` 行、删掉 `../ara2-bridge-core` 与
`../ara2-bridge-plugin` 的 `[patch.crates-io]` 段即可。
