# HiFiShifter

[简体中文](../../README.md) | [繁體中文](README_zh-TW.md) | [English](README_en.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

HiFiShifterは、グラフィカルなボーカル編集・合成ツールです。マルチトラックのオーディオクリップ処理をサポートし、トラックグループ単位で複数のボコーダーを使用してボーカルのピッチ補正やパラメータ調整を行い、人力VOCALOID制作における編集と調声を一体化します。

**このプロジェクトはまだ開発中です。全体的なテストは完了しておらず、多くのバグや不安定な問題が存在する可能性があります。**

![プレビュー](../preview.png)

## 機能一覧

- **トラック編集**：DAW ライクなマルチトラックタイムライン。クリップのトリミング、ストレッチ、スリップ、フェードイン / フェードアウトとクロスフェード、グループ化、テイク管理、無音検出、リップル編集、テンポマップ（BPM / 拍子 / 音階）、メトロノームや録音などに対応。
- **パラメータ編集**：トラックグループ単位で、ピアノロール式のパラメータエディターによりピッチ、ボリューム、ダイナミクス、パン、フォルマント、ブレス、テンションなどのパラメータラインを調整。描画 / 直線 / ビブラートツール、複数選択範囲の編集、ビブラートプリセット管理に対応し、子トラックでは `セント差` / `度数差` / `フォルマント差` を重ねてハーモニーを作成できます。
- **3 種類のボコーダーアルゴリズム**：nsf-hifigan（PC-NSF-HiFiGAN）、World、VsLib。詳細は[アルゴリズム](#アルゴリズム)を参照。
- **相互運用性**：REAPER（`.rpp`）および VocalShifter（`.vshp` / `.vsp`）プロジェクトのインポート、REAPER / VocalShifter クリップボードとの双方向の読み書きに対応。MIDI は音高リファレンスブロック / 音高パラメータ / テンポマップとしてインポートでき、クリップとピッチラインは MIDI としてエクスポートできます。
- **インポート / エクスポート**：一般的なオーディオ / 動画フォーマットをインポート（動画からは自動で音声トラックを抽出）。プロジェクトと分離トラックを `wav` / `mp3` / `flac` でエクスポート。
- **その他**：内蔵ファイルブラウザーとクイック検索、ノート、自動バックアップ、多言語 UI（简体中文 / 繁體中文 / English / 日本語 / 한국어）、ライト / ダークテーマ、推論デバイスの選択とベンチマーク。

## インストール

[Releases](https://github.com/ARounder-183/HiFiShifter/releases) ページから、お使いの OS とアーキテクチャに対応したインストーラーをダウンロードしてください：

- **Windows**：NSIS インストーラー（`installer`）またはポータブル版 ZIP（`portable`）。x86_64 と arm64 の両アーキテクチャを提供しています。
- **macOS**：未署名の dmg（Apple Silicon は `arm64`、Intel は `x86_64`）。初回起動時は手動で許可する必要があります。「ファイルが破損しています」と表示された場合は、[ユーザーマニュアル](USERMANUAL_ja.md#一インストール)の手順に従って対処してください。
- **Linux**：AppImage（x86_64 / arm64）。

GPU アクセラレーション：Windows では DirectML（DirectX 12）、macOS（Apple Silicon）では CoreML + WebGPU、Linux x86_64 では WebGPU（Dawn/Vulkan）を使用し、その他のプラットフォームでは CPU にフォールバックします。アプリ内の `オプション → 推論デバイス` からデバイスを切り替えて、ベンチマークを実行できます。

## 基本原則

HiFiShifterはUTAUと同様のオフラインレンダリング方式を使用し、タイムライン上の各オーディオクリップを処理、レンダリング、キャッシュしてから再生システムに送り込むため、短いクリップの処理効率が高くなっています。

HiFiShifterは統一されたレンダリングインターフェースを提供し、将来的にアルゴリズムを追加しやすくしています。

## 推奨ワークフロー

推奨するワークフローは以下の通りです：

1. 他のDAWやスライシングソフトウェアを使用して、人力ボーカルに必要な短いクリップソースを準備する。
2. HiFiShifterでオーディオのスプライシングとチューニングを完了する。

HiFiShifterは他のソフトウェアからのプロジェクト移行を容易にする以下の操作もサポートしています：

1. VocalShifterプロジェクトを直接開く。
2. Reaperプロジェクトを直接開く。
3. VocalShifterクリップボードの内容を解析し、VocalShifterのパラメータをHiFiShifterのパラメータ領域に貼り付ける。
4. Reaperクリップボードの内容を解析し、ReaperのアイテムをHiFiShifterに直接貼り付ける。

## UI とアルゴリズム

### レイアウト

HiFiShifter は大きく分けて上部のトラックパネルと下部のパラメータパネルで構成されます。トラックパネルはクリップの編集と配置を担当し、パラメータパネルはオーディオへのパラメータ調整を担当します。各パネルはドッキング・フロート・再配置が可能です（`表示 → ウィンドウ / レイアウト`）。

トラックはネストに対応しています。あるトラックを別のトラックの下にドラッグすると、トラックグループ（ルートトラック + 子トラック）を作成できます。トラックグループは 1 つのアルゴリズムと 1 組のパラメータラインを共有し、パラメータラインは位置に応じてグループ内のすべてのクリップに適用されます。パラメータ調整の前に、トラックのコンポーズボタン `C` を押しておく必要があります。

### アルゴリズム

現在、HiFiShifter は 3 種類のアルゴリズムで処理をサポートしています：

- **World**：定評あるボコーダー。`ピッチ`、`ボリューム`、`ダイナミクス`、`パン` パラメータの編集に対応。
- **PC-NSF-HiFiGAN**（アルゴリズムリストでは `nsf-hifigan` と表示）：OpenVPI がオープンソースで公開している歌声特化の HiFi-GAN ボコーダーで、デフォルトのアルゴリズムです。`ピッチ`、`フォルマントシフト`、`ブレスゲイン`、`テンション`、`ボリューム`、`ダイナミクス`、`パン` パラメータの編集に対応。このうち `ブレスゲイン` と `テンション` は `ハーモニック分離` スイッチに依存します。スイッチをオンにすると、クリップを倍音とノイズの 2 つの成分に分離し（hnsep モデルを使用）、**レンダリングコストが増加します**（初回は各クリップの分離が必要なため、時間がかかることがあります）。オフの場合は分離を一切行わず、これら 2 つのパラメータはグレーアウトされ、カーブは表示されたまま編集できず、合成にも関与しません。
- **VsLib**（アルゴリズムリストでは `vslib` と表示）：VocalShifter 公式のアルゴリズムライブラリ。`ピッチ`、`フォルマントシフト`、`ブレッシネス`、`ボリューム`、`ダイナミクス`、`パン`、`合成モード` パラメータの編集に対応。**Windows x86_64 版のみ使用可能**です。公式の dll はファイル I/O のみに対応しているため、VocalShifter 本体と比較して処理に時間がかかります。

トラックパネルとパラメータパネルの詳細な操作（フェード編集、スナップ設定、ビブラートプリセット、エクスポートや録音など）については、[ユーザーマニュアル](USERMANUAL_ja.md)を参照してください。

## よく使うショートカット

> ショートカットはアプリ内の `オプション → キーボードショートカット...` でカスタマイズできます。以下の表はデフォルト値です。macOS では `Ctrl` が `⌘` に、`Alt` が `⌥` に対応します。

| 操作                                         | ショートカット / マウス                                       |
| :------------------------------------------- | :------------------------------------------------------------ |
| 再生 / 一時停止（再生開始位置に戻らない）    | `Space`                                                       |
| 再生 / 停止（再生開始位置に戻る）            | `Enter`                                                       |
| メトロノームの切り替え                       | `K`                                                           |
| ビューのパン                                 | マウス中ボタンのドラッグ                                      |
| タイムラインのスクロール                     | マウスホイール（2 軸自由スクロール）                          |
| 横 / 縦スクロール                            | `Shift` / `Alt` + マウスホイール                              |
| トラック高さのズーム                         | `Ctrl` + マウスホイール                                       |
| 元に戻す / やり直す                          | `Ctrl + Z` / `Ctrl + Shift + Z`（または `Ctrl + Y`）          |
| 切り取り / コピー / 貼り付け                 | `Ctrl + X` / `Ctrl + C` / `Ctrl + V`                          |
| 選択クリップを削除                           | `Delete`                                                      |
| クリップを分割（再生ヘッド位置）             | `S`                                                           |
| グループ化 / グループ解除                    | `G` / `U`                                                     |
| テイクを循環切り替え                         | `T`（`Shift + T` で前のテイク）                               |
| 範囲選択                                     | タイムラインの空き領域で右ボタンを押しながらドラッグ          |
| コピードラッグ                               | `Ctrl` を押しながらクリップをドラッグ                         |
| ストレッチ / スリップ編集                    | `Alt` を押しながらクリップの端 / 本体をドラッグ               |
| スナップの一時切り替え                       | `Shift` を押しながらドラッグ                                  |
| 新規プロジェクト / プロジェクトを開く        | `Ctrl + N` / `Ctrl + Shift + O`                               |
| メディアファイルをインポート / オーディオをエクスポート | `Ctrl + O` / `Ctrl + E`                            |
| 保存 / 名前を付けて保存                      | `Ctrl + S` / `Ctrl + Shift + S`                               |
| トラックを追加 / 録音                        | `Ctrl + T` / `Ctrl + R`                                       |
| モード切替（選択 ↔ 描画系ツール）            | `Tab`                                                         |
| クイック検索                                 | `Ctrl + F`                                                    |
| パラメータライン全体を上 / 下へ移動          | `=` / `-`（選択範囲は `[` / `]`）、`Shift` で大幅、`Ctrl` で微調整 |
| パラメータエディターの複数選択               | `Ctrl` を押しながらドラッグで選択範囲を追加、既存の選択範囲をクリックで解除 |

完全なショートカットと修飾キーの一覧は、[ユーザーマニュアル](USERMANUAL_ja.md#三トラックインターフェース)またはアプリ内のショートカット設定を参照してください。

## 開発

このセクションは開発者向けです。一般ユーザーはスキップしてください。

### 1. リポジトリのクローン

```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HiFiShifter
```

### 2. 依存関係のインストール

#### Windows

以下のツールがインストールされていることを確認してください：

- **Node.js**（推奨 20+、CI では 24）および npm
- **Rust ツールチェーン**（`rust-toolchain.toml` を参照）
- **Tauri 2 CLI**：`cargo install tauri-cli --version "^2"`
- **CMake**（SoundTouch ライブラリのビルドに必要）

ONNX Runtime (DirectML) は ort crate がビルド時に自動的にダウンロードします。追加設定は不要です。

`scripts/install_deps_windows.ps1` を実行することもできます。Chocolatey で NSIS / CMake / LLVM をインストールし、tauri-cli を導入します（Chocolatey が事前にインストールされている必要があります）。

フロントエンドの依存関係をインストールします：

```bash
npm --prefix frontend install
```

#### macOS

```bash
chmod +x ./scripts/install_deps_macos.sh
SKIP_FRONTEND=0 bash ./scripts/install_deps_macos.sh
```

#### Linux

以下のツールがインストールされていることを確認してください：

- **Node.js**（推奨 20+、CI では 24）と npm
- **Rust ツールチェーン**（`rust-toolchain.toml` を参照 — プロジェクトが自動的にプラットフォームに対応した stable ツールチェーンを選択します）
- **Tauri 2 CLI**: `cargo install tauri-cli --version "^2"`
- **CMake**、**pkg-config**、およびシステムビルドツール
- **GTK3、WebKit2GTK、ALSA** などの Tauri ランタイム開発ライブラリ（下記のインストールスクリプトを参照）

ワンクリックインストールスクリプトを実行：

```bash
chmod +x ./scripts/install_deps_linux.sh
bash ./scripts/install_deps_linux.sh
```

このスクリプトはシステム依存関係、Node.js（存在しない場合）、appimagetool、およびフロントエンドの npm 依存関係をインストールします。

フロントエンド依存関係のインストール（スクリプトを使用しない場合）：

```bash
npm --prefix frontend ci
```

### 3. サードパーティのソースコード

SoundTouch、WORLD、Signalsmith Stretch はいずれもコンパイル時にソースからビルドされ、初回ビルド時に**自動クローン**されます。手動操作は不要です。

オフラインビルドの場合は、事前に手動でクローンしておくこともできます。例（SoundTouch）：

```bash
cd backend/src-tauri/third_party/soundtouch-static
git clone --depth 1 --branch 2.3.3 https://codeberg.org/soundtouch/soundtouch.git soundtouch
```

### 4. 開発とビルド

```bash
# 開発モード（ホットリロード）
cd backend
cargo tauri dev

# リリースビルド
# Windows / macOS（デフォルト機能：onnx + vslib）
cargo tauri build

# Linux AppImage（vslib は Windows 専用のため、デフォルト機能を除外）
cargo tauri build --bundles appimage -- --no-default-features --features onnx
# 補助スクリプトも利用可: bash scripts/build-linux-appimage.sh

# Windows ポータブル ZIP
.\scripts\pack-portable.ps1 -SkipBuild
```

`TAURI_UI_MODE` 環境変数でフロントエンドの起動モードを切り替えられます：

- `dev`：開発モード（デフォルト。Vite dev server を使用し、ホットリロード対応）
- `build`：ビルドモード（フロントエンドの静的アセットを先にビルドしてから起動）

Linux/macOS（bash/zsh）：

```bash
cd backend
TAURI_UI_MODE=build cargo tauri dev
```

Windows PowerShell：

```powershell
cd backend
$env:TAURI_UI_MODE='build'; cargo tauri dev
```

**注意：** 初回コンパイルには非常に長い時間がかかります。しばらくお待ちください。

### 5. GPU アクセラレーション

HiFiShifterは、サポートされているプラットフォームで自動的にGPU推論アクセラレーションを有効にします。メニューバーの**推論デバイス（Inference Device）**から Auto / CPU / GPU を選択でき、**ベンチマークを実行（Run Benchmark）** で各デバイスの推論遅延を比較できます。

| プラットフォーム            | GPU技術                      | 説明                                                                           |
| --------------------------- | ---------------------------- | ------------------------------------------------------------------------------ |
| Windows x86_64 / ARM64      | DirectML (DirectX 12)        | 成熟した安定GPUパス、NVIDIA / AMD / Intel Arcに対応                            |
| macOS ARM64 (Apple Silicon) | CoreML + WebGPU (Dawn/Metal) | CoreMLはApple Neural Engineを活用；WebGPUは補助GPUバックエンドとして利用可能   |
| macOS x86_64 (Intel)        | —                            | CPUのみ（ort-tract代替バックエンドを使用）                                     |
| Linux x86_64                | WebGPU (Dawn/Vulkan)         | DawnがVulkan APIを通じてGPUにアクセス；GPUがない場合はCPUにフォールバック      |
| Linux ARM64                 | —                            | CPUのみ（このターゲット向けのWebGPU ONNX Runtimeプリビルドバイナリがないため） |

> **注意**：WindowsではWebGPUは無効です。Dawn/D3D12バックエンドが一部のGPU/ドライバの組み合わせでネイティブクラッシュを引き起こす可能性があります。DirectMLはWindows向けの成熟した安定GPUパスです。

ONNX Runtimeのバイナリは、ort crateの `download-binaries` 機能によりビルド時に自動的にダウンロードされます。手動設定は不要です。GPUプロバイダ（DirectML / WebGPU / CoreML）は、各ターゲットプラットフォーム向けにコンパイル時に自動的に有効化されます。追加の `--features` フラグは不要です。

**WSL2 ユーザー**：

- WSL2 は Linux サブ環境にハードウェア Vulkan を公開しません。WebGPU/Dawn は Lavapipe（CPU ソフトウェアレンダリング）しか使用できず、極めて低速です。GPU アクセラレーションが必要な場合は、Windows ネイティブ版（DirectML）を使用してください。
- FUSE サポートがないため、Tauri bundler の linuxdeploy ステップが失敗することがあります（エラー：`failed to run linuxdeploy`）。これは WSL2 の既知の制限であり、実際の AppImage 出力には影響しません — AppDir は `target/release/bundle/appimage/` に正しく組み立てられます。`APPIMAGE_EXTRACT_AND_RUN=1` を設定してから手動で `appimagetool` を実行してパッケージ化できます。実際の Linux マシンや CI ではこの問題は発生しません。

## ログとトラブルシューティング

アプリは実行ログを自動的に OS 標準のログディレクトリへ書き込みます。コマンドライン引数は不要です：

| OS | ログディレクトリ |
| --- | --- |
| Windows | `%LOCALAPPDATA%\com.arounder.hifishifter\logs` |
| macOS | `~/Library/Logs/com.arounder.hifishifter` |
| Linux | `~/.local/share/com.arounder.hifishifter/logs` |

- アプリ内の **ヘルプ → ログフォルダーを開く** でログの場所を直接開けます。**ヘルプ → 診断情報をエクスポート...** では、システム情報・全ログ・推論デバイスのベンチマーク結果を含む診断パッケージをワンクリックで生成できます。issue 報告時に添付してください。
- ログはサイズに応じて自動ローテーションされます：1 ファイル上限 8 MiB、履歴は 3 世代まで保持（`hifishifter.1.log` ~ `hifishifter.3.log`）。
  - 繰り返し発生するエラー / 警告は自動的にスロットリングされます：同じ位置のログは 10 秒ごとに最大 1 件出力され、抑制された件数は次の出力前に `[throttled]` 行で補記されます。
- フロントエンドとバックエンドのエラーは同じログファイルに記録されるため、1 つのファイルで時系列に沿って調査できます。

詳細オプション：

- 起動引数 `--log-file=<path>`：ログを指定パスに書き込みます。`--log-file=-` でファイルログを明示的に無効化。
- 環境変数 `HIFISHIFTER_LOG=debug`（`trace` / `info` / `warn` / `error`）：ログの詳細度を調整。
- 環境変数 `HIFISHIFTER_LOG_DIR`：既定のログディレクトリを上書き。
- ターミナルで `HiFiShifter --benchmark` を実行：推論デバイスのベンチマークを実行し、JSON 結果を出力します。

## ドキュメント

- ユーザーマニュアル：[简体中文](../../docs/i18n/USERMANUAL.md) · [繁體中文](USERMANUAL_zh-TW.md) · [English](USERMANUAL_en.md) · [日本語](USERMANUAL_ja.md) · [한국어](USERMANUAL_ko.md)
- 拡張（Extension）API：[docs/extension-api.md](../../docs/extension-api.md)
- i18n スタイルガイド：[docs/i18n/style-guide.md](../../docs/i18n/style-guide.md)

## 謝辞

このプロジェクトは以下のオープンソースライブラリのコードやモデルアーキテクチャを使用しています：

- [WORLD](https://github.com/mmorise/World) - 高品質な音声分析・合成システム
- [SoundTouch](https://www.surina.net/soundtouch/) - オーディオタイムストレッチ・ピッチシフトライブラリ（LGPL）
- [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) - 高品質なオーディオタイムストレッチライブラリ（MIT）
- [VocalShifter Library (vslib)](https://ackiesound.ifdef.jp/) - 音声解析・合成ライブラリ
- [SingingVocoders](https://github.com/openvpi/SingingVocoders) - 歌声合成ボコーダー（OpenVPI）
- [HiFi-GAN](https://github.com/jik876/hifi-gan) - 高忠実度GANボコーダー
- [vocal-remover (hnsep)](https://github.com/stakira/vocal-remover) - 調波 / ノイズ分離モデル（ブレス分離に使用）

## ライセンス

このプロジェクトは [MITライセンス](../../LICENSE) の下で公開されています。
