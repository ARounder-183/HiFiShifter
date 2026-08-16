# HiFiShifter

[简体中文](../../README.md) | [繁體中文](README_zh-TW.md) | [English](README_en.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

HiFiShifterは、グラフィカルなボーカル編集・合成ツールです。マルチトラックのオーディオクリップ処理をサポートし、トラックグループ単位で複数のボコーダーを使用してボーカルのピッチ補正やパラメータ調整を行い、人力VOCALOID制作における編集と調声を一体化します。

**このプロジェクトはまだ開発中です。全体的なテストは完了しておらず、多くのバグや不安定な問題が存在する可能性があります。**

![プレビュー](../preview.png)

## インストール

リポジトリのサイドバーから、お使いのシステムに適したリリースバージョンをダウンロードしてインストールしてください。

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

## 機能紹介

### レイアウト

HiFiShifterは大きく分けて2つの機能エリアに分かれています。上部のトラックパネルと下部のパラメータパネルです。トラックパネルは主にオーディオクリップの処理を担当し、パラメータパネルはパラメータ調整を担当します。

### トラックパネル

HiFiShifterは、ほとんどの現代的なDAWと同様に、かなり完全なトラックパネルとオーディオクリップ編集機能を提供します。

#### メディアのインポート（音声 / 動画）

HiFiShifterは3つの方法でメディアファイルをインポートできます。動画ファイルは自動的に音声トラックを使用します：

1. システムのファイルマネージャーからトラックに音声または動画ファイルを直接ドラッグ＆ドロップする。
2. ツールバーのフォルダアイコンをクリックして内蔵ファイルブラウザを開き、メディアファイルをトラックにドラッグする。
3. `Ctrl + F` を押してクイック検索を開き、メディアファイルを選択してトラックにインポートする（クイック検索のファイルパスは内蔵ファイルブラウザの現在のパスと同じ）。

#### オーディオ編集

- **グリッドにスナップ**：クリップの移動/トリミングはデフォルトでグリッドにスナップします。`Shift` を押すと一時的にスナップをオフにできます。
- **トリミング/ストレッチ範囲**：クリップの左右の端をドラッグしてトリミングまたは延長します。
- **タイムストレッチ**：`Alt` + 左マウスボタンを押しながらクリップの左右の端をドラッグすると、オーディオをストレッチできます。
- **スリップ編集**：`Alt` + 左マウスボタンを押しながらクリップの本体をドラッグすると、内部コンテンツを左右にスライドできます。
- **フェードイン/アウト**：クリップの左上/右上の角をドラッグしてフェードイン/アウトの長さを調整します。
- **ゲイン（dB）**：クリップ左上のノブを上下にドラッグしてゲインを調整します。現在のdBは右上に表示されます。
- **クリップミュート（M）**：クリップ左上の `M` ボタンをクリックしてミュートします。ミュートするとクリップはグレー表示になります。
- **マーキー選択**：タイムラインの空き領域で右マウスボタンを押しながらドラッグすると、複数のクリップを選択できます。
- **コピードラッグ**：`Ctrl` を押しながらクリップをドラッグすると、ターゲット位置にコピーを作成します（元のクリップはそのまま；コピーはドロップ時に有効）。
- **グルー**：クリップを右クリックしてメニューから「グルー」を選択します（同じトラックに少なくとも2つのクリップが必要）。
- **分割**：クリップを選択して `S` を押すと、再生ヘッドの位置で分割します。
- **コピー/貼り付け**：クリップを選択して `Ctrl + C` を押すと、アプリケーションクリップボードにコピーします。`Ctrl + V` は、選択したクリップの最も左の開始位置を再生ヘッド位置に合わせ、他のクリップの相対的な間隔を維持します。

トラックはネストをサポートしていることに注意してください。あるトラックを別のトラックの下にドラッグして子トラックにし、トラックグループを形成できます。これはその後のパラメータ調整で非常に役立ちます。

### パラメータパネル

HiFiShifterのパラメータパネルは、VocalShifterと同様の操作をサポートしており、パラメータ調整を容易に行えます。

各トラックには特別な `C` ボタンがあることに注意してください。このボタンが押されているトラックのオーディオだけが、その後のパラメータ調整の対象となります。

パラメータ調整では、HiFiShifterはトラックグループを単位として動作します。ルートトラックの `C` ボタンがグループ全体のアルゴリズムとパラメータカーブを決定します。パラメータカーブは各オーディオクリップの位置に基づいて適用されます。

各アルゴリズムには異なる調整可能なパラメータがあります。共通のパラメータはピッチです。

初回起動時、HiFiShifterはクリップのピッチ分析に時間がかかります。分析後、パネルの実線はグループの現在の全体ピッチを、破線は元の全体ピッチを、色付きの線は各クリップの元のピッチを表します。

他のパラメータパネルはピッチパネルと似ていますが、個々のクリップの元のピッチは表示されません。

パネルの横にある小さな目のアイコンは、非選択時のパネルの可視性を切り替えます。

### アルゴリズム

HiFiShifterは現在3つのアルゴリズムをサポートしています。

#### Worldアルゴリズム

定評あるボコーダー。  
`ピッチ` 編集のみをサポート。

#### PC-NSF-HiFiGAN

OpenVPIのオープンソースの歌声特化型hifiganボコーダー。  
`ピッチ`、`ブレス`、`テンション`、`フォルマントシフト`、`ボリューム` の編集をサポート。  
ブレス編集は追加の有効化が必要で、hnsep UVRモデルを使用してブレス分離を行います。初回使用時は長い時間がかかることがあります。テンションを編集する場合は、必ずブレスを有効にしてください。

#### Vslib

VocalShifterが提供するアルゴリズムライブラリ。  
`ピッチ`、`パン`、`フォルマントシフト`、`ボリューム`、`ブレス` の編集をサポート。  
公式DLLはファイルI/Oのみをサポートしているため、VocalShifter本体と比較して処理に時間がかかります。

## よく使うショートカットキー

| 操作                                         | ショートカット / マウス                     |
| :------------------------------------------- | :------------------------------------------ |
| ビューパン（タイムライン）                   | マウス中ボタンドラッグ                      |
| 水平ズーム（タイムライン）                   | マウスホイール（カーソル中心）              |
| 垂直ズーム（トラック高）                     | Ctrl + マウスホイール                       |
| 垂直ズーム（パラメータ軸）                   | Ctrl + マウスホイール（パラメータパネル内） |
| 再生 / 一時停止                              | スペース                                    |
| 再生 / 停止                                  | Enter                                       |
| 元に戻す / やり直す                          | Ctrl + Z / Ctrl + Y                         |
| 新規プロジェクト                             | Ctrl + N                                    |
| プロジェクトを開く                           | Ctrl + Shift + O                            |
| 保存                                         | Ctrl + S                                    |
| 名前を付けて保存                             | Ctrl + Shift + S                            |
| オーディオをエクスポート                     | Ctrl + E                                    |
| モード切替（選択/描画）                      | Tab                                         |
| 選択クリップを削除                           | Delete                                      |
| 選択クリップをコピー（アプリクリップボード） | Ctrl + C                                    |
| 再生ヘッドに貼り付け                         | Ctrl + V                                    |
| 選択範囲カーブをコピー（パラメータ）         | Ctrl + C（選択モード）                      |
| 選択範囲の先頭に貼り付け                     | Ctrl + V（選択モード）                      |
| クリップを分割                               | S（再生ヘッド位置で選択クリップを分割）     |
| 新規トラック                                 | Ctrl + T                                    |
| クイック検索                                 | Ctrl + F                                    |

## 開発環境セットアップ

このセクションは開発者向けです。一般ユーザーはスキップしてください。

### 1. リポジトリのクローン

```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HiFiShifter
```

### 2. 依存関係のインストール

#### Windows

以下のツールがインストールされていることを確認してください：

- **Node.js**（推奨18+）および npm
- **Rustツールチェーン**（`rust-toolchain.toml` を参照）
- **Tauri 2 CLI**：`cargo install tauri-cli --version "^2"`
- **CMake**（SoundTouchライブラリのビルドに必要）

ONNX Runtime (DirectML) は ort crate がビルド時に自動的にダウンロードします。追加設定は不要です。

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

- **Node.js**（推奨 20+）と npm
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

#### Linux AppImage ビルド

`vslib` アルゴリズムは Windows 専用のため、Linux ビルドではデフォルト機能を無効にする必要があります：

```bash
# backend/ ディレクトリから実行（tauri.conf.json のパスはこのディレクトリからの相対パスです）
cd backend
cargo tauri build --bundles appimage -- --no-default-features --features onnx
```

または提供されているヘルパースクリプトを使用：

```bash
bash scripts/build-linux-appimage.sh
```

> **注意：** WSL2 環境では FUSE サポートがないため、Tauri bundler の linuxdeploy ステップが失敗する場合があります（エラー：`failed to run linuxdeploy`）。これは WSL2 の既知の制限であり、実際の AppImage 出力には影響しません — AppDir は `target/release/bundle/appimage/` に正しく構築されます。`APPIMAGE_EXTRACT_AND_RUN=1` を設定して手動で `appimagetool` を実行してパッケージ化してください。この問題は実際の Linux マシンや CI では発生しません。

### 3. SoundTouch ソース

SoundTouch オーディオタイムストレッチライブラリはコンパイル時にソースからビルドされます。初回ビルド時に**自動クローン**されるため、手動操作は不要です。

オフラインビルド用に、事前に手動でクローンすることも可能です：

```bash
cd backend/src-tauri/third_party/soundtouch-static
git clone --depth 1 --branch 2.3.3 https://codeberg.org/soundtouch/soundtouch.git soundtouch
```

### 4. GPUアクセラレーション

HiFiShifterは、サポートされているプラットフォームで自動的にGPU推論アクセラレーションを有効にします。メニューバーの**推論デバイス（Inference Device）**から Auto / CPU / GPU を選択でき、**ベンチマークを実行（Run Benchmark）** で各デバイスの推論遅延を比較できます。

| プラットフォーム                | GPU技術                               | 説明                                                                           |
| ------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------ |
| Windows x86_64 / ARM64          | DirectML (DirectX 12)                 | 成熟した安定GPUパス、NVIDIA / AMD / Intel Arcに対応                            |
| macOS ARM64 (Apple Silicon)     | CoreML + WebGPU (Dawn/Metal)          | CoreMLはApple Neural Engineを活用；WebGPUは補助GPUバックエンドとして利用可能   |
| macOS x86_64 (Intel)            | —                                     | CPUのみ（ort-tract代替バックエンドを使用）                                     |
| Linux x86_64                    | WebGPU (Dawn/Vulkan)                  | DawnがVulkan APIを通じてGPUにアクセス；GPUがない場合はCPUにフォールバック      |
| Linux ARM64                     | —                                     | CPUのみ（このターゲット向けのWebGPU ONNX Runtimeプリビルドバイナリがないため） |

> **注意**：WindowsではWebGPUは無効です。Dawn/D3D12バックエンドが一部のGPU/ドライバの組み合わせでネイティブクラッシュを引き起こす可能性があります。DirectMLはWindows向けの成熟した安定GPUパスです。
>
> **WSL2ユーザー**：WSL2はLinuxサブ環境にハードウェアVulkanを公開しません。WebGPU/DawnはLavapipe（CPUソフトウェアレンダリング）しか使用できず、非常に低速です。WSL2でGPUアクセラレーションが必要な場合は、WindowsネイティブビルドのDirectMLを使用してください。

#### 全プラットフォーム

ONNX Runtimeのバイナリは、ort crateの `download-binaries` 機能によりビルド時に自動的にダウンロードされます。手動設定は不要です。GPUプロバイダ（DirectML / WebGPU / CoreML）は、各ターゲットプラットフォーム向けにコンパイル時に自動的に有効化されます。追加の `--features` フラグは不要です。

```bash
# 開発モード（ホットリロード）
cd backend
cargo tauri dev

# リリースビルド
# Windows / macOS（デフォルト機能：onnx + vslib）
cargo tauri build

# Linux（vslibはWindows専用のため、デフォルト機能を除外）
cargo tauri build --bundles appimage -- --no-default-features --features onnx

# Windows ポータブルZIP
.\scripts\pack-portable.ps1 -SkipBuild
```

## クイックスタート

### 開発モードの実行

```bash
cd backend/src-tauri
cargo tauri dev
```

`TAURI_UI_MODE` 環境変数でフロントエンドの起動モードを切り替えできます：

- `dev`：開発モード（デフォルト、Vite dev server を使用しホットリロード対応）
- `build`：ビルドモード（フロントエンド静的アセットを先にビルドしてから起動）

Linux/macOS（bash/zsh）：

```bash
cd backend/src-tauri
TAURI_UI_MODE=build cargo tauri dev
```

Windows PowerShell：

```powershell
cd backend/src-tauri
$env:TAURI_UI_MODE='build'; cargo tauri dev
```

**注意：** 初回コンパイルには非常に長い時間がかかります。しばらくお待ちください。

## ドキュメント

- [ユーザーマニュアル](USERMANUAL_ja.md)
- [Todoリスト](../../todo.md)

## 謝辞

このプロジェクトは以下のオープンソースライブラリのコードやモデルアーキテクチャを使用しています：

- [WORLD](https://github.com/mmorise/World) - 高品質な音声分析・合成システム
- [SoundTouch](https://www.surina.net/soundtouch/) - オーディオタイムストレッチ・ピッチシフトライブラリ（LGPL）
- [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) - 高品質なオーディオタイムストレッチライブラリ（MIT）
- [VocalShifter Library (vslib)](https://ackiesound.ifdef.jp/) - 音声解析・合成ライブラリ
- [SingingVocoders](https://github.com/openvpi/SingingVocoders) - 歌声合成ボコーダー（OpenVPI）
- [HiFi-GAN](https://github.com/jik876/hifi-gan) - 高忠実度GANボコーダー

## ライセンス

このプロジェクトは [MITライセンス](../../LICENSE) の下で公開されています。
