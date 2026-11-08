# Third-Party Licenses

HiFiShifter ships third-party libraries, SDK headers and model weights that keep
their own licenses. This directory carries the **full texts** and the
component-specific notices that must travel with the binaries.

The project's own source code is MIT-licensed: see [`../LICENSE`](../LICENSE),
which is also the authoritative component list (names, versions, upstream links).

| File | Covers |
| --- | --- |
| `COPYING.LGPL-2.1` | SoundTouch (`SoundTouchDLL`) - GNU Lesser General Public License 2.1 |
| `Apache-2.0.txt` | ARA SDK headers, `rusty_mp3`, `rusty_flac` - Apache License 2.0 |
| `NSF-HiGAN-vocoder-NOTICE.txt` | DiffSinger Community Vocoder weights - CC BY-NC-SA 4.0 |
| `GAME-NOTICE.txt` | GAME note segmentation ONNX weights - CC BY-NC-SA 4.0 |

CLAP is a header-only MIT SDK; its upstream license text travels in the pinned SDK checkout and the authoritative component list in `../LICENSE`.

## SoundTouch (LGPL-2.1) - dynamically linked, unmodified upstream

`SoundTouchDLL` is loaded at run time from a separate file next to the
executable or plug-in. It is built from the **unmodified** upstream source
(tag `2.3.3`, commit `e83424d5928ab8513d2d082779c275765dee31b9`): the build only
passes extra CMake options (an include directory for the Windows resource
compiler and a CMake policy floor), no file in the SoundTouch tree is edited. Because the library stays a separate, replaceable
file, LGPL-2.1 section 6(b) is satisfied by the dynamic link itself - that is why
no source archive is bundled.

To rebuild or replace it:

- Upstream: <https://codeberg.org/soundtouch/soundtouch> (mirror: <https://www.surina.net/soundtouch/>)
- Check out the pinned tag, configure with `-DSOUNDTOUCH_DLL=ON`, build the
  `SoundTouchDLL` target with CMake, then drop the resulting
  `SoundTouchDLL.dll` / `libSoundTouchDLL.so` / `libSoundTouchDLL.dylib` in
  place of the shipped one. No source change is required.

## NSF-HiFiGAN vocoder weights (CC BY-NC-SA 4.0)

The bundled vocoder weights are provided by the OpenVPI Team and are licensed
for **non-commercial** use only. `NSF-HiGAN-vocoder-NOTICE.txt` carries the
required attribution, the license link and the training-data acknowledgements.
Keep that file with any redistribution, and do not sell this software while it
contains those weights.

---

# 第三方许可（简体中文）

本目录只放**随发行物一同分发的第三方许可全文与声明**。项目自身的源代码以 MIT
许可发布，见 [`../LICENSE`](../LICENSE) —— 那里同时也是第三方组件的权威清单
（名称、版本、上游链接）。

- `COPYING.LGPL-2.1`：SoundTouch（`SoundTouchDLL`）的 LGPL-2.1 全文。
  SoundTouch 以**动态库**形式随程序分发，源码为**未修改的上游版本**（tag `2.3.3`）。
  DLL 是独立文件，你可以用任何接口兼容的自建版本替换它 —— LGPL-2.1 第 6(b) 条要求的
  正是这种可替换性，因此发行包不附带源码压缩包。
- `Apache-2.0.txt`：ARA SDK 头文件、`rusty_mp3`、`rusty_flac` 使用的
  Apache License 2.0 全文。
- `NSF-HiGAN-vocoder-NOTICE.txt`：DiffSinger 社区声码器权重的 **CC BY-NC-SA 4.0**
  声明（署名、许可链接、训练数据致谢）。该权重仅限**非商业使用**，再分发时请保留此文件。

# 第三方授權（繁體中文）

本目錄只放**隨發行物一同散佈的第三方授權全文與聲明**。專案自身的原始碼以 MIT
授權釋出，見 [`../LICENSE`](../LICENSE) —— 該檔同時是第三方元件的權威清單
（名稱、版本、上游連結）。

- `COPYING.LGPL-2.1`：SoundTouch（`SoundTouchDLL`）的 LGPL-2.1 全文。
  SoundTouch 以**動態函式庫**形式隨程式散佈，原始碼為**未修改的上游版本**（tag `2.3.3`）。
  DLL 是獨立檔案，你可以用任何介面相容的自建版本替換它 —— LGPL-2.1 第 6(b) 條要求的
  正是這種可替換性，因此發行包不附帶原始碼壓縮檔。
- `Apache-2.0.txt`：ARA SDK 標頭檔、`rusty_mp3`、`rusty_flac` 使用的
  Apache License 2.0 全文。
- `NSF-HiGAN-vocoder-NOTICE.txt`：DiffSinger 社群聲碼器權重的 **CC BY-NC-SA 4.0**
  聲明（姓名標示、授權連結、訓練資料致謝）。該權重僅限**非商業使用**，再散佈時請保留此檔案。

# サードパーティライセンス（日本語）

このディレクトリには、**配布物と一緒に配布する必要がある**サードパーティの
ライセンス全文と告知だけを置いています。プロジェクト自身のソースコードは MIT
ライセンスです（[`../LICENSE`](../LICENSE) を参照）。同ファイルがサードパーティ
コンポーネントの正式な一覧（名称・バージョン・上流リンク）でもあります。

- `COPYING.LGPL-2.1`：SoundTouch（`SoundTouchDLL`）の LGPL-2.1 全文。
  SoundTouch は**動的ライブラリ**として配布され、ソースは**未改変の上流版**（tag `2.3.3`）です。
  DLL は独立したファイルなので、インターフェース互換な自作ビルドに差し替えられます ——
  LGPL-2.1 第 6(b) 条が求めているのはこの差し替え可能性であり、ソースアーカイブを
  同梱していない理由でもあります。
- `Apache-2.0.txt`：ARA SDK ヘッダー、`rusty_mp3`、`rusty_flac` が使う
  Apache License 2.0 の全文。
- `NSF-HiGAN-vocoder-NOTICE.txt`：DiffSinger コミュニティ製ボコーダー重みの
  **CC BY-NC-SA 4.0** 告知（表示・ライセンスへのリンク・学習データの謝辞）。
  この重みは**非商用利用に限られ**ます。再配布時はこのファイルを残してください。

# 서드파티 라이선스（한국어）

이 디렉터리에는 **배포물과 함께 제공해야 하는** 서드파티 라이선스 전문과 고지만
들어 있습니다. 프로젝트 자체 소스 코드는 MIT 라이선스이며
[`../LICENSE`](../LICENSE) 를 참고하십시오. 같은 파일이 서드파티 구성 요소의
공식 목록(이름, 버전, 업스트림 링크)이기도 합니다.

- `COPYING.LGPL-2.1`：SoundTouch(`SoundTouchDLL`)의 LGPL-2.1 전문.
  SoundTouch 는 **동적 라이브러리**로 배포되며 소스는 **수정되지 않은 업스트림**(tag `2.3.3`)입니다.
  DLL 은 별도 파일이므로 인터페이스 호환 빌드로 교체할 수 있습니다 —— LGPL-2.1
  제6(b)조가 요구하는 것이 바로 이 교체 가능성이며, 소스 압축 파일을 동봉하지 않는 이유입니다.
- `Apache-2.0.txt`：ARA SDK 헤더, `rusty_mp3`, `rusty_flac` 이 사용하는
  Apache License 2.0 전문.
- `NSF-HiGAN-vocoder-NOTICE.txt`：DiffSinger 커뮤니티 보코더 가중치의
  **CC BY-NC-SA 4.0** 고지(저작자 표시, 라이선스 링크, 학습 데이터 감사의 글).
  이 가중치는 **비상업적 용도로만** 사용할 수 있습니다. 재배포 시 이 파일을 유지하십시오.
