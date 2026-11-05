# 模型权重：放在哪、怎么发

一句话：**清单进仓库，权重进 release**。仓库里只有
`backend/src-tauri/resources/models/models.json`（1.5 KB），151.6 MB 的权重以
`models-<version>.zip` 的形式挂在 tag `models-<version>` 下，按内容哈希取用。

这条线上只有三件事必须绝对正确，其余都是细节：

1. **版本号 = 内容哈希。** `models.json` 的 `version` 由
   `scripts/build/write-models-manifest.ps1` 从每个文件的 sha256 汇总得出（12 位）。没有人
   手写它，因此"模型换了、版本号忘了改"这件事不可能发生。
2. **资产的字节可复算。** 同一份权重在任何机器上打出来的 zip 逐字节相同 —— 所以它的
   sha256 才配当发布物的身份（`fetch-models.py --package` 保证这一点）。
3. **先发资产，后合并清单。** 顺序反了会让六平台一起红，见「[顺序](#顺序资产先上清单后合并)」。

## 谁写、谁读

| 位置 | 谁写 | 谁读 |
| --- | --- | --- |
| `backend/src-tauri/resources/models/`（工作树） | `scripts/setup/fetch-models.py` | 构建脚本、`write-models-manifest.ps1` |
| App / VST3 bundle 里的 `models/models.json` | `build.ps1`、`vst3/unix/build.sh`（只拷清单） | 运行时 `model_store`，用来判断共享库能不能用 |
| 共享模型库 `%PROGRAMDATA%\HiFiShifter\models\<version>`（仅 Windows） | 合并安装器（独占写） | App 与插件（只读） |
| macOS / Linux bundle 的 `Contents/Resources/models/` | 构建（自包含，权重随包） | 运行时直接读 |
| release 资产 `models-<version>.zip` | 人，按本文 | `fetch-models.py`、镜像 |

权重为什么不在仓库里、Windows 为什么有共享库而其它平台没有，见
[`ci.md`](ci.md) 与 `backend/hifishifter-kernel/src/model_store.rs` 的文件头。

## 发布一个新版本

换模型一年只有几次。下面七步按顺序做，每步都有可判读的输出。

### 0. 把权重放到位

```
backend/src-tauri/resources/models/
├── fcpe/fcpe.onnx
├── hnsep/hnsep.onnx
├── hnsep/config.yaml
├── nsf_hifigan/pc_nsf_hifigan.onnx
├── nsf_hifigan/config.json
└── nsf_hifigan/coreml_pads_patch.txt   ← scripts/build/generate-coreml-pads-patch.py 生成
```

放进去之后**就别再动**：任何一次字节改动都会得到一个新版本号。新增或替换权重时，
[`../licenses/README.md`](../licenses/README.md) 里对应的模型出处与许可也要跟着改 ——
许可是发行物的一部分。

### 1. 生成清单

```powershell
pwsh -File scripts/build/write-models-manifest.ps1
# wrote …/models.json (version d342aa989cee, 6 files)
```

版本号变了 = 真的换了权重；没变 = 你只是把同一份清单重新生成了一次。**不要手改
`models.json`**：它是唯一权威，而 `version` 是它唯一不能手写的字段。

### 2. 打包并自检

```powershell
python scripts/setup/fetch-models.py --package
# 资产 dist/models/models-d342aa989cee.zip（151.6 MB）
#   sha256 9750177077373869ebe0d4904b6b71674f82b032bb84ebd1ea30842c4042c6cc
python scripts/setup/fetch-models.py --verify-archive dist/models/models-d342aa989cee.zip
# models-d342aa989cee.zip 与清单一致（版本 d342aa989cee，6 个文件）
```

`--package` 在打包前做三件事：逐文件比对字节数与 sha256、重算版本号（手改过的清单
在这里当场失败）、把清单外的文件报出来（"新加了模型文件但忘了重生成清单"的典型症状）。
打包本身固定条目顺序、时间戳（1980-01-01）与存储方式（不压缩，因为 deflate 的字节流随
zlib 版本变化），所以同一份权重永远得到同一个 sha256：上传之后任何人重打一遍包就能对账。

顺带产出 `…zip.sha256` 边车（`sha256sum -c` 格式），给不装 Python 的下载方用。

### 3. 发 release

```powershell
gh release create models-d342aa989cee `
  dist/models/models-d342aa989cee.zip `
  dist/models/models-d342aa989cee.zip.sha256 `
  --title "Model weights d342aa989cee" `
  --target develop --prerelease --draft `
  --notes "models.json d342aa989cee，取自 <提交号>"
gh release edit models-d342aa989cee --draft=false
```

没有 `gh`（本机就没装）就在网页上做同样的事：Releases → Draft a new release → tag 填
`models-d342aa989cee`、target 选 `develop` → 附上两个文件 → 勾 **Set as a pre-release** →
Publish。网页上"新建 release"本来就是 draft，先别点 Publish 就对了。

三条硬约束，破一条就是 404：

- **tag 必须是 `models-<version>`，资产必须叫 `models-<version>.zip`。**
  `fetch-models.py` 直接按这两段拼地址
  （`<base>/models-<version>/models-<version>.zip`），改一个字符就取不到。
- **必须 `--prerelease`。** 模型发布不是产品发布：它不能出现在 `releases/latest`
  里，也不能长得像 `v*` tag —— `build.yml` 只认 `v*`，但真要有人把 tag 起成
  `v0.1.0-models`，六平台矩阵会被白白跑一遍，`assert-version` 还会红。
- **draft → 传资产 → 发布。** draft 期间 `releases/download/<tag>/…` 返回 404，所以
  "资产只传了一半"的窗口对下游不可见。

tag 指向哪个提交不重要 —— 它只是地址标签，资产才是载荷。把"取自哪个提交"写进 release
notes，比让 tag 指着某个提交有用。

### 4. 验线上资产

在你**有新版清单的工作树**里跑，证明"线上那份字节 = 本地这份字节"（上传截断、被同名文件
顶掉、CDN 还在发旧对象，都会在这里露出来）：

```powershell
curl -L -o "$env:TEMP\models.zip" `
  https://github.com/ARounder-183/HiFiShifter/releases/download/models-d342aa989cee/models-d342aa989cee.zip
python scripts/setup/fetch-models.py --verify-archive "$env:TEMP\models.zip"
# models-d342aa989cee.zip 与清单一致（版本 d342aa989cee，6 个文件）
```

`-L` 不能省：GitHub 的下载地址会 302 到对象存储。

### 5. 合并清单

顺序：**资产先上，清单后合并。** 上一步验的是字节，这一步赌的是顺序：清单一旦合并，
每个 PR 的六个平台都会去取这个资产；取不到就是六个 job 一起失败。所以宁可让清单的提交
晚一个小时。

合并后 CI 第一次会重新下载一遍权重 —— 缓存键是 `models.json` 的哈希，键变了就得重新
填，属正常。

### 6. 干净 clone 复验

这是唯一能证明"一台新机器从零到能用"的方法：它走的是别人会走的那条路（默认基地址 + 清单
里的版本号）。

```powershell
git clone --depth 1 https://github.com/ARounder-183/HiFiShifter.git "$env:TEMP\hfs-check"
cd "$env:TEMP\hfs-check"
python scripts/setup/fetch-models.py
# 下载 https://github.com/…/releases/download/models-d342aa989cee/models-d342aa989cee.zip
# 模型权重就绪（版本 d342aa989cee，6 个文件）
```

CI 替你跑同一件事的另一半：`build.yml` 的每个 job 都在编译前跑 `fetch-models.py`，
资产不在就是红的（并且是在矩阵上红，不是在某条 lint 上红）。

## 顺序：资产先上，清单后合并

```
权重就位 → 生成清单 → 打包 → 上传资产 → 验线上字节 → 合并清单 → 干净 clone 复验
```

反过来（先把清单合了）会造成两类后果，都被 CI 放大：

- `fetch-models.py` 退出码 1 → `build.yml` 的每个 job 都停在"取权重"这一步；
- 任何人 `git bisect` 到那个提交，都会拿到一棵"构建不了"的树。

## 不能做的事

| 反例 | 后果 |
| --- | --- |
| 手改 `models.json` | `--package` 拒绝打包（版本号与内容重算值不符）。绕过的话更糟：运行时 `looks_complete` 只比字节数，"看起来装了、推理全错" |
| 覆盖已发布 tag 的资产 | 同一个版本号对应两份字节。CI 的缓存键（`hashFiles(models.json)`）不变，会继续用旧权重；用户下到的也可能是旧的 |
| 用"压缩为 zip"生成资产 | 时间戳与条目顺序不定 → 每次 sha256 都不同，发布物失去身份 |
| 把权重提交回 git | checkout 从 150 MB 起步（一次 PR 十几次），且缓存键与内容无关 |
| 把 CoreML 变体打进资产 | Windows/Linux 白装 56 MB 用不上的权重；版本号还会取决于"这台机器编没编过 macOS 目标"（`--package` 因此把它列入已知构建产物） |
| 删除旧版本的 release | 老 App 与老的 CI 提交再也取不到对应版本。资产是内容寻址的，旧 tag 必须永久保留 |

## 镜像与离线

- **镜像**：设 `HIFISHIFTER_MODELS_BASE_URL`（或 `--base-url`），布局与官方一致：
  `<base>/models-<version>/models-<version>.zip`，且内容逐字节相同
  （用 `--verify-archive` 验）。
- **离线**：`--source <已有模型目录>` 或 `--archive <本地 zip>`，两者与下载路径
  共用同一套逐文件哈希校验。

## 故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `下载失败：HTTP Error 404` | tag / 资产名不对，或 release 还是 draft | `gh release view models-<v> --json assets`；draft 就先 publish |
| `模型权重校验失败：sha256 …` | 下载被截断、镜像是旧的，或手上是另一版权重 | 删掉本地这几个文件重取；与本文件第 2 步打印的 sha256 比对 |
| `源目录缺少 X` | `--source` 指到了 `models\` 的上一层 | 指向同时含 `fcpe/`、`hnsep/`、`nsf_hifigan/` 的目录 |
| `清单的 version … 与内容重算值 … 不符` | 手改过清单，或改了权重没重生成 | `pwsh -File scripts/build/write-models-manifest.ps1` |
| `警告：X 不在清单里，不会进资产` | 新加了模型文件但没重生成清单 | 同上，确认后重打包 |
| `build.yml` 在"Fetch model weights"红 | 清单已合并但资产没发（顺序反了） | 补发对应 tag 的资产，重跑 job |

## 发布前自检

- [ ] 权重是最终字节，`write-models-manifest.ps1` 跑过（版本号符合预期）
- [ ] `--package` 与 `--verify-archive` 都通过
- [ ] release 是 pre-release；tag 与资产名都是 `models-<version>`
- [ ] 线上资产下载回来 `--verify-archive` 通过
- [ ] 之后才合并 `models.json`
- [ ] 合并后干净 clone 里 `fetch-models.py` 通过哈希校验
