# `probe/` — 宿主实测取证

这个目录保存**在真实宿主里量出来的数据**，用来支撑那些无法从官方文档推出的常数。
仓库里多处代码/注释引用本目录下的文件；**引用必须指向真实存在的证据**，否则那条
常数就退化成"常数等于自己"的变更探测器。

## 为什么需要它

官方头文件只说"两套轴互补"，不公开 fade 求值函数，也不说七个预设各自对应哪一组
`(curvature, S)`。`backend/hifishifter-kernel/src/fade_axes.rs` 里的
`HOST_FADE_PRESET_AXES` 因此是**量出来的**，不是抄文档的。同样，倒放 take 的
`D_STARTOFFS` 是镜像坐标这一点也是实测结论（F-4）。

## 当前状态（诚实记录）

**本目录在被引入时是空的。** 代码与计划里引用过下列文件，它们**不在工作树里**：

| 被引用的路径 | 引用点 | 状态 |
|---|---|---|
| `probe/ara/captures/fade-axis-7.82.json` | `fade_axes.rs`、`hostFadeAxes.ts` 的模块文档 | **缺失** |
| `probe/ara/FADE-AXIS-FINDINGS.md` | 同上、`host_edit.rs` | **缺失** |
| `probe/ara/REVERSE-FINDINGS.md` | `editor/workspace.rs` 的镜像复原注释 | **缺失** |
| `probe/ara/MULTI-TAKE-FINDINGS.md` | `ara/model.rs` | **缺失** |

后果：`HOST_FADE_PRESET_AXES` 与倒放镜像约定目前**无法在本仓复核**。相关测试只钉住
"常数没有变"，不构成正确性证明。

## 必须补的取证

### 1. 淡变曲面（`fade-surface`）

**目的**：把 `(curvature, S) → gain` 从"HFS 自己的视觉定义"升级成"与宿主一致"。

**做法**：在 `c, S ∈ {-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1}` 的网格上，
让宿主渲染一段淡变并读回 PCM；据此拟合 `fade_curves.rs` / `reaperFade.ts` 里
`curvature` 与 `S` 各自的锚点间插值。

**产物**：
- `probe/ara/captures/fade-surface-<host-version>.json`（原始采样，**提交进仓**）
- `probe/ara/FADE-SURFACE-FINDINGS.md`（结论 + 与当前实现的差异）

### 2. 七个预设的轴坐标（`fade-axis`）

**做法**：只写 `C_FADEINSHAPE = k`（k = 0..6），读回 `D_FADEINDIR_NEW` /
`D_FADEINDIR2_NEW`；另取一个 baseline（全新 item 的默认读数）。

**产物**：`probe/ara/captures/fade-axis-<host-version>.json` + 结论文档。
**这是现有 `HOST_FADE_PRESET_AXES` 的出处**，必须把原始数据补回来。

### 3. 倒放坐标约定（`reverse`）

**做法**：源长已知的素材、非对称裁切、倒放后读 `D_STARTOFFS`，验证
`镜像起点 = D − 正向终点`；再分割一次，验证两半的起点推进量。

**产物**：`probe/ara/captures/reverse-<host-version>.json` + `REVERSE-FINDINGS.md`。
**这是 `reversed_forward_window` / `host_start_offset` 的出处**，补回来之后倒放
相关的两条测试才算有据可依。

## 约定

1. **原始数据提交进仓**，不要只写结论 —— 结论要能被重新拟合。
2. 捕获文件名带**宿主版本**（如 `fade-axis-7.82.json`）；换版本要重新量，不要覆盖。
3. 引用本目录的代码注释必须写**文件名**，并在文件缺失时视为"该常数未经复核"。
4. 采样脚本一并提交（可复现），不要只提交 JSON。
