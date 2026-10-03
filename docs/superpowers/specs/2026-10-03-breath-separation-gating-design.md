# 气声/张力对「气声分离」开关的依赖门禁

> **修订记录（重要）**：初稿只做了「UI 置灰 + 禁止编辑」与决策侧判定，
> **漏掉了需求 3 的合成侧落地**。经核对发现，真正施加张力的路径
> （`renderer/chain.rs`）用的是**自己的一份**张力判定，并不经过共享实现；
> 且 `breath_gain` 在分离路径内被**无条件**读取。因此「开关关闭 ⇒ 不参与合成」
> 需要在**唯一构造点剥离曲线**才能真正成立。详见 §3.3 与 §4.3。

## 1. 背景与动机

本次改动（`feature/hifisampler-param-parity`）把 HiFiShifter 的张力实现从
「渲染后处理式频谱倾斜」换成了 OpenUtau 的 **Rd 声门模型**方案
（`backend/src-tauri/src/audio/rd_tension.rs`）。Rd 张力**只重塑谐波支**，
而谐波/噪声的分支只在 HNSEP 分离路径里存在 —— 于是张力第一次真正**依赖**气声分离。

同时，`breath_enabled` 的语义被澄清并改名为「气声分离」（Harmonic Separation），
从 `breath_gain` 药丸里独立出来 —— 它是气声与张力**共同的前提**。

但 UI 尚未表达这层依赖：开关关闭时，气声与张力的药丸仍可选中、曲线仍可编辑，
用户画完却听不到任何效果（因为后端此刻已把「张力活跃」也当作分离的触发条件，
行为与界面不一致）。

**目标**：让界面如实反映依赖关系，并让后端口径与界面完全一致。

## 2. 需求（已与用户确认）

1. `breath_enabled` 关闭时，**气声音量**与**张力**两个参数药丸**置灰、不可选中**。
2. 它们的**曲线仍然可见**（用户能看到自己画过什么），但**无法编辑**。
3. 它们**不参与合成**：关闭时在渲染链里彻底不生效 —— 不是"画了但不听"，
   而是**处理器根本收不到这两条曲线**。这一条是需求的核心，且**不能靠 UI
   或决策侧判定实现**（见 §4.3）。
4. 已有工程中「画了张力/气声但没开开关」的情况，**自动置位开关**以免丢失听感。
5. **开关关闭时不走 HNSEP**：即关闭分离后不产生任何分离推理开销
   （这是需求 3 的直接推论 —— 曲线被剥离后分离路径不再被触发）。
6. UI 需**提示开启开关会增加额外渲染成本**：HNSEP 首次分离较慢，用户应被事先告知，
   而不是开了之后才发现卡。

### 明确不做

- **不**影响 `formant_shift_cents`（共振峰）：它走 mel 阶段的 `keyShift`，
  与 HNSEP 无关，关闭分离时照常可用。
- **不**隐藏曲线、**不**删除曲线数据：用户的编辑成果必须保留，重新打开开关即恢复。

## 3. 关键事实（已实测/核对）

### 3.1 旧版张力并不依赖气声 —— 迁移是必需的

`origin/develop` 上 `ensure_hifigan_tension_cache` 在 `render_background_pass` 中
**独立调用**，判据只有 `hifigan_tension_active_for_clip`，**不看** `breath_enabled`。
即：**旧版气声关闭时张力照常生效**。

因此 `README.md` 那句「如果需要编辑张力请务必开启气声」在旧版**并不成立**，
是本次改动让约束变成真的。这直接意味着：

> 已有工程里存在「张力曲线非默认 + 开关关闭」的组合，且它们**当前是有声的**。
> 若只加 UI 门禁而不迁移，这些工程会静默失去张力。**故迁移是必需项，不是可选项。**

### 3.2 `editParam` 是曲线编辑的唯一闸门

`PianoRollPanel.tsx` 中 `editParam` 有 204 处使用；画布上的绘制路径经
`usePianoRollInteractions`（其 `editParam: ParamName` 入参）读取同一状态。
**没有**绕开 `editParam` 的独立曲线编辑入口 —— 因此「禁止编辑」只需保证
`editParam` 不会停留在被禁用的参数上。

### 3.3 张力活跃判定存在**两份**实现（初稿判断有误，已更正）

共享实现 `pitch_editing::hifigan_tension_active_for_clip` 有 4 个调用点：

| 调用点 | 作用 | 属于 |
|---|---|---|
| `audio/mixdown.rs:1057` | 导出混音 | 决策侧 |
| `pitch_editing.rs:416` | 是否由处理器内部承担时间拉伸 | 决策侧 |
| `pitch_editing.rs:1742` | 预渲染判定 | 决策侧 |
| `pitch_editing.rs:2244` | 预渲染判定（另一分支） | 决策侧 |

**但这 4 处都是"决策侧"** —— 决定是否预渲染、是否跳过外部拉伸、能否复用缓存。
它们**都不负责真正施加张力**。

真正施加张力的路径在 `renderer/chain.rs`，而它**自己算了一遍** `tension_active`
（`chain.rs:316`，见 §4.3 泄漏路径 A），没有调用上面的共享实现。
`apply_rd_tension` 的唯一调用点是 `chain.rs:440`。

> **教训（写下来避免重犯）**：看到"某判定有 4 个调用点"就推断"改一处即全一致"
> 是错的 —— 必须先确认**消费侧**（真正起作用的地方）是否也走同一函数。
> 本例中消费侧另有一份实现，初稿因此漏掉了关键泄漏路径。

### 3.4 参数默认值（迁移判据用）

| 参数 | 默认值 | 含义 |
|---|---|---|
| `breath_gain` | `1.0` | 噪声支混入倍率（1.0 = 原样混入，即"没动过"） |
| `hifigan_tension` | `0.0` | 张力百分比（0 = 无效果） |
| `breath_enabled` | `0` | 开关 |
| `formant_shift_cents` | `0.0` | 不参与迁移（与分离无关） |

`extra_param_enabled()` 的判据是 `value >= 0.5`，缺失键视为 `0.0`（关闭）。

## 4. 设计

### 4.1 门禁判据（单一来源）

新增一个前端可复用的纯函数式判据（后端对应 `extra_param_enabled`）：

```
needsSeparation = breath_enabled            // 开关本身
gatedBySeparation(paramId) =
    paramId == "breath_gain" || paramId == "hifigan_tension"
```

即：**开关关闭 ⇒ `breath_gain` 与 `hifigan_tension` 被门禁**。

### 4.2 前端

1. **`ParamToolbarPill` 增加 `disabled` 属性**
   - 视觉：整体降低不透明度 + `cursor: default`（复用既有
     `.param-pill__seg--inert` 的置灰语汇，不新造样式体系）
   - 行为：`onSelect` 不触发；标签加 `aria-disabled="true"`
   - **保留** `onToggleEye` 可用 —— 需求要求曲线仍可见，眼睛是"显隐叠加曲线"的
     控件，与"能否编辑"无关。若把眼睛一起禁用，用户将无法控制曲线显隐，
     与需求 2 冲突。

   > **一处解释，若与预期不符请指出**："曲线仍可见"理解为**本次门禁不会隐藏曲线**，
   > 即置灰只影响"可选中/可编辑"。若用户此前**自己**把某条曲线眼睛关掉了
   > （`secondaryParamVisible`），关闭分离后它**仍保持隐藏** —— 那是用户先前的
   > 显式选择，不因门禁而改变。若期望的是"门禁时强制显示曲线"，需要额外强制
   > 置位 `secondaryParamVisible`，请告知。

2. **开关 tooltip 必须提示渲染成本（需求 6）**

   现有 `breath_separation_tooltip` 已提到"首次开启需要较长时间处理"，
   需**加强为明确告知这是额外渲染成本**，例如：
   关闭时不走 HNSEP、不产生分离开销；开启后需先做一次谐波/噪声分离
   （首次较慢），气声与张力才会生效。

   同一提示语需覆盖 5 个语言（`catalogIntegrity` 测试强制对齐）。

   置灰的两个参数药丸的 tooltip 也应说明**为何不可用**（"需先开启气声分离"），
   否则用户只看到灰掉却不知原因 —— 这是可用性要求，不是可选项。

3. **`editParam` 回退**
   - 在既有的 `processorParams` 变化回退 effect 中追加：若 `editParam` 是被门禁的
     参数，则 `dispatch(setEditParam("pitch"))`。
   - 这是需求 2「无法编辑」的**实际执行点**（见 §3.2）：曲线仍在画布上可见，
     但当前编辑参数已不在它身上，因此绘制操作不会落到它。

3. **`setEditParam` 的入口位置（已核查，结论：门禁不能放在 reducer）**

   全量检查结果：`setEditParam` 只有**一个** action 定义
   （`features/session/sessionSlice.ts:2453`），其余均为 `dispatch(setEditParam(..))`
   调用。**没有**绕过 reducer 的直接写入点。

   但门禁**不能**放在该 reducer 内：`editParam` 属于 session slice，而开关状态
   （`extra_params["breath_enabled"]`）是 `PianoRollPanel` 的**组件内 React state**
   （`processorStaticValues`，见 `PianoRollPanel.tsx:1960`），reducer 读不到它。

   因此门禁落在 **§4.2 第 2 点的回退 effect**（组件内，同时能看到
   `editParam`、`processorStaticValues` 与 `processorParams`）。
   该 effect 已存在且职责相同（"editParam 失效则回退 pitch"），
   追加门禁判据即可，无需新增机制。

   键盘快捷键/参数下拉等入口最终都经 `setEditParam` 落到 `editParam`，
   而回退 effect 以 `editParam` 为依赖 —— 因此无论从哪个入口进来都会被纠正。
   这也意味着回退是**收敛的**（不会反复 dispatch）：判据只看
   `editParam` 与开关状态，回退到 `pitch` 后判据不再成立。

### 4.3 后端严格跟随（★ 需求 3 的真正落点）

需求 3 是「**不参与合成**」。**仅加 UI 门禁与 `hifigan_tension_active_for_clip`
的开关判定是不够的** —— 初稿在此处判断有误，实际核对后存在**两条泄漏路径**，
必须一并堵住。

#### 泄漏路径 A：`renderer/chain.rs` 的本地张力判定（第 2 份实现）

`chain.rs:316` **自己算了一遍** `tension_active`（未调用 §3.3 的共享实现）：

```rust
let tension_active = tension_curve.is_some_and(|c| {
    c.iter().any(|v| v.is_finite() && v.abs() > TENSION_ACTIVE_EPSILON)
});
let needs_separation = tension_active || separation_switch_on;   // ← 开关关闭也可能为 true
...
let tensioned_harmonic = Self::apply_rd_tension(&harmonic, cc, tension_curve);  // ← 无条件施加
```

于是「开关关 + 张力曲线非默认」时 `needs_separation` 仍为 `true`，
`process_breath` 被调用，张力**照常施加**。

> 修正之前的错误结论：初稿称"在 `hifigan_tension_active_for_clip` 加判据，
> 四个调用点自动一致"。这只覆盖 `mixdown.rs` 与三个 `pitch_editing.rs` 调用点，
> **不覆盖 `chain.rs` 这条真正施加张力的路径** —— 而它恰恰是关键路径。

#### 泄漏路径 B：`breath_gain` 在分离路径内被无条件读取

`process_breath` 内 `breath_curve` 直接读 `cc.extra_curves["breath_gain"]`
（`chain.rs:489`），**不看开关**。只要因任何原因进入分离路径（含泄漏路径 A），
用户画的 `breath_gain` 曲线就会被真实施加。

（`snapshot.rs:640` 与 `mixdown.rs:1077` 的**预览/导出混音**侧已有开关 gate，
但链内这条路径没有。）

#### 落地方式：集中在唯一构造点剥离

`ClipProcessContext` 全仓库**只有一处**构造（`pitch_editing.rs:2052`），
这是理想的收口位置。开关关闭时，在此处把两个被门禁的曲线键**从下发的
`extra_curves` 中剔除**：

```
if !extra_param_enabled(extra_params, "breath_enabled") {
    // 需求 3：开关关闭时这两个参数不参与合成。
    // 剥掉键而非传零值：曲线缺失是"未编辑"的既有语义（默认值 1.0 / 0.0），
    // 传零值会让 breath_gain 变成"噪声全静音"，与"不参与合成"不同。
    ctx_curves.remove("breath_gain");
    ctx_curves.remove("hifigan_tension");
}
```

**为什么在构造点剥离、而不是在每个消费点判断**：

- `ClipProcessContext` 只有一处构造 ⇒ 一处改动同时覆盖
  `apply_rd_tension`（张力）与 `process_breath` 的 `breath_gain` 混音，
  以及未来任何新增的消费点；
- 避免再次出现 §3.3 那种"以为有单一实现、实际有第 2 份"的漂移；
- 语义清晰：**下发给处理器的就是"本次真正生效的参数"**。

只有在 `extra_curves` 确需修改时才克隆（关闭开关是少数情况），
开关开启时保持借用、零分配。

#### 同时仍要做的两件事

1. **`chain.rs` 消除第 2 份实现**：改为调用 §3.3 的共享判定
   （或至少加入开关判据），使"张力是否活跃"全仓库单一来源。
   这既修掉泄漏路径 A，也消除一个既有的重复实现隐患。
2. **`needs_separation` 语义收紧**：剥离曲线后此式自然等价于
   `separation_switch_on`，但要显式改写并更新注释 —— 让"进入分离路径"
   读作"开关开启"，而不是依赖"张力恰好被剥离"这一间接结果。

#### 与 `hifigan_tension_active_for_clip` 的关系

该函数仍**需要**加入开关判据：它服务的是**决策侧**（是否预渲染、是否跳过外部
拉伸、导出能否复用缓存）。若不改，开关关闭时 `mixdown.rs:1057` 会因
`clip_tension_active == true` 而放弃缓存复用，做无谓的重渲染 —— 结果正确但白费。

即：**决策侧**用 `hifigan_tension_active_for_clip`（§4.3 第 1 点），
**合成侧**用构造点剥离（本节）。两者都需要，且判据同源。

4. **迁移：`migrate_legacy_breath_separation`**

   并入既有 `migrate_legacy_common_param_curves` 的调用时机
   （`project.rs:225`、`project.rs:235`、`project.rs:595`），避免新增调用点。

   对每个 root track 的参数记录：

   ```
   若 extra_params["breath_enabled"] 已为开启 → 跳过
   若 hifigan_tension 曲线在任意区间偏离 0.0  → 置位 breath_enabled = 1
   若 breath_gain    曲线在任意区间偏离 1.0  → 置位 breath_enabled = 1
   ```

   同时需覆盖 **clip 级 `extra_params` / `extra_curves` 覆盖**（clip 可覆盖轨道，
   见 `Clip.extra_params: Option<HashMap<..>>`）：若某 clip 自带张力/气声曲线而
   轨道未开开关，则该 clip 所在轨道也要置位。

   **为何用"曲线非默认"而非"曲线存在"**：空曲线与全默认值曲线语义等价，
   把"存在但全默认"也当作需要开启会无端触发一次 HNSEP 渲染（首次较慢）。

### 4.4 文档

- `README.md:99` 与 `docs/i18n/README_en.md` 对应句：修正为"气声与张力都依赖
  气声分离开关；开启后曲线才可编辑并生效"。原句在旧版是错的（§3.1），
  改后才是准确描述。

## 5. 数据流

```
用户关闭开关
   │
   ├─ 前端：breath_gain / hifigan_tension 药丸 disabled
   │        editParam 若在其中 → 回退到 pitch（曲线仍可见，但不再可画）
   │        曲线数据本身不动
   │
   └─ 后端：extra_param_enabled("breath_enabled") == false
            ├─ 【决策侧】hifigan_tension_active_for_clip() → false
            │     └─ 4 个调用点认为"无张力"（不预渲染、不跳过拉伸、可复用缓存）
            └─ 【合成侧】ClipProcessContext 构造点剥离 breath_gain / hifigan_tension
                  ├─ chain.rs: 收不到张力曲线 → apply_rd_tension 原样返回谐波
                  ├─ chain.rs: 收不到 breath_gain → 噪声按默认 1.0 混回
                  └─ needs_separation → false → 不跑 HNSEP

   两路都必须做：只做决策侧 → chain.rs 仍会施加张力（泄漏路径 A）；
   只做合成侧 → 决策侧白跑重渲染（结果对但浪费）。

用户重新开启开关
   └─ 曲线数据仍在 → 立即可编辑、立即生效（无需重建任何缓存）
```

## 6. 错误处理

- **开关开启但 HNSEP 不可用**：沿用本次已实现的显式报错（不再静默丢弃），
  错误信息已区分"气声分离/张力/两者"。
- **迁移遇到非有限值曲线**：跳过（不置位），与 `TENSION_ACTIVE_EPSILON`
  对 NaN/Inf 的处理口径一致。
- **e2e 一致性**：前端门禁与后端 `extra_param_enabled` 必须同源。前端判据写在
  一处并加注释指向后端的同一常量语义，避免两侧对"0.5 阈值"的理解分叉。

## 7. 测试

### 后端

1. `tension_alone_requires_separation_regardless_of_switch` —— **需改写**：
   现状断言"张力活跃即分离"（不看开关），与本次需求相反。改为
   "开关关闭 ⇒ 张力不活跃"，并保留"曲线全 0 不触发分离"。
2. `hifigan_tension_active_for_clip`：开关开 + 曲线非默认 ⇒ true；
   开关关 + 曲线非默认 ⇒ **false**（新门禁）。
3. 迁移：
   - 张力曲线非默认 + 开关关 ⇒ 迁移后开关为开
   - 气声曲线非默认 + 开关关 ⇒ 迁移后开关为开
   - 两者都默认 + 开关关 ⇒ **不**置位（不无端触发 HNSEP）
   - 开关已开 ⇒ 保持开
   - clip 级覆盖同样触发轨道置位
   - 非有限值曲线不触发置位
4. `renderer::chain`：开关关闭时 `needs_separation == false`。
5. **合成侧剥离（需求 3 的核心断言）**：
   - 开关关闭 + `hifigan_tension` 曲线非默认 ⇒ 构造出的 `ClipProcessContext`
     的 `extra_curves` **不含** `hifigan_tension`
   - 开关关闭 + `breath_gain` 曲线非默认 ⇒ `extra_curves` **不含** `breath_gain`
   - 开关**开启** ⇒ 两个键**原样保留**（不得误剥）
   - 开关关闭 + 曲线存在 ⇒ `formant_shift_cents` **仍保留**（不受门禁）
   - 剥离是"键缺失"而非"传零值"：`breath_gain` 缺失的既有语义是默认增益 1.0，
     故断言应检查键不存在，而非值为 0
6. **开关关闭时不触发 HNSEP（需求 5）**：开关关闭 + 张力/气声曲线均非默认时，
   断言分离模型**未被调用**。可行做法：用 `hnsep_onnx` 的分离缓存或
   `probe_load` 侧的可观测计数（若无计数，则断言 `needs_separation == false`
   加上端到端逐样本一致已足以覆盖，不必为测试新增生产代码计数）。
7. **端到端（最重要的一条）**：证明开关关闭时张力确实没进合成。

   对照必须是**同为开关关闭**下的两次渲染，二者应**逐样本一致**：

   | | 开关 | 张力曲线 |
   |---|---|---|
   | 甲 | 关闭 | +80（明显非默认） |
   | 乙 | 关闭 | 不存在 |

   若甲 ≠ 乙，说明张力仍在起作用（即泄漏路径 A 未被堵住）。

   > **一个容易写错的对照**（初稿在此处写错，记录下来）：不能拿
   > "开关关闭 + 张力 +80" 去比 "开关**开启** + 曲线全 0"。后者会进入分离路径，
   > 而 `breath_gain` 全 0 会命中 `gain_is_zero` 提前返回（丢弃噪声支），
   > 与前者（非分离路径，噪声默认 1.0 混回）**本就不同** ——
   > 那样断言失败反映的是路径差异，而非张力泄漏，属于无效对照。

   同理，"开关关闭 + `breath_gain` 曲线非默认" 应与 "开关关闭 + 无 `breath_gain`
   曲线" 逐样本一致。

### 前端

- `tsc --noEmit` + lint + 既有 i18n 完整性测试。
- 若 `setEditParam` 的门禁落在 reducer：加一个纯函数单测覆盖
  "关闭时尝试切到 breath_gain/tension 会被拒绝或回退"。

## 8. 影响面与兼容性

| 方面 | 影响 |
|---|---|
| 旧工程 | 首次打开会自动开启开关（若画过张力/气声），**听感保持**；代价是该工程首次渲染需跑一次 HNSEP |
| 渲染缓存 | 迁移改变 `extra_params` ⇒ 渲染键变化 ⇒ 这些工程需重新渲染一次（一次性） |
| `formant_shift_cents` | 不受影响 |
| 未画过张力/气声的工程 | 完全无感知（开关保持关闭，不触发迁移） |
| 新工程 | 需先开开关才能画气声/张力曲线 —— 这是本次要建立的正确工作流 |

## 9. 待办（实现时确认）

- [x] `setEditParam` 是否存在绕过 reducer 的直接写入点 → **无**，且门禁不能放
      reducer（读不到开关状态），落在既有回退 effect（见 §4.2 第 3 点）
- [ ] clip 级 `extra_params` 覆盖在迁移中的遍历方式
- [x] 前端"参数下拉"是否也列出被门禁参数 → **列出**：下拉由
      `processorParams`（= `kind.type === "automation_curve"` 的过滤结果，
      `PianoRollPanel.tsx:2065/2231`）驱动，含 `breath_gain` 与 `hifigan_tension`。
      因两者在关闭时仍**可见**（需求 2），下拉中保留条目但需置灰不可选。
