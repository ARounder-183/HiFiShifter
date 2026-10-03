# 气声/张力对「气声分离」开关的依赖门禁

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
3. 它们**不参与合成**（关闭时彻底不生效）。
4. 已有工程中「画了张力/气声但没开开关」的情况，**自动置位开关**以免丢失听感。

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

### 3.3 张力活跃判定已有单一实现（改一处即全一致）

`pitch_editing::hifigan_tension_active_for_clip` 有 4 个调用点，全部经由它：

| 调用点 | 作用 |
|---|---|
| `audio/mixdown.rs:1057` | 导出混音 |
| `pitch_editing.rs:416` | 是否由处理器内部承担时间拉伸 |
| `pitch_editing.rs:1742` | 预渲染判定 |
| `pitch_editing.rs:2244` | 预渲染判定（另一分支） |

在该函数内加入开关判据，四处**自动**一致，无需逐个修改 —— 这正是把门禁放在
这里的理由（若分散到各调用点，必然漂移）。

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

2. **`editParam` 回退**
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

### 4.3 后端严格跟随

1. **`hifigan_tension_active_for_clip` 加入开关判据**

   在该函数开头增加：开关关闭 ⇒ 直接返回 `false`。这样 §3.3 的四个调用点
   一次性对齐，且与前端 UI 语义完全一致。

2. **`renderer/chain.rs` 的分离门禁**

   现状 `needs_separation = tension_active || separation_switch_on`。
   若张力活跃已含开关判定（上一步），则此式等价于 `separation_switch_on`；
   保留显式写法并更新注释，因为「张力活跃」在别处（如 mixdown）仍是有意义的概念，
   而分离路径的进入条件应读作"开关开启"。

3. **`breath_gain` 的生效条件**

   非分离路径本就不读 `breath_gain`（噪声支不存在），故无需额外改动；
   但要确认不存在"未分离却按 breath_gain 处理谐波"的旧逻辑残留（现状无）。

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
            ├─ hifigan_tension_active_for_clip() → false
            │     └─ 4 个调用点全部认为"无张力"（不预渲染、不导出张力）
            └─ renderer::chain needs_separation → false
                  └─ 不跑 HNSEP；张力曲线不参与合成

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
