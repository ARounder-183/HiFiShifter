# 文案风格指南（i18n Style Guide）

> 本文档是 `src/i18n/catalogIntegrity.test.ts` 的文字版说明。
> **每条规则都有对应的自动化检查** —— 只写在文档里、没有测试守着的约定，
> 在这个项目里已经证明会漂移（见文末「这份文档的由来」）。

---

## 1. 结构规则（由 `tsc` 与门禁测试强制）

### 1.1 键集合必须五语系完全一致

`MessageKey` 由 `en-US` 的键推导，`tsc` 会在任一语系缺键时报错。
门禁测试额外检查「多余键」与「键集合逐语系相等」。

**新增文案时，五个文件都要加。** 顺序不重要（各文件键序本来就不同），
但**必须齐全**。

### 1.2 占位符逐键一致

同一个键在五个语系里出现的占位符集合必须相同（去重后比较）。
`{n}` 拼成 `{N}`、或英文用了 `{count}` 而中文忘了，都会被拦下。

### 1.3 键名格式

- 一律 `snake_case`。
- 用 `域_子域_条目` 的形式，例如 `export_dialog_mp3_bitrate`。
- **不要**引入 camelCase 段（历史上有 7 个反例：`kb_preset_spaceReturnPlayhead`、
  `render_cache_skip_tooShort` 等，已列入待清理）。
- 通用词（`ok` / `cancel` / `close` / `none` …）用**裸键**；
  有特定语境的同类词用全限定键（`benchmark_close`）。

---

## 2. 排版规则

### 2.1 大小写

| 类别 | 规则 | 例 |
|---|---|---|
| 菜单项 / 对话框标题 / 按钮 / 标签 / 分组标题 | Title Case | `Import VocalShifter Project` |
| 描述 / 提示 / 状态 / tooltip（`_desc` `_hint` `_tip` `status_*`） | 句首大写 | `Folded 3 clips to mono automatically` |

**禁止整体大写。** 历史上 en-US 独有 7 处全大写（`TRACKS`、`STATUS`、`DELETED` …），
而 zh-CN / ja-JP / ko-KR 都是正常大小写 —— 只有英文在喊。
门禁测试会拦下新的全大写标签（纯缩写如 `BPM` / `MIDI` 除外）。

### 2.2 标点

- **省略号**：菜单项打开对话框时以 `...`（三个半角点）结尾。
  全仓统一用 ASCII 三点，**不使用** U+2026 `…`。
  （历史遗留：`kb_press_modifier` 曾用 `…`，且五个语系里 3 个用 `…`、2 个用 `...`。）
- **标签不带尾冒号**：`"BPM:"` 这种写法由布局负责分隔。
  需要「标签 + 值」拼接时用 `tVars` 组合，不要靠值尾部的空格。
  词典为此准备了**共享模板**：`common_label_value`（`"{label}：{value}"`，各语系自带
  冒号）、`common_value_sep`（富内容行的冒号段）、`common_parenthetical`
  （`"{value}（{note}）"`）。**代码里不得出现硬编码的 `：` 或 `": "` 拼接** ——
  那会让英文冒号变全角、中文冒号变半角（两种事故都发生过）。
- **CJK 用全角标点**：`，。：（）？！`。
  半角括号只允许包裹纯拉丁/数字内容（如 `MP3 (VBR)`）；
  门禁测试会拦下「中文里用半角括号包中文」，以及含汉字的值里出现
  半角 `, ; : ? !`（豁免：`time_unit_clock` 等格式掩码）。
- **值内无多余空白**：首尾空格、连续空格、制表符、全角空格都会被门禁拦下。
  唯一豁免是「纯标点/符号」值（如 `common_value_sep` 的 `": "`）——
  那是布局分隔符资产，不是文案片段。

### 2.3 单位与数字

- 数字与单位之间的空格**交给引擎**：用 `useI18n().unit(value, "kilohertz")`，
  不要手写 `"8kHz"`。
  历史上中文/日文/韩文都出现过 `8kHz`（紧）与 `1 kHz`（松）在同一文件里并存。
- 计数用 `Intl.NumberFormat`（`useI18n().number(...)`）以获得正确的千分位。

### 2.4 复数

**禁止伪复数 `(s)`。** 会渲染出 `"1 clip(s)"`。

词典里用 `单数|复数` 的形态，消费端用 `useI18n().plural(key, count)`：

```ts
// en-US
notebook_clip_unit_clips: "clip|clips",
// zh-CN —— 单形态表示该语言不区分复数
notebook_clip_unit_clips: "个音频块",
```

```tsx
{plural("notebook_clip_unit_clips", count)}
```

形态选择走 `Intl.PluralRules`，因此新增语系时不需要改代码。

**若调用点拿不到数量**（例如静态说明文字），改写成数-neutral 的表述，
不要保留 `(s)`：

```diff
- "The following media file(s) have been modified externally."
+ "The following media files have been modified externally."
```

### 2.5 快捷键

**禁止硬编码 `Ctrl`。** macOS 的主修饰键是 Command（`⌘`）。

词典里写 `{modifier}` 占位符，消费端用 `useI18n().shortcut(key)`：

```ts
notebook_toolbar_bold: "Bold ({modifier}+B)",
```

```tsx
<ToolbarButton tooltip={shortcut("notebook_toolbar_bold")} ... />
```

该占位符的取值与键位设置界面的 `formatKeybinding` 使用**同一条规则**
（`IS_MAC ? "⌘" : "Ctrl"`），因此两处显示必然一致。

---

## 3. 语义规则

### 3.1 同一命名族内的短标签不得重名

`param_btn_breath` 与 `param_btn_breathiness` 曾经在 en-US / ja-JP / ko-KR 里
**同为 `"BRE"`** —— 参数编辑器里出现两个标签完全相同的按钮。

门禁测试按「前两段作为命名族 + 短标签（≤12 字符）」比对，
族内重名即失败。已核对为同义复用的例外写在测试的 `KNOWN_BENIGN` 里，
**新增条目必须写明理由**。

### 3.2 不借用其他模块的键

历史上有 `MidiTrackSelectDialog` 借用 keybindings 的 `kb_close`、
`ChannelImportDialog` 借用 RenderCache 的 `render_cache_settings_saved`。
需要新文案就加新键。

### 3.3 组件里不拼字符串

不要 `t("a") + " (" + n + ")"` 或 `t("a").replace("{n}", String(n))`。

用 `tVars` / `plural` / `shortcut`。尤其**不要**把译文与后端返回的英文原文
拼接（`utils/statusText.ts` 曾如此）。

### 3.4 组件里的文案必须走键，不能写字面量

前面三条都在讲"怎么用键"；这一条讲**有没有用键**。

历史上这一点完全没有门禁：`catalogIntegrity` 只校验**词典**，
`keyReferenceIntegrity` 只校验**被引用的键存在**。于是硬编码中文对它完全隐形
—— 合并 `e834ef54`（ARA 插件）时，整整两个新面板（ARA 宿主会话、插件应用状态）
加一个工具栏按钮的文案全是中文字面量，五个语系**一个键都没加**，
而 CI 全绿。

现在由 `src/ui/araConformanceGates.test.ts` 守住：

- 用户可见的 JSX 属性（`title` / `aria-label` / `placeholder` / `alt` / `label`）
  与单行 JSX 文本节点里**不得出现中文**；
- 确实必须保留中文的位置（典型：字体预览样本要渲染中文字形），
  在**那一行**加 `hs-text-exempt` 标记并写明理由 —— 行级豁免，
  同一个文件里新写的硬编码文案仍然会被拦下。

**未覆盖**：`.ts` 文件里作为**普通字符串**存在的文案（如 `throw new Error("…")`、
`console.warn("…")`）。这类字符串无法与"日志/协议标识"区分，需要 review。

---

## 4. 门禁测试覆盖范围

`src/i18n/catalogIntegrity.test.ts`：

| 检查 | 抓什么 |
|---|---|
| 键集合逐语系相等 | 漏译 / 多余键 |
| 值非空 | 空字符串 |
| 占位符逐键一致 | `{n}` 拼写漂移 |
| 无伪复数 `(s)` | `"1 clip(s)"` |
| 复数分隔符格式 | `\|` 数量与两侧空值 |
| 无硬编码 `Ctrl+` | macOS 显示错误 |
| 英文标签不整体大写 | `TRACKS` / `STATUS` |
| 命名族内短标签不重名 | `BRE` 冲突 |
| 中文不用半角括号包中文 | CJK 排版混排 |
| 中文行文不用半角 `,;:?!` | `可用提供者:` 与全角冒号并存 |
| 值内无多余空白 | `" (unavailable)"` 式空格拼接 |

`src/ui/araConformanceGates.test.ts`（文案部分）：

| 检查 | 抓什么 |
|---|---|
| 用户可见属性 / JSX 文本里无中文 | 整个界面没走词表（见 §3.4） |

`src/i18n/format.test.ts` 覆盖格式化层本身的边界行为。

**未覆盖**（需要人判断）：措辞是否自然、术语是否统一（如「音频块 / 音訊塊 /
クリップ」的选词）、Title Case 判断中歧义项。这些仍需 review。

---

## 这份文档的由来

审查时的事实：1638 键 × 5 语系的**键结构**严丝合缝（`tsc` 强制），
但**文本风格完全失控** —— 因为除键配额外没有任何检查：

- 213 个测试文件里只有 2 个涉及 i18n，且 `historyOpLabels.test.ts`
  只覆盖 47/1638 键，而且它是在一次真实的五语系回归**之后**才补上的；
- `eslint.config.js` 没有任何 i18n 规则；
- 词典里 1611/1638 个键在各语系文件中的索引位置不同（最大位移 1482 位），
  **人肉 diff 根本不可行**。

结论：只靠 review 的约定一定会漂移。因此本文档的每一条都配了测试。
