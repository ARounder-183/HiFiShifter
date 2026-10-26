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

### 2.6 长度预算（状态片 / tooltip）

同一句话放在对话框正文里是正常的，放在**状态栏那一行**就是灾难：状态片是
`shrink-0`、父容器是 `truncate`，超长提示会被**从中间裁掉且没有省略号**
（`text-overflow` 不作用于 flex 子元素）—— 用户读到的恰好是"什么都没说"的那半句。

| 面 | 中文/日文/韩文 | 英文 | 判据 |
|---|---|---|---|
| 状态片（`status_*`、`cap_reason_*`、`editor_route_unavailable`、`host_key_focus_failed`） | ≤ 44 字 | ≤ 62 字符 | 一句话、一个**状态**；不写"请……" |
| tooltip（`*_tooltip` / `*_tip` + 登记在案的例外） | ≤ 46 字 | ≤ 80 字符 | 一句短语；解释写进对话框正文 |

**状态片只放状态，指令放进 tooltip。** `AppStatusChip` 支持 `tooltip`（走
`data-tooltip` 通道，会如实换行且跟随主题），那才是"下一步怎么做"该待的地方。

> 数值是**棘轮**：取"今天的最差值 + 余量"，不追认历史，但禁止新增更长的。
> 新增文案时按理想值写，别贴着上限写。

由 `src/i18n/catalogIntegrity.test.ts` 的「长度预算」两条用例强制。新增一条状态栏
文案必须同时改那里的 `STATUS_BAR_KEYS` —— "必须停下来改测试"正是要的那一步。

### 2.7 没有死键

目录里的每个键都必须**至少被引用一次**。历史教训：捕获面板从"一整块面板"改成
"工具栏一个图标 + 设置窗口"时，旧的一整族文案（`capture_guide_*`、`capture_output_*`
…）没有跟着删 —— 实测 **49 个键、245 条译文**再也不会被渲染，而它们不会报错、
不会被发现，只会让下一个维护者以为"词表里有，说明界面上有"。

动态拼接的键族（`history_op_<op>`、`theme_<mode>`、`ripple_tooltip_<mode>`、
`recording_error_<code>`、`installer_*`）豁免，但**新增拼接点必须登记**在
`src/i18n/keyReferenceIntegrity.test.ts` 的 `DYNAMIC_PREFIXES` 里。

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

### 3.5 悬停提示走 `data-tooltip`，不是 `title`

§3.4 管的是"文案有没有走键"，这一条管**走哪条显示通道**。同一个键挂在
`title` 上还是挂在 `data-tooltip` 上，用户看到的是两样东西：

| 通道 | 换行 | 明暗主题 | 实例 |
|---|---|---|---|
| 原生 `title` | `\n` 被折成一个空格 | 跟随浏览器/系统，常与页面撞色 | —— |
| `data-tooltip`（`components/AppTooltip.tsx`） | 如实渲染（`white-space: pre-wrap`） | 跟随本项目变量 | 状态片、列表行、菜单项 |

因此：**项目组件一律用 `tooltip` prop**（`AppStatusChip` / `AppListRow` /
`AppMenuItem` / dialog action），裸 DOM 元素写 `data-tooltip`。典型受益者是
需要两行说明的场景 —— `dock/DockLayoutMenu` 的 `dockDragHint` 与
`features/ara/PluginApplyStatus` 的修订读数都是靠它才能完整显示。

`title` 的合法用途只有两种，都**不是**悬停提示：自定义组件的标题槽
（`AppDialog title=` 渲染成 `<h2>` 而不是 DOM 的 `title` 属性），以及
文本被截断时的完整文本兜底（`keybindingsActionRow` 的做法）。
自定义组件（大写 JSX 名）上的 `title` prop 一律放行 —— 它不会渲染成原生 tooltip。

由 ESLint 强制（`frontend/eslint.config.js` 的 `no-restricted-syntax`）：
`src/features/ara`、`src/components/layout`、`src/components/dock` 下
**小写标签**上的 `title` 直接报错。其余目录尚未纳入，迁移到那一层时把
`files` 列表扩过去即可，规则本身不用改。

**菜单项一律不用 tooltip。** `AppTooltipProvider` 在菜单弹出期间全局抑制气泡
（既定规则，`AppTooltip.tsx:123-130`），所以菜单条目上无论 `title` 还是
`data-tooltip` 都不会显示。这类"为什么点不了"的理由应当**常驻**渲染成
第二行弱化小字（见 `MenuBar.tsx` 的 `StandaloneOnlyItem`），而不是挂在
悬停上 —— 它比悬停多一步，但至少真的看得见。

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
