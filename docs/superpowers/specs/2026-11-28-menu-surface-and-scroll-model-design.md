# 菜单表面与滚动模型（设计记录）

> 状态：**已实现**。本文记录"全仓只有一套菜单表面"这条结论、它的形状、以及新增菜单时
> 该怎么写。实现见 `frontend/src/ui/Menu.tsx`、`MenuScrollScope.tsx`、
> `menuScrollModel.ts`、`useMenuKeyboard.ts`、`index.css` 的「上下文菜单样式模型」块。

## 1. 一句话

**全仓只有一种菜单表面 `.hs-menu`；它没有竖直滚动条，只有两条 sticky 提示行。**

## 2. 为什么把 Radix 菜单全部换掉

迁移前有两族实现：

| 族 | 实现 | 用在哪 | 滚动装置 |
|---|---|---|---|
| 自绘 | `src/ui/Menu.tsx` 三个原语 + 5 个手写壳 | 全部右键菜单、全部按钮下拉、子面板 | 原生滚动条 |
| Radix | `@radix-ui/themes` 的 `DropdownMenu` | 菜单栏 8 个下拉、停靠布局菜单、参数工具栏 4 个下拉、日志级别子菜单 | Radix 自绘的 4px 滑块 |

两族各自演化出了自己的键盘模型、焦点模型、定位算法与 CSS 覆盖。**"哪一套生效"取决于
用户点的是菜单栏还是画布** —— 这是分叉的根源，不是细节问题。

迁移后：`src/` 内不再出现 `<DropdownMenu.`（门禁 G-5 守住），`index.css` 里对
`.rt-DropdownMenuContent` / `.rt-ContextMenuItem` 等的一整块覆盖被删除。

**`Select.Content` 不在此列**：它是 `role="listbox"` 的表单控件（下拉选值），不是命令
菜单。原生 `<select>` 的下拉本来就有滚动条，用户对它有明确预期。

## 3. DOM 契约

```
.hs-menu                        position: fixed；自身是滚动容器
│   overflow: hidden auto; padding: 0; scrollbar-width: none
├─ .hs-menu__scroll-hint--up    position: sticky; top: 0; height: 0
├─ .hs-menu__content            padding 的持有者；内容高度变化的观测点
│    └─ …条目 / 手写内容…
└─ .hs-menu__scroll-hint--down  position: sticky; bottom: 0; height: 0
```

三条必须同时成立的结构事实：

1. **壳的 `padding` 是 0**。sticky 的约束盒是父元素的**内容盒**：壳留 4px 内边距，
   提示行就跟着内缩 4px，边缘露出一条没被渐变的空白（实测 165 vs 期望 161）。
2. **提示行是壳的直接子元素**。`MenuScrollScope` 统一渲染，调用方不手写。
3. **壳上不得出现 `transform` / `filter` / `contain` / `will-change`**。子面板是
   `position: fixed`，任何建立包含块的属性都会把它钉回壳里、被 `overflow` 裁掉
   （历史事故：子菜单可见宽度 5px）。

## 4. 滚动模型

### 4.1 为什么没有滚动条

菜单是瞬态表面，滚动条是常驻机械装置。它还**偷宽度**（原生条 10px）：出现时内容右缘
整体左移，右对齐的快捷键列立刻不在同一列。而它给出的语义只是"能滚"，不说"这个方向
还有几项"。

`scrollbar-width: none` 只改**渲染**，不改可滚范围 —— 滚轮、方向键、程序化
`scrollTop` 全部照常。不用 `::-webkit-scrollbar` 自绘：本仓实测分数 DPR 下拖拽映射会
放大 dpr² 倍。

### 4.2 提示行

渐变 + 三角：内容在边缘溶进菜单底色（"下面还有"的语法），中间一支小箭头（"可以滚"的
语法）。**不做 REAPER 那种实心灰条** —— 那是把提示做成"另一行菜单项"，而菜单项的语法
是"点一下执行一条命令"。

| 手势 | 行为 | 参数（鼠标 / 触摸） |
|---|---|---|
| 悬停 | 等 `hoverDelayMs` 后慢滚，速度从 40% 爬到 100% | 180ms / —；220 / 320 px/s |
| 点击 | 立即翻一页（`easeOutCubic`） | 140ms / 0ms |
| 长按 | ≥ `pressDelayMs` 后匀速快滚 | 420ms；900 px/s |
| 停止 | 指针离开 / 抬起 / 取消 / 到边界 / 滚轮介入 / 失焦 / 卸载 | — |

零占位：`height: 0` + 内部绝对定位，`scrollHeight` 在提示行出现 / 消失时不变。

**命中区在 `::after` 上**，不在宿主上：宿主的盒高是 0，0 高的元素永远打不中。

### 4.3 高度预算

`--qt-menu-max-h: calc(100vh - var(--qt-menu-edge-gap) * 2)`。

曾经是 `60vh` —— 那是"菜单最多占屏幕六成"的观感约定，与"这张菜单到底放不放得下"
无关：900px 高的窗口里 60vh = 540px，而多选音频块时的右键菜单实测 574px，于是**哪怕
视口还剩 360px 没用上，菜单也照样出滚动条**。

## 5. 键盘模型

焦点**不离开触发器**（`aria-activedescendant`），方向键只移动"当前项"。这与 Radix 的
roving tabindex 是两种都合法的 ARIA 模式，但不能混 —— 换掉它等于同时改掉 30 个右键菜单
的焦点行为与时间轴"关闭后焦点回画布"的契约。

唯一的例外是**键盘打开**的锚定菜单：`autoFocus` 把焦点收进菜单（鼠标打开时留在触发器
上，与 VS Code / Windows 菜单栏一致）。

| 按键 | 行为 |
|---|---|
| `↑` / `↓` | 在可选项之间循环（跳过禁用项与标题行） |
| `Home` / `End` | 到首 / 尾 |
| `PageUp` / `PageDown` | 移动**选中项**一页（不是只滚 —— 只滚会让高亮停在屏幕外） |
| `→` | 展开当前项的子菜单；已展开时把焦点交给面板第一项 |
| `←` | 收起已展开的子菜单；在子面板里则退出该面板 |
| 打字 | typeahead：首字母跳项、连续字母拼前缀、500ms 后重置、无匹配不动 |
| `Enter` / `空格` | 激活当前项 |
| `Esc` | **只关最内层** |

键盘高亮/焦点项**永远完整可见**：滚动跟随按"扣掉两端提示行"的窗口计算
（`revealMenuItem`），而不是 `scrollIntoView({ block: "nearest" })` —— 后者不知道那里
盖着一条 18px 的提示行。

## 6. 菜单里再开菜单（dismiss 层栈）

参数编辑器工具栏的「⋯」溢出面板里放着 `ReferenceTrackGroup` / `ChildPitchGroup`，
它们各自带一个下拉。里层面板是 portal 到 body 的独立节点，外层判"点在菜单外面"只看
自己的 DOM 子树 —— 于是**外层先关、内层跟着卸载**。

`src/ui/menuDismissStack.ts` 补上 Radix `DismissableLayer` 那一层语义：

> 点在某处时，只有"比包含该处的最深一层更浅"的层被关掉。

即：点在里层菜单里 → 外层不动；点在所有菜单之外 → 全关。`Esc` 走同一条栈，因此一次
按键只关一层。

## 7. 移动形态

| 场景 | 处理 |
|---|---|
| 右键菜单 | 变成底部 sheet（`max-height: 70dvh`，贴底通栏） |
| **住在 sheet 里的菜单**（移动形态的菜单栏） | `data-hs-menu-inline="1"`，保持锚定弹出 —— 否则点一次「视图」会叠出第二层 sheet |
| 子菜单面板 | 同样 `data-hs-menu-inline="1"`：它的坐标由 JS 写在**内联样式**里，sheet 变换会把它们覆盖成"贴底通栏"，子菜单于是跑到屏幕底部 |

判据取**结构事实**（锚点是否住在 `[data-mobile-menu-sheet]` 里 / 是不是子面板），不是
`isMobileForm()` 分支 —— 形态是 CSS 令牌层的一等公民，多一个 JS 分支就等于给"这是哪种
形态"开了第二个答案。

移动 sheet 的选择器必须是 `[data-hs-context-menu="1"]`：该属性有**两种**写法，带 =1 的是
已打开的菜单表面，**无值**是常驻的锚点容器。写成不带值的选择器会把 8 个锚点容器一起
套上 sheet 装饰（实测：radius 16px + shadow 0 -12px + bottom: 0）。

## 8. 新增菜单时怎么写

| 场景 | 用什么 |
|---|---|
| 右键菜单（扁平条目） | `<AppContextMenu x y items onClose />` |
| 工具栏按钮下拉（扁平条目） | `<AppAnchoredMenu anchorRef menuRef items onClose align />` |
| 菜单栏下拉 | `<AppMenuBarItem open onOpenChange items ariaLabel />`（`open` 由菜单栏自己的一个 state 控制，保证同一时刻只开一张） |
| 需要内联控件（滑杆 / 输入框 / 双列） | `<AppContextMenu header>` 或 `AppAnchoredMenu` 的 `children` |
| 子菜单 | 条目上的 `items` 字段（任意层嵌套） |
| 自渲染一个壳 | 必须包一层 `<MenuScrollScope>`（门禁"每个自渲染的菜单表面都包了 MenuScrollScope"） |

条目字段（`AppMenuItemSpec`）里最容易用错的两个：

- `checked` + `radio: true` → `menuitemradio`；`checked` 单独用 → `menuitemcheckbox`。
  互斥档位必须是 radio，否则读屏会念"复选框"。
- `keepOpen: true` → 激活后不收起菜单。**多选**（复选项）必须打开它：勾一次就关掉菜单
  等于让用户重开 N 次。单选与普通条目保持默认收起。

## 9. 门禁

| # | 判据 | 防的是 |
|---|---|---|
| G-1 | `.hs-menu` 必须有 `scrollbar-width: none` | 滚动条回来（删掉不会有任何报错） |
| G-2 | 提示行必须 `position: sticky` + `height: 0` + `::after` 命中区 | 提示行开始占位 |
| G-3 | `.hs-menu` 的 `padding` 必须是 `0` | 提示行内缩、边缘露出没被渐变的空白 |
| G-4 | 移动 sheet 只命中 `[data-hs-context-menu="1"]` | 锚点容器被套上 sheet 装饰 |
| G-5 | `src/` 内不得出现 `<DropdownMenu.` | 两套菜单重新分叉 |
| G-6 | `hs-menu__scroll-hint` 只允许出现在 `MenuScrollScope.tsx` | 第二套滚动提示 |
| — | 每个自渲染的壳都包了 `MenuScrollScope` | 有滚动条却没有任何提示 |

另有两条**开发期探针**（要真浏览器，不在 vitest 里）：

```powershell
cd frontend
# 滚动：几何、提示行可见性、悬停 / 点击 / 长按、键盘跟随
$env:VH="320"; node scripts/menu-scroll-probe.mjs all
# 条目树：迁移前后逐字段对比（回归门禁；基线更新必须显式提交）
node scripts/menu-tree-baseline.mjs compare scripts/menu-tree.baseline.json
```

`menu-tree.baseline.json` 是**迁移前**录的条目树。它故意保持"迁移前"的形态：任何一次
条目改动（增删、重排、改文案、改禁用条件）都必须重新 record 并提交，这样 diff 里就
一定能看见"菜单变了"。

## 10. 不变量

| # | 不变量 |
|---|---|
| I1 | 菜单表面上不存在竖直滚动条（原生与自绘都算） |
| I2 | 高度上限 = 视口预算，不是视口的比例 |
| I3 | 提示行隐藏时零占位、零命中、零重排 |
| I4 | 提示行可见性只由滚动位置决定，1px 以内视为"装得下" |
| I5 | 键盘高亮/焦点项永远完整可见（不被提示行覆盖） |
| I6 | 壳是"测量 → 夹紧"的唯一尺寸来源 |
| I7 | 滚动容器不得引入 `transform` / `filter` / `contain` / `will-change` |
| I8 | 引擎每一步都能在 `prefers-reduced-motion: reduce` 下退化成瞬时跳转 |
| I9 | 触摸形态不依赖悬停事件 |
| I10 | 全仓只有一个菜单表面实现；`src/` 内不得再出现 `DropdownMenu` 菜单 |
| I11 | 每个可滚的菜单表面都渲染提示行 |
