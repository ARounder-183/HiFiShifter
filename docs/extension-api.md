# 扩展 API 指南

> 面向为 HiFiShifter 编写扩展控件的作者。
> 本轮**尚未实现插件加载器**（无 manifest 发现、无动态模块加载、无沙箱）。
> 本文档描述的是已经就位的表面 —— 加载器接上后即可直接使用。

---

## 1. 可以做什么

| 能力 | API | 状态 |
|---|---|---|
| 注册一个面板（含停靠/浮动/独立窗口） | `registerPanel` + `setPanelComponent` | ✅ 可用 |
| 往标签右键菜单加条目 | `registerPanelTabMenuItem` | ✅ 可用 |
| 注册一个可绑定命令 | `registerCommand` | ⚠️ 已注册，键位绑定界面尚未消费 |
| 往工具栏加按钮 | `registerToolbarItem` | ⚠️ 已注册，宿主工具栏尚未消费 |
| 面板自有配置的持久化读写 | `getPanelProps` / `setPanelProps` | ✅ 可用 |
| 打开/关闭/浮动面板、套用布局 | `dockApi` 系列 | ✅ 可用 |
| 注册自己的文案 | `registerExtensionMessages` | ✅ 可用 |
| 使用应用 UI 原语 | `@hs/ui` | ✅ 可用 |
| 读主题令牌 | `var(--qt-*)` | ✅ 可用 |
| 调用后端命令 | — | ❌ 暂无权限模型，见 §6 |

`@hs/ui` / `@hs/sdk` 两个导入别名在 `vite.config.ts` 与 `tsconfig.app.json` 中同步声明。

---

## 2. 注册一个面板

```ts
import { registerPanel, setPanelComponent } from "@hs/sdk";
import { MyPanel } from "./MyPanel";

registerPanel({
    id: "myext.inspector",          // ⚠️ 发布后不可更改，见 §5
    titleKey: "panel.myext.title",  // 需要注册文案，见 §4
    icon: MyIcon,                   // 可选，标签栏图标
    defaultWidth: 360,
    defaultHeight: 480,
    minWidth: 240,
    defaultPlacement: { side: "right", sizePx: 360 },
    detachable: true,               // 允许拆到独立窗口
    singleton: true,                // 同名面板只开一个
    order: 60,                      // Window 菜单里的排序权重
});

setPanelComponent("myext.inspector", MyPanel);
```

面板组件收到：

```ts
interface DockPanelProps {
    formId: string;                  // 本窗体的唯一 id（同一面板可有多个实例）
    panelId: string;                 // 面板 id
    props: Record<string, unknown>;  // 你持久化的配置，见 §3
}
```

体积大的面板用 `React.lazy` 交给注册中心，渲染侧已包好 `Suspense`：

```ts
setPanelComponent("myext.inspector", lazy(() => import("./MyPanel")));
```

**注册面板不需要改任何宿主代码**：Window 菜单会遍历注册表，你的面板自动出现。

---

## 3. 持久化面板配置

`props` 随布局一起进存档，不必自建存储：

```tsx
function MyPanel({ panelId, props }: DockPanelProps) {
    const dispatch = useAppDispatch();
    const getState = useStore().getState;   // 或经 useAppSelector 读取
    const columns = (props.columns as number) ?? 3;

    return (
        <AppIconButton
            icon={<GridIcon />}
            tooltip="3 columns"
            onClick={() => setPanelProps(dispatch, getState, panelId, { columns: 4 })}
        />
    );
}
```

写入是**浅合并**：只传要改的键。要删除某个键，显式传 `undefined`。

> 同一面板有多个实例（`singleton: false`）时用 `setFormPropsById(dispatch, formId, props)`，
> 因为 `setPanelProps` 按面板 id 只取第一个窗体。

---

## 4. 注册自己的文案

内置词典是**封闭联合**（`MessageKey` 由 en-US 推导，五语系缺键即编译失败），
第三方键不在其中，因此走运行时通道：

```ts
import { registerExtensionMessages } from "@hs/sdk";

const dispose = registerExtensionMessages({
    "en-US": { "panel.myext.title": "Inspector" },
    "zh-CN": { "panel.myext.title": "检查器" },
    "ja-JP": { "panel.myext.title": "インスペクタ" },
});
```

组件里用 `tf`（无类型翻译）：

```tsx
const { tf } = useI18n();
<span>{tf("panel.myext.title")}</span>;
```

**解析顺序**：`静态词典 → 扩展层 → en-US → 键名`。
扩展层排在静态词典之后是**有意的**：第三方只能新增键，不能改写内置文案
（否则注册一个 `ok` 就能改掉全应用的「确定」按钮）。这条顺序有测试守着。

**卸载时必须调用 `dispose()`**，否则你的文案会一直留在内存里。

---

## 5. 稳定性与兼容

- **面板 id 一旦发布就不可更改**。它写进用户的布局存档；`dockSchema` 的归一化会
  丢弃"面板已不存在"的记录，也就是用户升级后你改了 id，他们的面板会被静默移除。
- `SDK_VERSION` 是**契约版本**，独立于应用版本。破坏性变更会递增它。
- `src/sdk/index.ts` 导出的类型是契约；未导出的内部类型随时可变。
- 面板崩溃会被 `PanelErrorBoundary` 隔离，不影响其他面板；但**面板之外**的崩溃
  会由根级 `AppRootErrorBoundary` 兜住（整窗降级为错误页，而不是白屏）。

---

## 6. 目前**不**具备的（不要依赖）

| 缺失 | 影响 | 备注 |
|---|---|---|
| 插件加载器 | 扩展需自行设法把代码注入页面 | 本轮只做前置件 |
| 权限模型 / 沙箱 | 扩展拥有完整写权限（含 `saveProject` 等破坏性命令） | 见方案 §3.5 E9 |
| 后端命令扩展 | 不能新增 Tauri 命令 | `invoke.wiring.test.ts` 会校验命令契约 |
| 键位绑定界面消费 `registerCommand` | 命令注册了但用户看不到绑定项 | 待接入 |
| 宿主工具栏消费 `registerToolbarItem` | 同上 | 待接入 |
| 第三方文案的编译期检查 | `tf` 是字符串键，拼错只会显示键名 | 有意为之，见 §4 |

---

## 7. 视觉一致性

扩展有两条路，**都受支持**：

1. **用 `@hs/ui` 的原语**（推荐）：`AppButton` / `AppIconButton` / `AppDialog` /
   `AppContextMenu` / `AppField` / `AppListRow` / `AppEmptyState` / `AppStatusChip`。
   它们已把全应用的交互协议收口 —— 用 `AppDialog` 就自动获得 Enter 确认、
   Esc 护栏、快捷键抑制、高度上限与统一按钮顺序。

2. **自造控件，但消费 `--qt-*` 令牌**。令牌是唯一允许被外部直接读取的层，
   且用户的自定义主题就是靠覆写这些变量生效的，因此只要用 `var(--qt-*)`
   就自动跟随主题：

```css
.my-widget {
    background: var(--qt-panel);
    color: var(--qt-text);
    border: 1px solid var(--qt-border);
    border-radius: var(--qt-radius-md);
    padding: var(--qt-space-4);
    font-size: var(--qt-fs-md);
}
```

**不要**硬编码颜色或像素值：应用里的暗/亮主题与用户自定义主题都只改令牌，
硬编码的值不会跟着变。

可用令牌清单见 `src/index.css` 的「度量令牌」注释块，以及
`themeTypes.ts` 的 `QT_COLOR_TOKENS`（颜色，已随用户主题导出 JSON 持久化，
因此**名称不可更改**）。
