/*
 * 停靠窗体系统的类型定义。
 *
 * 【布局模型为什么是二叉树 + 标签组】每个容器矩形要么被一条分割线切成两半
 * （`split`），要么是一组以标签页共享同一矩形的窗体（`tabset`）。这恰好覆盖
 * REAPER / VEGAS / VS Code 用户的心智模型：任意嵌套的分割 + 任意合并的标签页，
 * 而模型本身小到可以整体序列化、整体校验、整体复原。
 *
 * 【为什么窗体状态与布局树分离】树只记录"哪些窗体在哪个标签组里、顺序如何"，
 * 窗体的身份、标题、浮动几何、面板私有 props 都在 `forms` 里。这样：
 * - 关闭一个窗体 = 把它从树里摘掉，`forms` 记录保留 → 重开时标题与 props 还在；
 * - 浮动 = 树里没有它，`float` 有几何 → 树操作完全不需要知道浮动的存在；
 * - 可见性是**派生**的（在树里 ∨ 有浮动几何），不存冗余布尔值，也就不会出现
 *   "树说它在 A 组、标志位说它隐藏"这种自相矛盾的状态。
 *
 * 【为什么布局树是持久化格式本身】`DockLayout` 直接就是落盘的 JSON —— 未来
 * 开放 API 时，用户拿到的那份 JSON 与内部状态是同一份，不需要第二套 schema
 * 与随之而来的转换/漂移问题。
 */

/** 分割方向：`row` = 左右并排（分割线竖直），`col` = 上下堆叠（分割线水平）。 */
export type DockSplitDir = "row" | "col";

/** 停靠落点：四边拆分 + 中央并入标签组；`float` 表示不落任何 Zone。 */
export type DockDropZone = "center" | "left" | "right" | "top" | "bottom" | "float";

/**
 * 打开面板时的默认落点。
 *
 * 定义在 `dockTypes`（纯类型、零依赖）而不是 `dockSchema`：面板注册表要引用
 * 它，而注册表又被 schema 引用 —— 放在 schema 会形成循环。
 */
export interface DockPlacement {
    side: "left" | "right" | "top" | "bottom" | "center";
    /** 拆出新组时该组的固定像素尺寸（`center` 时无效）。 */
    sizePx?: number;
    /** 优先并入该面板所在的标签组（若它当前可见）。 */
    tabWith?: string;
}

/** 矩形（视口坐标，逻辑像素）。 */
export interface DockRect {
    x: number;
    y: number;
    w: number;
    h: number;
}

export interface DockSplitNode {
    t: "split";
    id: string;
    dir: DockSplitDir;
    /** 侧 A 的占比（0..1，已钳制）。仅在 `fixed === null` 时生效。 */
    ratio: number;
    /**
     * 非空 = 某一侧固定像素尺寸，另一侧吸收窗口缩放带来的全部变化。
     *
     * 右侧栏这类"内容宽度与窗口无关"的容器必须固定像素，否则窗口变宽时它
     * 跟着变宽，用户看到的宽度与拖出来的不一致。做成 `{side, px}` 而不是
     * 只固定侧 A，是因为右侧停靠栏（文件浏览器/记事本）恰恰是固定侧 B 的
     * 典型场景，把侧 B 也纳入后无需为了"固定在右边"而人为翻转 a/b 顺序。
     */
    fixed: { side: "a" | "b"; px: number } | null;
    a: DockNode;
    b: DockNode;
}

export interface DockTabsetNode {
    t: "tabset";
    id: string;
    /** 窗体 id，顺序即标签顺序。 */
    tabs: string[];
    /** 活动标签（必须是 `tabs` 成员；规范化时修正）。 */
    active: string;
    /**
     * 折叠为一条标签条（REAPER 底部 docker 行为）：只渲染标签条，内容区收起。
     * 这是"界面拥挤"最彻底的解法 —— 比缩小尺寸保留更多可用空间。
     */
    collapsed?: boolean;
    /** 折叠后标签条的尺寸（px）；缺省取该组的默认值。 */
    collapsedPx?: number;
}

export type DockNode = DockSplitNode | DockTabsetNode;

/**
 * 浮动窗体的锚点：位置由主窗口尺寸推导，而不是写死坐标。
 *
 * 【为什么需要】"默认落在右下角"这类意图在**布局创建时**并不知道真实的窗口尺寸
 * （窗口几何是异步恢复的，创建那一刻量到的 innerHeight 可能还是初始值），写死坐标
 * 会让浮窗停在偏高的位置；窗口尺寸变化后也会跑偏。把锚点作为语义存下来、在**渲染
 * 时**按当前视口解析，位置就总是对的。用户一旦手动移动/缩放浮窗，锚点即被清除
 * （见 `setFloatGeometry`），此后它就是一个普通的固定位置。
 */
export type DockFloatAnchor = "bottom-right";

/** 浮动窗体的几何与状态。 */
export interface DockFloatGeometry {
    /** 未使用锚点时是绝对坐标；使用锚点时由 `resolveFloatRect` 推导，此处为占位值。 */
    x: number;
    y: number;
    w: number;
    h: number;
    /** 非空 = 位置随主窗口尺寸推导（见 `DockFloatAnchor`）。 */
    anchor?: DockFloatAnchor | null;
    /** 锚点距视口边缘的间距（px）。 */
    anchorMarginPx?: number;
    /** 最大化（铺满主窗口可用区），保留原几何以便还原。 */
    maximized?: boolean;
    /** 最大化前的几何快照。 */
    restore?: { x: number; y: number; w: number; h: number } | null;
    /** 最小化为标题条。 */
    minimized?: boolean;
}

/** 一个窗体实例。 */
export interface DockForm {
    id: string;
    /** 指向面板注册表；一个面板可有多个窗体实例。 */
    panelId: string;
    /** 用户重命名后的标题；缺省用面板的 i18n 标题。 */
    title?: string;
    /**
     * **记住的**浮动几何。
     *
     * 【为什么它在停靠时也要保留】停靠态与浮动态的尺寸是两个独立的意图：
     * 用户把小浮窗拖进一大片区域，停靠后它会撑大；再拆下来时应当回到"他当初
     * 把浮窗调成的大小"，而不是继承刚停靠时那片区域的尺寸。因此本字段是**记忆**，
     * 不是"当前是否浮动"的判据 —— 后者是下面的 `floating`。
     */
    float?: DockFloatGeometry | null;
    /**
     * 当前是否处于浮动状态。
     *
     * 与 `float`（几何记忆）分开，是为了让"停靠 ⇄ 浮动"不再丢失浮窗尺寸。
     */
    floating?: boolean;
    /**
     * 浮动形态：进程内浮层（默认）还是**独立操作系统窗口**。
     *
     * 【为什么是逐窗体】拆到独立窗口的面板会被重新挂载（跨窗口无法搬 DOM），
     * 因此只有声明了 `detachable` 的面板才可能拿到 `"osWindow"`；其它面板即使
     * 被请求也保持 `"inApp"`（见 `floatFormDetached`）。
     */
    floatMode?: DockFloatMode;
    /**
     * 独立窗口在**屏幕坐标**下的位置（`floatMode === "osWindow"` 时有效）。
     *
     * 尺寸仍用 `float.w/h`（浮窗尺寸的语义两处一致）；位置不能用 `float.x/y`
     * —— 那是主窗口视口坐标，跨窗口后毫无意义。
     */
    floatScreen?: { x: number; y: number } | null;
    /** 面板私有状态（随布局持久化，未来 API 面板可直接受益）。 */
    props?: Record<string, unknown>;
}

/**
 * 用户可调的"内嵌沟槽"尺寸（面板内部的固定宽/高分区）。
 *
 * 【为什么只有轨道头一项】参数编辑器左侧琴键轴的宽度在 WebGL 内核里是
 * **构造期烘焙**的（`pianoRollKernelHost` 建宿主时读一次 `axisWidthPx`，
 * 之后按它算键盘几何、刻度标签与 GL 轴画布尺寸）。要让它可调，就得把该值
 * 改成每帧现读的 getter 并串进场景重建 —— 而宿主的生命周期约束是"一旦重建，
 * 滚动位置与手势状态静默归零"。为一个次要项去动最复杂的文件、承担这个风险
 * 不划算，因此这里只保留真正需要的轨道头宽度。将来若要加，正确的做法是把
 * 轴宽改成宿主内的活值 + 场景失效，而不是重建宿主。
 */
export interface DockGutterSizes {
    /** 时间轴左侧轨道头宽度（取代原先写死的 `w-64` = 256px）。 */
    timelineTrackHeaderPx: number;
}

/** 一份完整的布局（持久化格式 = 运行时格式）。 */
/**
 * 标签行在窗体中的位置。
 *
 * 默认 `"bottom"`：标签是"这个窗格里放了什么"的身份说明，放在内容下方更贴近
 * 用户直觉（与浏览器标签、大多数 DAW 的 docker 一致），也不会在视觉上压住内容
 * 的顶部工具条。
 */
export type DockTabPosition = "top" | "bottom";

/** 浮动形态（见 `DockForm.floatMode`）。 */
export type DockFloatMode = "inApp" | "osWindow";

export interface DockLayout {
    /** schema 版本号，用于迁移。 */
    schema: number;
    tree: DockNode;
    forms: Record<string, DockForm>;
    /** 窗体创建顺序（稳定顺序，用于挂载顺序与浮动 z 序的基线）。 */
    order: string[];
    /** 浮动窗体 z 序，末尾为最上层。 */
    floatOrder: string[];
    gutters: DockGutterSizes;
    /** 标签行的位置（见 `DockTabPosition`）。 */
    tabPosition: DockTabPosition;
    /** 用户命名预设。 */
    presets?: Record<string, DockPreset>;
    activePreset?: string | null;
}

/** 命名预设只存"排布"，不存面板私有 props —— 预设的语义是换一套工作区，不是换内容。 */
export interface DockPreset {
    name: string;
    tree: DockNode;
    forms: Record<string, DockForm>;
    order: string[];
    floatOrder: string[];
    gutters: DockGutterSizes;
    /** 创建时间（Unix 毫秒），用于管理界面排序。 */
    createdAtMs: number;
}

/** 当前布局的 schema 版本。 */
export const DOCK_LAYOUT_SCHEMA = 1;

/** 停靠区最小尺寸（低于此值不允许再分割/收缩）。 */
export const DOCK_MIN_ZONE_PX = { w: 120, h: 90 };

/** 标签条高度。 */
export const DOCK_TAB_BAR_PX = 26;

/** 分隔条厚度与热区。 */
export const DOCK_SPLITTER_PX = 4;
export const DOCK_SPLITTER_HIT_PX = 9;

/** 折叠标签条的默认尺寸。 */
export const DOCK_COLLAPSED_PX = 26;
