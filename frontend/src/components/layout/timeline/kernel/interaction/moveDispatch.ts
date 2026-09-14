/**
 * 时间轴渲染内核 · 逐帧手势分派表
 *
 * 【主要内容】
 * 把「当前手势种类」映射为「本次 pointermove 应该调用哪个预览器」的**纯函数**
 * `resolveMoveDispatch()`，以及手势种类联合 `TimelineGestureKind`。
 *
 * 【作用：为什么必须把它抽出来】
 * 宿主 `onGesturePointerMove` 原本是一串 `if (gesture.kind === …) apply…()`。
 * 这种结构**没有任何穷尽性保障**：新增一种手势（或像 `snap-offset-drag` 那样
 * 在阈值升级分支里被创建）时，分派链漏掉一条也不会编译失败——症状是"该控件
 * 拖不动/拖一下就冻结"，而不是报错。
 *
 * 实测事故：吸附偏移（◣）手柄缺失分派分支（9 种手势里唯一缺失的一种），
 * 于是 `applySnapOffsetPreview` 整个文件只有 2 处引用（定义 + 那一处升级调用），
 * 每次拖拽只派发**一次**预览，标记停在跨越 4px 阈值的瞬间。表现即
 * 「对齐标记无法正确的被移动」。
 *
 * 抽成纯函数后，测试可以对**手势种类全集**做表驱动断言：任何新增种类若忘记
 * 登记（返回了不该返回的 null），回归测试立刻失败。宿主侧仍是显式 `switch`，
 * 分派目标一目了然。
 *
 * 【与其他模块的关系】
 * - 上游：`host/timelineKernelHost` 的 `onGesturePointerMove` 每帧调用。
 * - 下游：宿主据返回值调用对应的 `apply*Preview`。
 * - 独立性：纯函数，不依赖 DOM / React / 宿主局部类型，可直接单测。
 *
 * 【维护说明】新增手势时**必须**同时更新 `TimelineGestureKind` 与
 * `resolveMoveDispatch`，并补一条表驱动用例（`moveDispatch.test.ts` 已覆盖全集）。
 */

/**
 * 左键手势种类全集。
 *
 * 与宿主内部的 `Gesture` 判别联合一一对应（宿主按 `kind` 字段取用）。
 * 取值语义：
 * - `none`：无手势（空闲）；
 * - `pending-select`：按下命中 clip、等待「抬起=选中 / 超阈值=拖拽」二选一；
 * - `seek`：按下命中空白，拖拽中持续 seek；
 * - `box-select`：右键框选；
 * - `clip-drag` / `clip-trim` / `clip-fade` / `gain-drag` /
 *   `crossfade-grip` / `snap-offset-drag`：已升级的编辑手势。
 */
export type TimelineGestureKind =
    | "none"
    | "pending-select"
    | "seek"
    | "box-select"
    | "clip-drag"
    | "clip-trim"
    | "clip-fade"
    | "gain-drag"
    | "crossfade-grip"
    | "snap-offset-drag";

/**
 * 逐帧分派目标。
 *
 * `null` 表示**本帧没有逐帧预览可做**，调用方不应调用任何预览器。
 * 取值与宿主内的 `apply*` 函数对应：
 * - `"seek"` → `onSeek`（seek 手势的逐帧语义）；
 * - `"box-select"` → `applyBoxSelect`；
 * - `"drag"` → `applyDragPreview`；
 * - `"trim"` → `applyTrimPreview`；
 * - `"fade"` → `applyFadePreview`；
 * - `"gain"` → `applyGainPreview`；
 * - `"crossfade-grip"` → `applyCrossfadeGripPreview`；
 * - `"snap-offset"` → `applySnapOffsetPreview`。
 */
export type MoveDispatch =
    | "seek"
    | "box-select"
    | "drag"
    | "trim"
    | "fade"
    | "gain"
    | "crossfade-grip"
    | "snap-offset"
    | null;

/**
 * 把当前手势种类解析为本次 pointermove 应执行的预览动作。
 *
 * 流程：显式 `switch` 逐一登记每种已升级手势；`none` 无动作。
 *
 * 特殊说明 1：`pending-select` 返回 `null` 是**刻意的**——它尚未升级，其
 * 「超过位移阈值后升级为具体手势」的逻辑在宿主内单独处理（升级时立即派发一次
 * 预览）。若不返回 `null`，同一帧会既走升级路径又走常规分派，造成重复派发。
 *
 * 特殊说明 2：默认分支返回 `null` 而不是抛错（宿主在渲染热路径上，抛错会中断
 * 整个手势链）；但 `TimelineGestureKind` 是封闭联合，TypeScript 仍能保证所有
 * 已登记种类被覆盖——真正的穷尽性由 `moveDispatch.test.ts` 的表驱动用例守住。
 *
 * @param kind 当前手势种类。
 * @returns 分派目标；无需逐帧预览时为 `null`。
 */
export function resolveMoveDispatch(kind: TimelineGestureKind): MoveDispatch {
    switch (kind) {
        case "seek":
            return "seek";
        case "box-select":
            return "box-select";
        case "clip-drag":
            return "drag";
        case "clip-trim":
            return "trim";
        case "clip-fade":
            return "fade";
        case "gain-drag":
            return "gain";
        case "crossfade-grip":
            return "crossfade-grip";
        case "snap-offset-drag":
            return "snap-offset";
        case "pending-select":
        case "none":
            return null;
        default:
            return null;
    }
}
