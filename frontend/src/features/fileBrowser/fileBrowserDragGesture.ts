/**
 * 文件浏览器拖拽的手势判定与事件载荷（纯函数）。
 *
 * 【为什么抽出来】拖拽逻辑住在 `FileBrowserPanel` 的一个 effect 里，而面板依赖
 * Redux、Tauri 与 `AudioContext` —— 想验证"右键原地点击算打断、右键按下后移动
 * 20px 不算"就必须渲染整个面板。把判定与载荷构造抽成纯函数后，这些规则可以直接
 * 单测；effect 只负责"读指针事件、调这些函数、派发事件"。
 */

/**
 * 一次指针移动构成拖拽的最小位移（CSS px）。
 *
 * 与 `rightDragContextMenuGuard.RIGHT_DRAG_THRESHOLD_PX` 同值同源：拖拽的激活
 * 阈值与"反向键点击 = 打断"里"点击"的判定阈值是同一个边界（超过它就不是点击）。
 */
export const FILE_DRAG_THRESHOLD_PX = 5;

/** 一次拖拽的源信息（结构类型：面板的 `FileDragState` 在此基础上加坐标与激活位）。 */
export interface FileDragSource {
    filePath: string;
    fileName: string;
    allFilePaths: string[];
    /** 其中是目录的那些路径（拖入时间轴 = 目录导入）。 */
    dirPaths: string[];
    isRightDrag: boolean;
}

/** 拖拽收尾载荷：`drop`（在此处放下）与 `cancel`（放弃）共用同一个形状。 */
export interface FileDragFinishDetail {
    type: "drop" | "cancel";
    filePath: string;
    fileName: string;
    filePaths: string[];
    dirPaths: string[];
    clientX: number;
    clientY: number;
    isRightDrag: boolean;
}

/** 发起本次拖拽的鼠标键（右键拖拽是 2，否则是 0）。 */
export function dragButtonOf(isRightDrag: boolean): 0 | 2 {
    return isRightDrag ? 2 : 0;
}

/**
 * 构造 `drop` / `cancel` 的事件载荷。
 *
 * 【为什么两个 type 共用构造】`cancel` 也必须带全 `dirPaths` —— 接收方据此清掉
 * 目录导入的落点提示与吸附高亮。此前 `onPointerCancel` 走 `drop + canceled`，
 * detail 里漏了 `dirPaths`，接收方拿到的是形状不全的事件。
 */
export function buildFileDragFinishDetail(
    source: FileDragSource,
    type: "drop" | "cancel",
    clientX: number,
    clientY: number,
): FileDragFinishDetail {
    return {
        type,
        filePath: source.filePath,
        fileName: source.fileName,
        filePaths: source.allFilePaths,
        dirPaths: source.dirPaths,
        clientX,
        clientY,
        isRightDrag: source.isRightDrag,
    };
}

/** 反向键按下时的打断候选：位置用于判断"这是点击还是又一次拖动"。 */
export interface DragInterruptCandidate {
    button: number;
    x: number;
    y: number;
}

/**
 * 一次 `pointerdown` 是否构成打断候选。
 *
 * 只在拖拽已越过阈值激活、且按下的是"与发起键相反"的左/右键时记录。返回
 * `null` 表示这不是打断信号 —— 例如按下的是发起键本身（"再加一个键"之外的
 * 中键也在此列）。
 */
export function interruptCandidateOnDown(args: {
    /** 拖拽是否已越过阈值激活。 */
    active: boolean;
    /** 本次拖拽由哪个键发起。 */
    isRightDrag: boolean;
    /** 本次 pointerdown 的键。 */
    button: number;
    x: number;
    y: number;
}): DragInterruptCandidate | null {
    if (!args.active) return null;
    if (args.button !== 0 && args.button !== 2) return null;
    if (args.button === dragButtonOf(args.isRightDrag)) return null;
    return { button: args.button, x: args.x, y: args.y };
}

/**
 * 指针移动到某处后，打断候选是否应当作废。
 *
 * 【为什么基准是候选记录的位置】"点击"的定义是"按下后没怎么动就松开"，
 * 基准是按下那一刻，而不是拖拽起点。
 */
export function interruptCandidateMoved(
    candidate: DragInterruptCandidate,
    x: number,
    y: number,
    thresholdPx: number = FILE_DRAG_THRESHOLD_PX,
): boolean {
    const dx = x - candidate.x;
    const dy = y - candidate.y;
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) return true;
    return dx * dx + dy * dy >= thresholdPx * thresholdPx;
}

/** 一次 `pointerup` 是否构成打断（反向键的那一次点击）。 */
export function isInterruptRelease(
    candidate: DragInterruptCandidate | null,
    button: number,
): boolean {
    return candidate !== null && candidate.button === button;
}
