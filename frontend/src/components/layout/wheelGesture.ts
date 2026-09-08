/**
 * wheelGesture.ts - 滚轮/触摸板手势 → 动作语义解析。
 *
 * 主要内容：
 * - 区分鼠标滚轮、触摸板的轴向特征（getWheelGestureAxis、isLikelyTouchpadWheelGesture）。
 * - 将原始 deltaX/deltaY + 修饰键（zoom/scroll/pan request）映射到具体动作：
 *   getParamEditorWheelAction（参数编辑器）、getTimelineWheelAction（Timeline）、
 *   getVibratoDragWheelTarget（颤音拖拽幅度/频率）。
 *
 * 与其他模块的关系：
 * - 被 PianoRollPanel.tsx、usePianoRollInteractions.ts、TimelineScrollArea.tsx 调用，
 *   将解析得到的动作分发给具体的 scroll/zoom 实现。
 *
 * 维护说明：
 * - "horizontal/vertical scroll request" 与 "free-scroll"（双轴自由滚动）必须明确区分；
 *   首条 free-scroll 分支只能用 `&&`（同时按下），不能用 `||`（任一按下），否则会
 *   吞掉单独的横向/纵向快捷键，造成"快捷键有时失效"。详见各函数头部注释。
 */

const WHEEL_AXIS_EPSILON = 0.5;
const VIBRATO_TOUCHPAD_DELTA_THRESHOLD = 220;
const VIBRATO_TOUCHPAD_FREQUENCY_AXIS_RATIO = 0.75;

/** 指针悬停的原生滚动条区域（滚动条悬停语境下滚轮语义只归属该轴）。 */
export type ScrollbarZone = "vertical" | "horizontal";

function isLikelyDiscreteWheelStep(absDelta: number): boolean {
    const rounded = Math.round(absDelta);
    if (Math.abs(absDelta - rounded) > WHEEL_AXIS_EPSILON) {
        return false;
    }

    return rounded % 100 === 0 || rounded % 120 === 0;
}

export type ParamEditorWheelAction =
    | "free-scroll"
    | "horizontal-scroll"
    | "vertical-pan"
    | "vertical-zoom"
    | "horizontal-zoom";

export type TimelineWheelAction =
    | "free-scroll"
    | "horizontal-scroll"
    | "vertical-scroll"
    | "vertical-zoom"
    | "horizontal-zoom"
    | "native";

export type VibratoDragWheelTarget = "amplitude" | "frequency" | "none";

function isLikelyTouchpadWheelGesture(input: {
    deltaX: number;
    deltaY: number;
    deltaMode: number;
}): boolean {
    const absX = Math.abs(input.deltaX);
    const absY = Math.abs(input.deltaY);

    if (absX > WHEEL_AXIS_EPSILON) {
        return true;
    }

    if (input.deltaMode !== 0) {
        return false;
    }

    if (absY <= WHEEL_AXIS_EPSILON) {
        return false;
    }

    if (!Number.isInteger(input.deltaY)) {
        return true;
    }

    if (isLikelyDiscreteWheelStep(absY)) {
        return false;
    }

    return absY <= VIBRATO_TOUCHPAD_DELTA_THRESHOLD;
}

export function getVibratoDragWheelTarget(input: {
    deltaX: number;
    deltaY: number;
    deltaMode: number;
    amplitudeRequested: boolean;
    frequencyRequested: boolean;
}): VibratoDragWheelTarget {
    const absX = Math.abs(input.deltaX);
    const absY = Math.abs(input.deltaY);

    if (absX <= WHEEL_AXIS_EPSILON && absY <= WHEEL_AXIS_EPSILON) {
        return "none";
    }

    // Touchpad gestures do not require modifiers while dragging with line/vibrato tool.
    if (isLikelyTouchpadWheelGesture(input)) {
        if (absX > WHEEL_AXIS_EPSILON && absX >= absY * VIBRATO_TOUCHPAD_FREQUENCY_AXIS_RATIO) {
            return "frequency";
        }
        return "amplitude";
    }

    if (input.frequencyRequested) {
        return "frequency";
    }

    if (input.amplitudeRequested) {
        return "amplitude";
    }

    return "none";
}

export function getWheelGestureAxis(input: {
    deltaX: number;
    deltaY: number;
}): "horizontal" | "vertical" {
    const absX = Math.abs(input.deltaX);
    const absY = Math.abs(input.deltaY);

    if (absX > WHEEL_AXIS_EPSILON && absX > absY) {
        return "horizontal";
    }

    return "vertical";
}

/**
 * 悬停滚动条语境的滚轮语义：滚轮只作用于该滚动条的轴。
 *
 * - 无修饰键：滚轮 = 该轴滚动（竖直滚动条 → 滚动；水平滚动条 → 水平滚动）；
 * - 按住 modifier.scrollbarZoom（默认 Alt）：滚轮 = 该轴缩放。
 *
 * 该判定优先于一切按键绑定解析 —— 悬停滚动条时用户意图明确指向该轴，
 * 不应再被"默认滚轮 = 水平缩放"之类的全局绑定覆盖。
 */
function scrollbarZoneAction<T extends string>(
    zone: ScrollbarZone,
    zoomRequested: boolean,
    verticalScrollAction: T,
): T {
    if (zone === "vertical") {
        return (zoomRequested ? "vertical-zoom" : verticalScrollAction) as T;
    }
    return (zoomRequested ? "horizontal-zoom" : "horizontal-scroll") as T;
}

/**
 * 解析 ParamEditor 的滚轮手势动作。
 *
 * 流程：
 * 0. 悬停在原生滚动条上（scrollbarZone）→ 滚轮只归属该轴（滚动 / 缩放）。
 * 1. 横向滚动 + 纵向 pan 同时按下 → free-scroll（双轴自由滚动）。
 * 2. 单独按下显式快捷键时直接命中对应分支。
 * 3. 没有显式请求时，回退到基于 deltaX/deltaY 主轴的轴向判断；
 *    水平占主导时走 horizontal-scroll，否则兜底 horizontal-zoom（保留原默认行为）。
 *
 * 历史问题：原实现首条分支为 `horizontalScrollRequested || verticalPanRequested
 * → free-scroll`，导致只按横向滚动键时也被错误判定为 free-scroll，且后续
 * `horizontal-scroll` 分支变成死代码，进而引发"横向滚动快捷键有时失效"
 * （Bug 修复，2026-06-30）。
 */
export function getParamEditorWheelAction(input: {
    deltaX: number;
    deltaY: number;
    horizontalScrollRequested: boolean;
    verticalPanRequested: boolean;
    verticalZoomRequested: boolean;
    horizontalZoomRequested: boolean;
    /** 指针悬停的原生滚动条区域；悬停时优先于一切按键绑定。 */
    scrollbarZone?: ScrollbarZone | null;
    /** 悬停滚动条时是否按下了 modifier.scrollbarZoom 缩放修饰键。 */
    scrollbarZoomRequested?: boolean;
}): ParamEditorWheelAction {
    if (input.scrollbarZone) {
        return scrollbarZoneAction(
            input.scrollbarZone,
            Boolean(input.scrollbarZoomRequested),
            "vertical-pan",
        );
    }

    // 双轴自由滚动：仅当横向滚动 + 纵向 pan 同时按下
    if (input.horizontalScrollRequested && input.verticalPanRequested) {
        return "free-scroll";
    }

    // 单独的显式快捷键各自命中
    if (input.horizontalScrollRequested) {
        return "horizontal-scroll";
    }

    if (input.verticalPanRequested) {
        return "vertical-pan";
    }

    if (input.verticalZoomRequested) {
        return "vertical-zoom";
    }

    if (input.horizontalZoomRequested) {
        return "horizontal-zoom";
    }

    // 没有显式请求：根据 deltaX/deltaY 主轴回退判断
    const axis = getWheelGestureAxis(input);
    if (axis === "horizontal") {
        return "horizontal-scroll";
    }

    return "horizontal-zoom";
}

/**
 * 解析 Timeline 的滚轮手势动作。
 *
 * 流程：
 * 0. 悬停在原生滚动条上（scrollbarZone）→ 滚轮只归属该轴（滚动 / 缩放）。
 * 1. 横向滚动 + 纵向滚动快捷键同时按下 → free-scroll（双轴自由滚动）。
 * 2. 单独按下显式快捷键时直接命中对应分支。
 * 3. 没有显式请求时，回退到基于 deltaX/deltaY 主轴的轴向判断；
 *    水平占主导时走 horizontal-scroll，否则兜底 native（让浏览器原生滚动）。
 *
 * 历史问题同 `getParamEditorWheelAction`：原首条分支误把单独的横向滚动快捷键
 * 吞入 free-scroll，导致 horizontal-scroll 分支永远不会命中
 * （Bug 修复，2026-06-30）。
 */
export function getTimelineWheelAction(input: {
    deltaX: number;
    deltaY: number;
    horizontalScrollRequested: boolean;
    verticalScrollRequested: boolean;
    verticalZoomRequested: boolean;
    horizontalZoomRequested: boolean;
    /** 指针悬停的原生滚动条区域；悬停时优先于一切按键绑定。 */
    scrollbarZone?: ScrollbarZone | null;
    /** 悬停滚动条时是否按下了 modifier.scrollbarZoom 缩放修饰键。 */
    scrollbarZoomRequested?: boolean;
}): TimelineWheelAction {
    if (input.scrollbarZone) {
        return scrollbarZoneAction(
            input.scrollbarZone,
            Boolean(input.scrollbarZoomRequested),
            "vertical-scroll",
        );
    }

    // 双轴自由滚动：仅当横向滚动 + 纵向滚动同时按下
    if (input.horizontalScrollRequested && input.verticalScrollRequested) {
        return "free-scroll";
    }

    // 单独的显式快捷键各自命中
    if (input.horizontalScrollRequested) {
        return "horizontal-scroll";
    }

    if (input.verticalScrollRequested) {
        return "vertical-scroll";
    }

    if (input.verticalZoomRequested) {
        return "vertical-zoom";
    }

    if (input.horizontalZoomRequested) {
        return "horizontal-zoom";
    }

    // 没有显式请求：根据 deltaX/deltaY 主轴回退判断
    const axis = getWheelGestureAxis(input);
    if (axis === "horizontal") {
        return "horizontal-scroll";
    }

    return "native";
}
