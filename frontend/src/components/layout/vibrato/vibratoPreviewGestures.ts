/**
 * 颤音预览画布的手势命中与映射（纯函数，可单测）。
 *
 * 【设计原则】预览画布在编辑时就是**另一组滑杆**：拖动的是与参数表单同一个草稿，
 * 不是独立的第二事实源。因此这里的映射规则必须与渲染端一致 ——
 * 渐入 / 渐出与渲染端同样以**整段时长**为上限（见 `vibratoCurve.ts` 的包络段），
 * 深度 / 相位的换算是绘制所用几何的逆运算。
 *
 * 【为什么单独成文件】命中测试与换算全是可以脱离 React 验证的算术；混在画布里
 * 就只能靠手感回归。"拖不动 / 拖反了"这类缺陷不会抛错，只能靠单测钉住。
 */

import { advanceFineAxisDrag, createFineAxisDragState } from "../../../utils/fineAxisDrag";
import type { FineAxisDragState } from "../../../utils/fineAxisDrag";
import { VIBRATO_LIMITS } from "../../../features/vibrato/vibratoPresets";
import type { VibratoPreset } from "../../../features/vibrato/vibratoTypes";

/** 手势区域：左右缘手柄，或波形主体。 */
export type PreviewZone = { kind: "attack" } | { kind: "release" } | { kind: "body" };

/** 手柄的归一化横向位置（0..1）。 */
export interface PreviewHandleLayout {
    attackFrac: number;
    releaseFrac: number;
}

/** 手柄的命中半径（CSS 像素）。 */
export const HANDLE_HIT_PX = 9;
/** 外缘兜底命中带占画布宽度的比例（手柄被拖到别处时的后备抓手）。 */
export const EDGE_ZONE_FRAC = 0.12;

/**
 * 命中测试：**先手柄、后外缘、最后主体**。
 *
 * 【为什么手柄优先】手柄画在包络斜坡的实际位置（渐入很长时可能到画布中部），
 * 若只按"左右 12%"判定，用户抓住看得见的手柄却落在主体区 —— 拖出的是相位而不是
 * 渐入。手柄附近优先保证"抓哪个就是哪个"；外缘 12% 是手柄被拖远后的后备。
 *
 * 【两个手柄重叠时取更近的那个】渐入 / 渐出各自都能拖满整条线，因此它们完全可能
 * 落在同一处（例如渐入拉满、渐出归零，两个手柄都贴到右缘）。此时"渐入优先"会让
 * 渐出手柄永远抓不到 —— 按距离取近的，只有真正等距（同一个像素）时才退回渐入。
 */
export function hitTestPreviewZone(
    x: number,
    width: number,
    layout: PreviewHandleLayout,
): PreviewZone {
    // 画布尚未布局（宽度为 0）时没有可抓手柄，一律当主体 —— 也避免用 1px 的
    // 假宽度去比 9px 的命中半径（那会把整条轴都判成手柄）。
    if (!(width > 0)) return { kind: "body" };
    const attackDist = Math.abs(x - layout.attackFrac * width);
    const releaseDist = Math.abs(x - layout.releaseFrac * width);
    const attackHit = attackDist <= HANDLE_HIT_PX;
    const releaseHit = releaseDist <= HANDLE_HIT_PX;
    if (attackHit || releaseHit) {
        return releaseHit && releaseDist < attackDist ? { kind: "release" } : { kind: "attack" };
    }
    if (x < width * EDGE_ZONE_FRAC) return { kind: "attack" };
    if (x > width * (1 - EDGE_ZONE_FRAC)) return { kind: "release" };
    return { kind: "body" };
}

/** 区域对应的鼠标指针形状。 */
export function cursorForZone(zone: PreviewZone): string {
    return zone.kind === "body" ? "move" : "ew-resize";
}

/**
 * 一次手势的起点快照。
 *
 * 【为什么必须快照】主体拖动的纵向换算依赖画布**当前的**纵轴定标，而深度一改，
 * 定标（按峰值自适应）也跟着变 —— 若每帧都重新取标尺，拖动会变成非线性甚至
 * 反向。起点取一次，整段手势按同一套几何走。
 */
export interface PreviewGestureSnapshot {
    attackMs: number;
    releaseMs: number;
    depthCents: number;
    startPhaseDeg: number;
    /** 可见窗口时长（ms）。 */
    windowMs: number;
    /** 画布宽度（CSS 像素）。 */
    widthPx: number;
    /** 一个可见周期的像素宽度（相位换算用）。 */
    cycleWidthPx: number;
    /**
     * 纵向**每像素对应多少 cents**（与绘制同一套换算）。
     *
     * 【单位陷阱】它是 cents/px，所以"像素位移 → cents 位移"要**乘**它，不是除。
     * 反过来用会让灵敏度随深度反比变化：深度越小、每像素走过的 cents 越多，
     * 拖一点点就把幅度拉飞。
     */
    centsPerPx: number;
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/** 相位取模到 [0,360)。 */
export function wrapPhaseDeg(value: number): number {
    const wrapped = value % 360;
    const positive = wrapped < 0 ? wrapped + 360 : wrapped;
    // `-0` 与 `0` 数值相等但 `Object.is` 不等；归一成 `0` 以免下游断言 / 显示出现 "-0°"。
    return Object.is(positive, -0) ? 0 : positive;
}

/**
 * 由手势位移求出要写进草稿的字段。
 *
 * | 区域 | 映射 |
 * | - | - |
 * | 渐入 | 水平位移 → `attackMs`（`dx/width × windowMs`），钳 `0..windowMs` |
 * | 渐出 | 同上但**取反**（手柄画在斜坡起点，向右拖即缩短渐出）→ `releaseMs` |
 * | 主体 | `dx → startPhaseDeg`（一个可见周期 = 360°，**取反**使波形跟手）；`dy → depthCents`（向上加深） |
 *
 * 【三个水平映射里有两个是反的，这不是笔误】符号一律由"看得见的东西往哪动"决定：
 * 手柄 / 波形都要跟着指针走。渐出与相位的字段定义方向恰好与屏幕方向相反，于是取反。
 *
 * 【渐入 / 渐出为什么以整段时长为上限】它们各自都能铺满整条颤音线：渐入拉满 =
 * 整条线由弱到强，渐出拉满 = 整条线由强到弱。上限曾经是"时长的一半"（为了让
 * 两者不互相吃掉），但那让手柄只能拖到中线，用户没法做一条全程渐强的颤音。
 * 两者重叠时渲染端把两段增益**相乘**（见 `vibratoCurve.ts`），手柄位置与包络
 * 斜坡起点始终一一对应 —— 这正是"手柄就是那条斜坡"的直观语义。
 *
 * 纵向是**与画布 1:1** 的：画布把深度按自身峰值放大，`centsPerPx` 正是那套标尺的
 * 斜率，所以"把波峰拖到中线"恰好把深度拖到 0。这正是用户对"波形就是旋钮"的预期。
 */
export function applyPreviewGesture(
    zone: PreviewZone,
    snapshot: PreviewGestureSnapshot,
    deltaX: number,
    deltaY: number,
): Partial<VibratoPreset> {
    const width = snapshot.widthPx > 0 ? snapshot.widthPx : 1;
    const maxSpan = Math.max(1, snapshot.windowMs);
    const msPerPx = snapshot.windowMs / width;

    if (zone.kind === "attack") {
        return { attackMs: clamp(snapshot.attackMs + deltaX * msPerPx, 0, maxSpan) };
    }
    if (zone.kind === "release") {
        // 渐出手柄画在 `windowMs - releaseMs` 处（渐出斜坡的起点），因此**向右拖**
        // 是让斜坡起点右移、渐出变**短**。映射取负号，手柄才会跟着指针走 —— 与
        // 渐入手柄同向会得到"往右拖、手柄却往左跑"的反直觉手感。
        return { releaseMs: clamp(snapshot.releaseMs - deltaX * msPerPx, 0, maxSpan) };
    }

    const cycleWidthPx = snapshot.cycleWidthPx > 0 ? snapshot.cycleWidthPx : width;
    /*
     * 相位：向右拖 → 波形**向右**走（跟手）。
     *
     * 【为什么是减号】波形取 `sampleCycle(u + φ)`，`φ` 变大意味着同一个波形特征
     * （比如波峰）出现在**更早**的时刻 —— 画面上整条波形向左跑。而用户是"抓住波形
     * 往右拖"，所以映射必须取负号，让画出来的东西跟着指针走。
     *
     * 与渐出手柄同一个理由（见上）：符号由"看得见的东西往哪动"决定，不由字段名的
     * 字面方向决定。
     */
    const phase = wrapPhaseDeg(snapshot.startPhaseDeg - (deltaX / cycleWidthPx) * 360);
    const centsPerPx = snapshot.centsPerPx > 0 ? snapshot.centsPerPx : 1;
    // 像素 × (cents/像素) = cents。向上拖（deltaY < 0）即加深。
    const depthCents = clamp(
        snapshot.depthCents - deltaY * centsPerPx,
        VIBRATO_LIMITS.depthCents.min,
        VIBRATO_LIMITS.depthCents.max,
    );
    return { startPhaseDeg: phase, depthCents };
}

/** 一次预览手势里「精细调整」的累计状态：两轴各一份。 */
export interface PreviewFineDragState {
    x: FineAxisDragState;
    y: FineAxisDragState;
}

/** 建立手势的精细调整状态（喂进去的是"相对按下点的累计位移"，故起点为 0）。 */
export function createPreviewFineDragState(fineActive: boolean): PreviewFineDragState {
    return {
        x: createFineAxisDragState(0, fineActive),
        y: createFineAxisDragState(0, fineActive),
    };
}

/**
 * 把画布上报的**累计原始位移**换算成**累计的已缩放位移**。
 *
 * 【为什么必须按增量累计，而不是"总位移 × 当前比例"】后者在拖拽途中按下 / 松开
 * 「精细调整」时会把已经累计的位移整体重算 —— 数值瞬间跳回去，用户看到的是
 * "闪回"，正在进行的拖拽被打断。按增量缩放则让比例变化只影响**此后**每帧走多少，
 * 累计量始终连续。规则与时间轴的增益旋钮同源（见 `utils/fineAxisDrag.ts`）。
 *
 * @param state 手势期间持续复用的状态（就地更新）。
 * @param deltaX 相对手势起点的累计水平位移（画布原始上报）。
 * @param deltaY 同上，垂直。
 * @param fineActive 本帧精细调整是否生效。
 */
export function advancePreviewFineDrag(
    state: PreviewFineDragState,
    deltaX: number,
    deltaY: number,
    fineActive: boolean,
): { deltaX: number; deltaY: number } {
    return {
        deltaX: advanceFineAxisDrag(state.x, deltaX, fineActive),
        deltaY: advanceFineAxisDrag(state.y, deltaY, fineActive),
    };
}

/**
 * 求一个可见周期的像素宽度。
 *
 * 与渲染端同一套有效速率：`hz` 模式直接用 `rateHz`，`cycles` 模式把整段周期数
 * 摊到窗口时长上（见 `vibratoCurve.ts` 的 `effectiveRateHz`）。
 */
export function cycleWidthPxFor(preset: VibratoPreset, widthPx: number, windowMs: number): number {
    const windowSec = Math.max(1e-6, windowMs / 1000);
    const rateHz =
        preset.rateMode === "cycles"
            ? Math.max(0, preset.cycles) / windowSec
            : Math.max(0, preset.rateHz);
    if (!(rateHz > 0)) return widthPx;
    return widthPx / (rateHz * windowSec);
}

/** 手柄的归一化横向位置：渐入在 `attackMs` 处，渐出在 `windowMs - releaseMs` 处。 */
export function handleLayoutFor(
    preset: Pick<VibratoPreset, "attackMs" | "releaseMs">,
    windowMs: number,
): PreviewHandleLayout {
    const window = Math.max(1, windowMs);
    return {
        attackFrac: clamp(preset.attackMs / window, 0, 1),
        releaseFrac: clamp(1 - preset.releaseMs / window, 0, 1),
    };
}
