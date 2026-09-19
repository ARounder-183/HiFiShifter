/**
 * 数位板 / 压感笔（stylus / pen）输入语义的集中判定。
 *
 * 【主要内容】
 * 把"这个 PointerEvent 是不是数位笔产生"的判断抽成纯函数：pointerType 归类、
 * 橡皮 / 笔杆键识别、悬停副作用门控、并发指针（掌压拒识）判定。全部只读输入
 * 事件的字段，不接触 DOM / React，可在 node 环境完整单测。
 *
 * 【为什么必须有这一层】
 * WebView（Chromium）里 pen 与 mouse 的 `button` 语义几乎一致（笔尖 = 0），
 * 但 pen 还有鼠标没有的维度：
 * - **感应高度悬停**：笔尖未接触就持续上报 `pointerType:"pen", buttons:0` 的
 *   pointermove / pointerover——若把这些事件当"鼠标移动"处理，时间轴的悬停
 *   光标 / 预览会在用户落笔前乱跳；
 * - **橡皮端**：笔尾橡皮接触上报 `button:5`（`buttons` 位 32），旧代码只认
 *   0/1/2，橡皮会被静默丢弃；
 * - **笔杆键**：上报 `button:2`，与鼠标右键同语义；
 * - **掌压**：书写时手掌落在触屏上会产生第二个指针（pointerType:"touch"），
 *   若不拒识，会中止并接管正在进行的笔手势。
 *
 * 【与其他模块的关系】
 * - 上游：`usePianoRollInteractions`、`timelineKernelHost`、`pianoRollKernelHost`、
 *   `App`（分割条）、`TrackList` 等所有指针入口在分派手势前调用本模块。
 * - 下游：无。本模块是纯判定，不保存状态。
 * - 独立性：纯函数 + 常量，不依赖 DOM / WebGL / React。
 *
 * 【设计约束】
 * 1. **未知 pointerType 按鼠标对待**（宽松回退）：`pointerType` 为空字符串的
 *    合成事件（见 `usePianoRollInteractions` 的修饰键合成 pointermove）与旧
 *    测试桩必须保持原有行为，不能因为本模块而被意外拦截。
 * 2. **一切判定只读事件字段**。`pointerType` 缺省时降级到压力/倾斜探测：
 *    真实硬件 pen 事件总带 `pressure > 0` 或非零 `tiltX/tiltY`，而 mouse 的
 *    `pressure` 恒为 0.5（按下时）。据此可以在驱动上报 `pointerType:"mouse"`
 *    （WinTab 直通模式）时仍识别出笔——保守起见只在 `pointerType` 明确缺失
 *    （合成事件）时才做压力探测，避免把压住左键的真鼠标误判为笔。
 */

/** 指针设备的归类。`"unknown"` 表示字段缺失（合成事件 / 测试桩）。 */
export type PointerKind = "mouse" | "pen" | "touch" | "unknown";

/** pen 橡皮端的主按钮值（`PointerEvent.button`）。 */
export const PEN_ERASER_BUTTON = 5;

/** pen 橡皮端在 `PointerEvent.buttons` 位掩码中的位。 */
export const PEN_ERASER_BUTTONS_MASK = 32;

/**
 * 归类一次指针事件的设备类型。
 *
 * @param pointerType `PointerEvent.pointerType`（"mouse" | "pen" | "touch" | ""）。
 * @returns 归类结果；空串 / 未定义 → `"unknown"`。
 */
export function pointerKindOf(pointerType: string | null | undefined): PointerKind {
    if (pointerType === "mouse") return "mouse";
    if (pointerType === "pen") return "pen";
    if (pointerType === "touch") return "touch";
    return "unknown";
}

/**
 * 该事件是否由数位笔（或疑似数位笔）产生。
 *
 * 判定顺序：
 * 1. `pointerType === "pen"` → 是。
 * 2. `pointerType` 缺失（合成事件 / 旧驱动）→ 按 **压力 / 倾斜** 兜底探测：
 *    真实 pen 事件带 `pressure > 0` 或非零 `tiltX/tiltY`。mouse 按下时
 *    `pressure` 恒为 0.5，但 mouse 事件总有明确的 `pointerType:"mouse"`，
 *    不会走到探测分支。
 * 3. 其余（mouse / touch / unknown 且无压力特征）→ 否。
 *
 * 特殊说明：`pointerType:"unknown"` 且带压力的事件同样按笔对待——这是唯一
 * 能覆盖"驱动把笔上报成 mouse/直通"场景的保守兜底；同时合成事件
 * （无压力字段）不受影响，保持宽松回退。
 *
 * @param e 事件或事件快照（只需 pointerType / pressure / tiltX / tiltY）。
 */
export function isStylusLike(e: {
    pointerType?: string | null;
    pressure?: number;
    tiltX?: number;
    tiltY?: number;
}): boolean {
    if (e.pointerType === "pen") return true;
    if (e.pointerType != null && e.pointerType !== "") return false;
    // 合成 / 直通事件：无明确类型时用硬件特征兜底。
    const pressure = typeof e.pressure === "number" ? e.pressure : 0;
    const tilt =
        Math.abs(typeof e.tiltX === "number" ? e.tiltX : 0) +
        Math.abs(typeof e.tiltY === "number" ? e.tiltY : 0);
    return pressure > 0 || tilt > 0;
}

/**
 * 该按下是否来自 pen 的橡皮端（笔尾擦除头）。
 *
 * Chromium 对橡皮端上报 `button:5`、`buttons` 位 32；鼠标永远不会产生 5。
 * 非法输入（非有限值）一律判定为否——与 `gestureHitTest` 对非有限输入的
 * 显式拒绝同一约定。
 */
export function isEraserButton(
    button: number | null | undefined,
    pointerType?: string | null,
): boolean {
    if (!Number.isFinite(button)) return false;
    if (button !== PEN_ERASER_BUTTON) return false;
    // button===5 只可能出现在 pen 上，但保留类型校验以防驱动直通。
    return pointerType == null || pointerType === "" || pointerType === "pen";
}

/**
 * 该按下是否应按"右键 / 次级动作"解释（鼠标右键或 pen 笔杆键）。
 *
 * pen 笔杆键上报 `button:2`，与鼠标右键天然同值，因此本函数在大多数调用点
 * 只是让 `e.button === 2` 的既有语义对 pen 显式成立；单独抽出是为了让
 * "笔杆键 = 右键"这一约定有唯一出处，后续若把笔杆键改映射到其它动作
 * 只改这里。
 */
export function isSecondaryButtonDown(
    button: number | null | undefined,
    pointerType?: string | null,
): boolean {
    if (!Number.isFinite(button)) return false;
    return button === 2 && (pointerType == null || pointerType !== "touch");
}

/**
 * 悬停 / 轻扫类副作用（按下即 seek、按下即弹参数值浮窗、按下即改布局等）
 * 是否应对该设备关闭。
 *
 * 约定：**只有鼠标保留"悬停副作用"**。pen 的高频悬停事件（笔尖未接触即
 * 上报 move/over）与轻扫起笔都极易误触这些副作用；触摸（手指）按下是明确
 * 的接触意图，但也应排除——触屏用户滑动浏览时误触面板级拖拽的代价同样
 * 很高。`"unknown"`（合成事件 / 测试桩）按鼠标对待，保持原有行为。
 */
export function shouldSuppressHoverSideEffects(e: { pointerType?: string | null }): boolean {
    const kind = pointerKindOf(e.pointerType);
    return kind === "pen" || kind === "touch";
}

/**
 * 第二个指针的按下是否应被拒识（掌压拒识）。
 *
 * 语义：**已有手势进行中**（`activeGestureActive === true`）时，触摸指针
 * （笔尖书写时的手掌）不得接管或中止手势；pen/mouse 的第二指针按下维持
 * 原有"先收尾旧手势再开新手势"语义（这是单手用户切笔尖/橡皮的正常路径）。
 *
 * @param e 新按下的事件。
 * @param activeGestureActive 是否已有手势进行中（单槽手势收尾非空）。
 */
export function shouldRejectConcurrentPointer(
    e: { pointerType?: string | null },
    activeGestureActive: boolean,
): boolean {
    if (!activeGestureActive) return false;
    return pointerKindOf(e.pointerType) === "touch";
}

// ── 兼容 mouse 事件的设备判定 ─────────────────────────────────────
//
// 旧式 mouse 入口（标尺的 mousedown / mousemove 等）不携带 pointerType，
// 而 pen / touch 在 Chromium 里会派生这些兼容 mouse 事件。要判断"这次
// mousedown 是不是笔划出来的"，可靠依据是**最近一次真实 pointer 事件**
// （pointerdown / pointermove 恒带 pointerType）——同一设备的按下序列里，
// mouse 事件之前必然有同设备的 pointer 事件。
//
// 记录器惰性安装、capture + passive（只读不拦截），与 `gestureFocusGuard`
// 的"惰性挂载全局监听"先例同一模式；经 globalThis 解析 window，node 测试
// 环境无 window 时安静降级（返回 null → 不拦截，宽松回退）。

let lastPointerType: string | null = null;
let trackerInstalled = false;

function ensurePointerTypeTracker(): void {
    const g = globalThis as {
        window?: {
            addEventListener(
                type: "pointerdown" | "pointermove",
                listener: (ev: PointerEvent) => void,
                options?: { capture?: boolean; passive?: boolean },
            ): void;
        } | null;
    };
    const win = g.window;
    if (!win || trackerInstalled) return;
    trackerInstalled = true;
    const record = (ev: PointerEvent) => {
        lastPointerType = ev.pointerType || null;
    };
    win.addEventListener("pointerdown", record, { capture: true, passive: true });
    win.addEventListener("pointermove", record, { capture: true, passive: true });
}

/**
 * 最近一次真实指针事件的设备类型（"mouse" | "pen" | "touch" | null）。
 *
 * 仅在浏览器环境可用；node / 早期调用返回 null（表示"未知"，宽松回退）。
 */
export function lastRealPointerType(): string | null {
    ensurePointerTypeTracker();
    return lastPointerType;
}

/**
 * 兼容 mouse 事件（无 pointerType 的 mousedown 等）是否来自数位笔 / 触摸。
 *
 * 判定依据：最近一次真实 pointer 事件的类型。若尚未记录到任何 pointer
 * 事件（测试环境 / 无先序事件），返回 false——宽松回退，不拦截。
 */
export function isLegacyMouseEventFromStylus(): boolean {
    const kind = pointerKindOf(lastRealPointerType());
    return kind === "pen" || kind === "touch";
}

/**
 * 读取一个 pointer 事件的全部同帧采样点（`getCoalescedEvents`）。
 *
 * 【为什么需要】pen 采样率常见 133–266Hz（高端 500Hz+），高于渲染帧率：
 * 快速运笔时一帧内会积累多个 pointermove，只取最后一个事件会丢掉中间
 * 轨迹点，曲线出现折线感。逐帧取全部同帧采样点后，提交/预览的轨迹与
 * 笔尖实际路径逐点一致。
 *
 * 兼容性：`getCoalescedEvents` 在合成事件 / 旧 WebView / 测试桩上不存在
 * 或返回空数组，两种情况都回退为 `[event]`——调用方语义不变。
 */
export function coalescedEventsOf<T extends { getCoalescedEvents?: () => T[] }>(event: T): T[] {
    try {
        const list = event.getCoalescedEvents?.();
        if (Array.isArray(list) && list.length > 0) {
            return list;
        }
    } catch {
        // 合成事件 / 测试桩上的异常调用：回退为单事件。
    }
    return [event];
}
