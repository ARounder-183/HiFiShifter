/**
 * 「精细调整」修饰键的轴向拖拽累计器。
 *
 * 【它解决什么】拖拽途中按下 / 松开精细调整修饰键时，**不能**把已经累计的位移
 * 按新比例重算一遍 —— 那会让数值（增益、音分、深度…）在一瞬间跳回某个旧位置，
 * 用户看到的是"闪回"，正在进行的拖拽被打断。
 *
 * 正确的做法是**只按增量缩放**：整段手势维护一个累计量，每帧只把"这一帧新走的
 * 那一小段"按当前比例加进去。比例变了，改变的只是**此后**每帧走多少，已经走过
 * 的部分原封不动 —— 于是切换修饰键时数值连续、不跳。
 *
 * 【输入是坐标还是增量都行】状态只比较相邻两次 `nextRaw` 的差，因此 `raw` 既可
 * 以是绝对的指针坐标（增益旋钮直接喂 `clientY`），也可以是"相对按下点的累计
 * 位移"（内核手势喂累计 delta）。两种喂法等价。
 *
 * 【为什么单独成文件】它是时间轴（轨道 / Clip 增益、Clip 音高拖拽）与颤音预设
 * 预览画布共用的手感规则；"修饰键中途切换会不会闪回"这类缺陷不会抛错，只能靠
 * 单测钉住（见 `fineAxisDrag.test.ts`）。
 */

export type FineAxisDragState = {
    /** 上一次的原始输入（坐标或累计位移）。 */
    raw: number;
    /** 累计的**已缩放**输入，与 `raw` 同量纲、同起点。 */
    adjusted: number;
    /** 上一帧精细调整是否生效。 */
    fineActive: boolean;
};

/** 精细调整生效时，位移缩到这个比例。 */
const FINE_AXIS_DRAG_SCALE = 0.2;

/**
 * 修饰键**刚按下**的那一帧，位移按这个比例走。
 *
 * 比 `FINE_AXIS_DRAG_SCALE` 更接近 1：比例从 1 直接掉到 0.2 会让指针"突然拽不动
 * 了"，这一帧先走 65%，下一帧起才进入 0.2 —— 手感上是"顺滑地减速"而不是"卡住"。
 */
const FINE_AXIS_DRAG_TRANSITION_SCALE = 0.65;

/**
 * 推进一帧，返回**累计的已缩放位移**。
 *
 * @param state 手势期间持续复用的状态（就地更新）。
 * @param nextRaw 本帧的原始输入，与 `state.raw` 同量纲。
 * @param fineActive 本帧精细调整是否生效。
 * @returns 从手势起点累计的已缩放位移。
 */
export function advanceFineAxisDrag(
    state: FineAxisDragState,
    nextRaw: number,
    fineActive: boolean,
): number {
    const delta = nextRaw - state.raw;
    if (fineActive && !state.fineActive) {
        state.adjusted += delta * FINE_AXIS_DRAG_TRANSITION_SCALE;
    } else {
        const scale = fineActive ? FINE_AXIS_DRAG_SCALE : 1;
        state.adjusted += delta * scale;
    }
    state.raw = nextRaw;
    state.fineActive = fineActive;
    return state.adjusted;
}

/** 新建一份手势状态：`raw` 与 `adjusted` 必须同起点，否则首帧会凭空产生位移。 */
export function createFineAxisDragState(start: number, fineActive: boolean): FineAxisDragState {
    return { raw: start, adjusted: start, fineActive };
}
