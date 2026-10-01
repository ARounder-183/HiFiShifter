/**
 * 音高参数的「未设置」哨兵在颤音链路上的适配。
 *
 * 【语义】`pitch` 的参数值里 **0 = 未检测到音高**（未浊 / 静音帧）；真实音高是
 * 1..127 的半音值。后端（`commands/params.rs` 的读写两侧）与前端写入口
 * （`paramRanges.clampParamWriteValue`）都以此为准，参数编辑器里每个变换也都遵守
 * 「0 进 0 出」。
 *
 * 【为什么颤音必须单独适配】颤音的基线锚点是**从选区两端读出来的**。首 / 末帧
 * 一旦不是真正的音符，锚点就被它带偏：`baseline: "line"` 会把整条选区拉向那个错值，
 * 中间真实唱出来的音高被整段抹掉。而"选一整句"几乎必然包含句首句尾的气口，所以
 * 这是常态而不是边角。
 *
 * 【"不是真正的音符"有两种，只判 `== 0` 会漏掉第二种】
 *
 * 1. **未检测**：值为 0。判据是 `> 0`（后端四处一致：`renderer/utils.rs`、
 *    `pitch_editing.rs::is_voiced_at_time`、`params.rs`、`midi_export.rs`）。
 * 2. **浊清边界的过渡帧**：音高跟踪器在起声 / 收声处常给出**低而非零**的 f0 估计
 *    （20~40 Hz → MIDI 15~28）。它们既躲过 `== 0`，又通过写入口的 `1..127`
 *    合法性检查，于是被当成"正常音高"。只加 `> 0` 也治不了它们。
 *
 * 因此这里不逐帧判断，而是**按音符段**判断，判据取自 MIDI 导出的既有做法
 * （`commands/midi_export.rs::pitch_curve_to_track_events` 的"候选音符"阶段）：
 * 一段连续有声帧要**够长**才算一个音符，短抹痕一律不成其为音符。导出那边用它决定
 * "哪些段值得导成一个 MIDI 音符"，这里用它决定"哪些段值得加颤音" —— 同一个问题的
 * 两种消费。
 *
 * 【刻意不抄导出的两条】导出还会按"跨度超过 1 半音"和"在某个半音上稳定 8 帧"把
 * 连续段切成多个音符**事件**。那对锚点只有坏处：颤音选区常跨几个音、本身还可能带
 * 大颤音（摆动幅度轻松超过 1 半音），照抄会把锚点切进颤音内部。
 *
 * 【锚点为什么取音符段的代表值（中位数）】音符的**边界帧**恰恰是跟踪器最不可靠的
 * 地方（起声 / 收声的衰减尾巴就贴着气口）。取代表值，锚点才会稳定落在音符本身上；
 * 导出侧同样是取中心值（`round((min+max)/2)`）而不是边界采样。一个可见的推论：
 * `baseline: "line"` 落在**单个**音符上时两端锚点相同，于是把这段拉平到该音符的
 * 代表音高 —— 这正是"直线"该有的意思（把音高拉直），而不是"跟随音头到音尾"。
 *
 * 调用方只需一对函数：{@link planVibratoTarget} 规划落点，
 * {@link suppressNonTargetFrames} 在写回前把不该动的帧还原成哨兵。
 */

import { PITCH_PARAM_ID } from "./vibratoDepth";

/**
 * 该参数是否用 0 表示「无数据」。
 *
 * 目前只有音高。落成一个具名判定而不是散落的 `param === "pitch"`，是为了让
 * "哨兵适配"这件事有唯一的入口 —— 将来若某个参数也采用 0 哨兵，改这里一处即可。
 */
export function usesUnsetValue(param: string): boolean {
    return param === PITCH_PARAM_ID;
}

/**
 * 该帧是否为「未设置」。
 *
 * 判据是 `> 0` 而不是 `== 0`：后端一律用 `is_finite() && > 0.0` 表示"有音高"
 * （`renderer/utils.rs` 的插值、`pitch_editing.rs` 的 `is_voiced_at_time`、
 * `compute_clip_export_pitch_offsets` 的"`≤ 0` 不做修正"），负值与 0 同义。
 * 非有限值一并算未设置 —— 静音 / 未分析帧在上游可能以 `NaN` 出现。
 */
export function isUnsetValue(param: string, value: number): boolean {
    if (!usesUnsetValue(param)) return false;
    return !(Number.isFinite(value) && value > 0);
}

/**
 * 连续有声帧要够长才算一个音符（毫秒）。
 *
 * ⚠ 与后端 `commands/midi_export.rs::MIN_NOTE_DURATION_MS` 同源（同一套
 * 前后端常量镜像的惯例，见 `paramRanges.ts` 的 `DYN_FOLLOW_ORIG` ↔
 * `renderer/common_params.rs`）。改动其一必须同步另一处，否则"导出里有这个音、
 * 加颤音时却当它不存在"会成为很难归因的不一致。
 */
export const MIN_NOTE_MS = 100;

/**
 * 音高值域（MIDI 半音）。
 *
 * ⚠ 与 `PianoRollPanel` 的 `currentParamRange`（音高：`24..108`）一致 —— 那是参数
 * 编辑器自己认定的"有意义的音高值域"。这一条专治上面第 2 类过渡帧里**持续时间够长**
 * 的那种：跟踪器把一段 30~40 Hz 的低估坚持上百毫秒时，"够长"这道关拦不住它，但
 * MIDI 24（C1，32.7 Hz）以下不可能是人声基频，值域这道关能拦。
 */
export const PLAUSIBLE_PITCH_MIN = 24;
export const PLAUSIBLE_PITCH_MAX = 108;

/** 一个音符段。 */
export interface VibratoNoteRun {
    /** 起帧（含）。 */
    startIndex: number;
    /** 止帧（不含）。 */
    endIndex: number;
    /** 该段的稳健代表音高（中位数）。 */
    pitch: number;
}

/** 基线锚点：曲线首尾的**有效**值。 */
export interface VibratoAnchors {
    startValue: number;
    endValue: number;
}

/**
 * 一次颤音套用的目标规划。
 *
 * 【为什么要连掩码一起给】锚点回答"基线摆在哪"，掩码回答"哪些帧该被改写"。
 * 两者必须出自**同一次**音符段切分，否则会出现"按音符 A 的锚点画线、却把线写进
 * 不是音符的帧里"这种自相矛盾的结果。
 */
export interface VibratoTargetPlan {
    anchors: VibratoAnchors;
    /** 与 `values` 等长；`true` = 这一帧参与颤音。 */
    modulatable: boolean[];
}

/** 该帧是否"像个音符"：有音高，且落在有意义的音高值域内。 */
function isNoteFrame(param: string, value: number): boolean {
    if (isUnsetValue(param, value)) return false;
    return value >= PLAUSIBLE_PITCH_MIN && value <= PLAUSIBLE_PITCH_MAX;
}

/** 中位数（稳健代表值：抗边界衰减尾巴，也抗单帧离群）。 */
function medianOf(values: readonly number[], start: number, end: number): number {
    const slice = values.slice(start, end).sort((a, b) => a - b);
    const mid = slice.length >> 1;
    return slice.length % 2 === 1 ? slice[mid] : (slice[mid - 1] + slice[mid]) / 2;
}

/** 切出音符段（只对哨兵参数有意义）。 */
function noteRuns(
    param: string,
    values: readonly number[],
    framePeriodMs: number,
): VibratoNoteRun[] {
    const fp = Number.isFinite(framePeriodMs) && framePeriodMs > 0 ? framePeriodMs : 5;
    // 门槛按**实际帧周期**折算，不写死帧数：帧周期是可配的（默认 5ms → 20 帧）。
    const minFrames = Math.max(1, Math.round(MIN_NOTE_MS / fp));
    const runs: VibratoNoteRun[] = [];
    let start = -1;
    const flush = (end: number) => {
        if (start < 0) return;
        if (end - start >= minFrames) {
            runs.push({ startIndex: start, endIndex: end, pitch: medianOf(values, start, end) });
        }
        start = -1;
    };
    for (let i = 0; i < values.length; i += 1) {
        if (isNoteFrame(param, values[i])) {
            if (start < 0) start = i;
        } else {
            flush(i);
        }
    }
    flush(values.length);
    return runs;
}

/**
 * 规划一次颤音的落点。
 *
 * - 非哨兵参数：整段都可调制，锚点 = 首末帧（与既有行为逐字一致）；
 * - 音高：只有**音符帧**可调制；锚点 = 首 / 末个音符段的稳健代表值。
 *
 * @returns 没有任何音符段时 `null` —— 没有可调制的对象，调用方应当放弃，
 *          而不是照着一片 0 写出 0（那会把气口当成"要加颤音的音高"）。
 */
export function planVibratoTarget(
    param: string,
    values: readonly number[],
    framePeriodMs: number,
): VibratoTargetPlan | null {
    if (values.length === 0) return null;

    if (!usesUnsetValue(param)) {
        return {
            anchors: { startValue: values[0], endValue: values[values.length - 1] },
            modulatable: new Array<boolean>(values.length).fill(true),
        };
    }

    const runs = noteRuns(param, values, framePeriodMs);
    if (runs.length === 0) return null;

    const modulatable = new Array<boolean>(values.length).fill(false);
    for (const run of runs) {
        for (let i = run.startIndex; i < run.endIndex; i += 1) modulatable[i] = true;
    }
    return {
        anchors: { startValue: runs[0].pitch, endValue: runs[runs.length - 1].pitch },
        modulatable,
    };
}

/**
 * 把「不受颤音影响」的帧写回 0（**就地**修改 `result`；按较短者与掩码对齐）。
 *
 * 【为什么写 0 就是"不改这一帧"】后端读侧对 pitch 有 `e_raw == 0 && o != 0 → o`
 * 的折叠，渲染侧 `edit_midi_at_time_or_none` 对整段未设置的帧也回落到原曲线 ——
 * 所以 0 的含义是"这一帧不做编辑"，不是"把音高删掉"。这与
 * `compute_clip_export_pitch_offsets` 的"`≤ 0` 的帧一律不做音高修正、**不得用邻近
 * 帧桥接**"是同一条规则，也与这个仓库里其它每个变换的"0 进 0 出"一致。
 *
 * 幂等：对已经还原过的结果再调用一次不会变。调用时机是"变换之后、写回之前"。
 */
export function suppressNonTargetFrames(result: number[], plan: VibratoTargetPlan): void {
    const count = Math.min(result.length, plan.modulatable.length);
    for (let i = 0; i < count; i += 1) {
        if (!plan.modulatable[i]) result[i] = 0;
    }
}
