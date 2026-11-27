/**
 * Clip / Take / Track 增益的唯一数学入口。
 *
 * 【值域】kernel 权威：线性 [0, 16]，0 是唯一静音；显示下界 −150 dB，
 * 锥度下界 −60 dB。前端只镜像这些常量与边界，不另写一套。
 *
 * 【dB 只住在边界】模型层（Redux / 手势结果）一律线性；dB 只在文本、
 * 输入框与角度计算里出现。
 */

/** 线性增益下界（0 = −inf dB，静音）。 */
export const GAIN_MIN_LINEAR = 0;

/** 线性增益上界（16 = +24.08 dB，与 REAPER item 上限一致）。 */
export const GAIN_MAX_LINEAR = 16;

/** 显示下界（REAPER 口径）：低于它显示 -inf。 */
export const GAIN_DISPLAY_FLOOR_DB = -150;

/** dB 输入/显示下界（与显示下界同源）。 */
export const GAIN_MIN_DB = GAIN_DISPLAY_FLOOR_DB;

/** 旋钮锥度下界：−60 以上保持 1.25 dB/px 的手感。 */
export const GAIN_TAPER_FLOOR_DB = -60;

/** 显示下界对应的线性值。 */
export const GAIN_FLOOR = 10 ** (GAIN_DISPLAY_FLOOR_DB / 20);

/** 两层增益的乘积上限（16 × 16），只做防溢出。 */
export const GAIN_PRODUCT_MAX = GAIN_MAX_LINEAR * GAIN_MAX_LINEAR;

/** dB 上界（由线性上界反算）。 */
export const GAIN_MAX_DB = 20 * Math.log10(GAIN_MAX_LINEAR);

const TRAVEL_UP_PX = 48;
const TRAVEL_DOWN_PX = 48;
const TRAVEL_DEEP_PX = 9;
const DETENT_PX = 1;
const ZERO_PX = DETENT_PX + TRAVEL_DEEP_PX + TRAVEL_DOWN_PX;
const TOP_PX = ZERO_PX + TRAVEL_UP_PX;
const TAPER_PX = DETENT_PX + TRAVEL_DEEP_PX;
const KNOB_MAX_ANGLE_DEG = 135;

/** 非负层钳制：NaN → 1；≤0 / 低于显示下界 → 0；上界 16。 */
export function clampGain(linear: number): number {
    if (Number.isNaN(linear)) return 1;
    if (!(linear > GAIN_MIN_LINEAR) || linear < GAIN_FLOOR) return 0;
    return Math.min(GAIN_MAX_LINEAR, linear);
}

/** take 层钳制：保号（负值 = 极性反转）；幅度低于显示下界归零。 */
export function clampTakeGain(linear: number): number {
    if (Number.isNaN(linear)) return 1;
    const magnitude = Math.abs(linear);
    if (magnitude < GAIN_FLOOR) return 0;
    return Math.max(-GAIN_MAX_LINEAR, Math.min(GAIN_MAX_LINEAR, linear));
}

/** 把 dB 值钳进值域（输入边界用）。 */
export function clampGainDb(db: number): number {
    if (!Number.isFinite(db)) return GAIN_DISPLAY_FLOOR_DB;
    return Math.min(GAIN_MAX_DB, Math.max(GAIN_DISPLAY_FLOOR_DB, db));
}

/** 线性 → dB；静音返回 -Infinity（不是 −150 的“实用下界”）。 */
export function gainToDb(linear: number): number {
    const gain = clampGain(linear);
    return gain > 0 ? 20 * Math.log10(gain) : Number.NEGATIVE_INFINITY;
}

/** 线性 → dB，静音按显示下界 −150 处理（锥度/角度/增量基准用）。 */
export function dbOrFloor(linear: number): number {
    const db = gainToDb(linear);
    return Number.isFinite(db) ? db : GAIN_DISPLAY_FLOOR_DB;
}

/** @deprecated 仅过渡用；新代码请用 gainToDb / dbOrFloor。 */
export function gainToDbSafe(linear: number): number {
    return dbOrFloor(linear);
}

/** dB → 线性；低于 −150 或非有限值 → 0（静音）。 */
export function dbToGain(db: number): number {
    if (!Number.isFinite(db) || db < GAIN_DISPLAY_FLOOR_DB) return 0;
    return Math.min(GAIN_MAX_LINEAR, 10 ** (db / 20));
}

/** dB → 展示文本（不带单位）。 */
export function formatGainDbValue(db: number): string {
    if (!Number.isFinite(db) || db < GAIN_DISPLAY_FLOOR_DB - 1e-9) return "-inf";
    const rounded = Math.round(db * 10) / 10;
    if (Math.abs(rounded) < 0.05) return "0";
    const sign = rounded > 0 ? "+" : "";
    // 统一一位小数：+6.0 / -6.0。只有精确的 0 dB 与 −150 dB 显示整数。
    if (rounded === GAIN_DISPLAY_FLOOR_DB) return "-150";
    return `${sign}${rounded.toFixed(1)}`;
}

/** 线性 → 完整展示标签（唯一格式化入口，含单位）。 */
export function formatGainDbLabel(linear: number): string {
    return `${formatGainDbValue(gainToDb(linear))} dB`;
}

/**
 * 解析用户输入的增益。返回 dB；`-inf` 返回 -Infinity（调用方 dbToGain 后是 0）。
 * 接受 `+3`、`3 dB`、`-6,5`；正无穷与其他非法值返回 null。
 */
export function parseGainDbInput(raw: string): number | null {
    let text = raw
        .trim()
        .toLowerCase()
        .replace(/\s*d\s*b\s*$/u, "")
        .trim();
    if (text === "") return null;
    if (text === "-inf" || text === "-∞" || text === "−inf") return Number.NEGATIVE_INFINITY;
    text = text.replace(/[−–]/g, "-").replace(",", ".");
    const value = Number(text);
    if (!Number.isFinite(value)) return null;
    return clampGainDb(value);
}

/** 与 Rust `GainScope` 对应的前端只读语义（UI 判断用）。 */
export const GainScope = { AppOwned: "app", HostOwned: "host" } as const;

export interface GainTaper {
    readonly zeroPx: number;
    readonly topPx: number;
    readonly taperPx: number;
    positionOf(linear: number): number;
    gainAt(positionPx: number): number;
    angleDeg(linear: number): number;
}

/** 线性值 → 底部起算的行程位置（px）。 */
export function positionOf(linear: number): number {
    const gain = clampGain(linear);
    if (gain <= 0) return 0;
    const db = 20 * Math.log10(gain);
    if (db >= 0) return ZERO_PX + (db / GAIN_MAX_DB) * TRAVEL_UP_PX;
    if (db >= GAIN_TAPER_FLOOR_DB) {
        return ZERO_PX + (db / -GAIN_TAPER_FLOOR_DB) * TRAVEL_DOWN_PX;
    }
    return (
        DETENT_PX +
        ((db - GAIN_DISPLAY_FLOOR_DB) / (GAIN_TAPER_FLOOR_DB - GAIN_DISPLAY_FLOOR_DB)) *
            TRAVEL_DEEP_PX
    );
}

/** 行程位置（px）→ 线性值；0..1px 是静音棘齿。 */
export function gainAt(positionPx: number): number {
    if (!(positionPx >= DETENT_PX)) return 0;
    if (positionPx <= TAPER_PX) {
        const db =
            GAIN_DISPLAY_FLOOR_DB +
            ((positionPx - DETENT_PX) / TRAVEL_DEEP_PX) *
                (GAIN_TAPER_FLOOR_DB - GAIN_DISPLAY_FLOOR_DB);
        return dbToGain(db);
    }
    if (positionPx <= ZERO_PX) {
        const db = ((positionPx - ZERO_PX) / TRAVEL_DOWN_PX) * -GAIN_TAPER_FLOOR_DB;
        return dbToGain(db);
    }
    const db = ((positionPx - ZERO_PX) / TRAVEL_UP_PX) * GAIN_MAX_DB;
    return dbToGain(Math.min(db, GAIN_MAX_DB));
}

/** 指针角度：0 dB = 0°，+24.08 = +135°，−60 = −117°，−150 = −135°。 */
function angleDeg(linear: number): number {
    const db = dbOrFloor(linear);
    if (db >= 0) return (db / GAIN_MAX_DB) * KNOB_MAX_ANGLE_DEG;
    if (db >= GAIN_TAPER_FLOOR_DB) {
        return (db / -GAIN_TAPER_FLOOR_DB) * 117;
    }
    return (
        -117 + ((db - GAIN_TAPER_FLOOR_DB) / (GAIN_DISPLAY_FLOOR_DB - GAIN_TAPER_FLOOR_DB)) * -18
    );
}

/** 唯一锥度对象：Clip 旋钮、轨道旋钮、滚轮与拖拽共用。 */
export const gainTaper: GainTaper = Object.freeze({
    zeroPx: ZERO_PX,
    topPx: TOP_PX,
    taperPx: TAPER_PX,
    positionOf,
    gainAt,
    angleDeg,
});

/** 线性入参的角度（保留旧名，签名从 dB 改为线性）。 */
export function gainKnobAngleDeg(linear: number): number {
    return angleDeg(linear);
}

export interface GainDragParticipant {
    readonly clipId: string;
    readonly baseLinear: number;
}

export interface GainDragResult {
    /** 锚点 clip 的新线性增益；0 = 静音。 */
    readonly anchorLinear: number;
    /** 其余参与者按同一 dB 增量平移后的结果。 */
    readonly others: ReadonlyArray<{ clipId: string; linear: number }>;
}

/**
 * 唯一手势解析器：锚点精确等于指针位置，其余成员按 dB 增量平移。
 * 返回线性值；静音严格是 0，绝不返回“0 dB”的哨兵。
 */
export function resolveGainDrag(args: {
    anchorBaseLinear: number;
    deltaPx: number;
    fine: boolean;
    participants: ReadonlyArray<GainDragParticipant>;
    anchorId: string;
}): GainDragResult {
    const { anchorBaseLinear, deltaPx, fine, participants, anchorId } = args;
    const startPx = positionOf(anchorBaseLinear);
    const scale = fine ? 0.25 : 1;
    const anchorLinear = gainAt(startPx + deltaPx * scale);
    const anchorBaseDb = dbOrFloor(anchorBaseLinear);
    const anchorNextDb = dbOrFloor(anchorLinear);
    const deltaDb = anchorNextDb - anchorBaseDb;
    const others = participants
        .filter((participant) => participant.clipId !== anchorId)
        .map((participant) => {
            const base = clampGain(participant.baseLinear);
            const nextDb = clampGainDb(dbOrFloor(base) + deltaDb);
            // 原本就是静音的成员不应因为“−150 的数值表示”变成 3.16e-8：
            // 0 是唯一静音，db 边界只是锥度/增量的计算域。
            const linear = base === 0 && nextDb <= GAIN_DISPLAY_FLOOR_DB ? 0 : dbToGain(nextDb);
            return { clipId: participant.clipId, linear };
        });
    return { anchorLinear, others };
}
