/**
 * 「气声分离」开关对其它参数的依赖门禁。
 *
 * # 主要内容
 * - `SEPARATION_GATED_PARAMS`：被门禁的参数 id 列表。
 * - `isGatedBySeparation()`：某参数当前是否因开关关闭而不可用。
 * - `findBlockedEditParam()`：`editParam` 是否停在被门禁的参数上（需回退）。
 *
 * # 作用
 * `breath_enabled`（UI 名「气声分离」）是**气声音量与张力共同的前提**：
 *
 * - `breath_gain`（气声音量）：噪声支只存在于 HNSEP 分离路径；
 * - `hifigan_tension`（张力）：Rd 张力只重塑**谐波**支，同样依赖分离。
 *
 * 开关关闭时，这两个参数在 UI 上置灰、不可选中，其曲线**保持可见但不可编辑**；
 * 同时后端会把这两条曲线从下发数据中剥离，使其**不参与合成**且**不触发 HNSEP**
 * （见 `backend/src-tauri/src/pitch_editing.rs` 的 `gate_separation_curves`）。
 *
 * `formant_shift_cents`（共振峰）**不**被门禁：它走 mel 阶段的 `keyShift`，
 * 与 HNSEP 无关，关闭分离时照常可用。
 *
 * # 与其他模块的关系
 * - 消费方：`PianoRollPanel`（药丸置灰、`editParam` 回退、参数下拉置灰）。
 * - 后端对应实现：`pitch_editing::gate_separation_curves`（合成侧剥离）与
 *   `hifigan_tension_active_for_clip`（决策侧判定）。两侧判据同源，
 *   都归结为「开关是否为开」，因此本文件不重复实现阈值语义。
 *
 * # 维护说明
 * 新增"依赖分离"的参数时，只需加入 `SEPARATION_GATED_PARAMS`；
 * 后端的曲线剥离列表 `HIFIGAN_SEPARATION_GATED_CURVES` 需同步更新
 * （两侧列表必须一致，否则会出现"UI 置灰但实际生效"或反之）。
 */

/** 分离开关的静态参数 id（与后端 `HIFIGAN_SEPARATION_PARAM_ID` 一致）。 */
export const SEPARATION_PARAM_ID = "breath_enabled";

/**
 * 依赖气声分离、因而在开关关闭时被门禁的参数 id。
 *
 * 必须与后端 `pitch_editing::HIFIGAN_SEPARATION_GATED_CURVES` 保持一致。
 */
export const SEPARATION_GATED_PARAMS: readonly string[] = ["breath_gain", "hifigan_tension"];

/**
 * 开关状态是否为"开"。
 *
 * 阈值语义与后端 `extra_param_enabled` 一致：`>= 0.5` 视为开，
 * 缺失键视为关（默认值 0）。两侧必须同口径，否则会出现
 * "UI 显示可用但后端已剥离"这类不一致。
 *
 * @param rawValue 开关的原始值；`undefined` 表示尚未加载，按描述符默认值处理
 * @param defaultValue 描述符声明的默认值（当前为 0 = 关）
 */
export function isSeparationEnabled(rawValue: number | undefined, defaultValue: number): boolean {
    return (rawValue ?? defaultValue) >= 0.5;
}

/**
 * 该参数当前是否因分离开关关闭而不可用（应置灰、不可选中）。
 *
 * @param paramId 参数 id
 * @param separationEnabled 分离开关是否为开
 */
export function isGatedBySeparation(paramId: string, separationEnabled: boolean): boolean {
    return !separationEnabled && SEPARATION_GATED_PARAMS.includes(paramId);
}

/**
 * 若 `editParam` 停在一个被门禁的参数上，返回应当回退到的参数；否则返回 `null`。
 *
 * # 为什么需要回退而不是就地禁用
 * 曲线编辑的唯一闸门是 `editParam`（画布绘制路径经 `usePianoRollInteractions`
 * 读取同一状态）。被门禁的参数若仍停留在 `editParam` 上，用户在画布上的
 * 绘制操作仍会落到它身上 —— 那与"不可编辑"矛盾。把 `editParam` 移开，
 * 曲线依然**可见**（仍会被绘制），但不再接受编辑。
 *
 * # 参数
 * - `editParam`：当前编辑参数
 * - `separationEnabled`：分离开关是否为开
 * - `fallback`：回退目标（通常是 `"pitch"`）
 */
export function findBlockedEditParam(
    editParam: string,
    separationEnabled: boolean,
    fallback: string,
): string | null {
    if (!isGatedBySeparation(editParam, separationEnabled)) return null;
    // 已在回退目标上（理论上不会发生，因为 fallback 自身不被门禁）——
    // 返回 null 以避免无意义的 dispatch 循环。
    return editParam === fallback ? null : fallback;
}

/**
 * 关闭开关触发 `editParam` 回退时，需要顺带把"眼睛"打开的参数。
 *
 * # 为什么需要它
 * 曲线只在两种情况下被绘制：它是当前 `editParam`，或它的可见性标记为真。
 * 回退会把被门禁的参数从 `editParam` 移走，而它的眼睛默认是关的 ——
 * 若不补这一步，用户在编辑张力/气声音量时关掉开关，曲线会**直接消失**，
 * 这与"置灰但保持可见、只是不可编辑"的要求相反。
 *
 * @returns 需要打开可见性的参数 id；无需回退时返回 `null`
 */
export function paramNeedingVisibilityOnGate(
    editParam: string,
    separationEnabled: boolean,
    fallback: string,
): string | null {
    return findBlockedEditParam(editParam, separationEnabled, fallback) === null ? null : editParam;
}

/**
 * Compose 关闭时不可用的参数：**音高之外**的轨道级"合成"参数。
 *
 * 需求口径：Compose 只影响**音高、共振峰、气声、张力**这四类合成参数；
 * 音量 / 声相 / 动态是**混音级**参数，刻意不受 Compose 限制，不在此列。
 * 音高由 `pitch_requires_compose` 单独提示，故这里只列其余三项。
 */
export const COMPOSE_GATED_PARAMS: readonly string[] = [
    "formant_shift_cents",
    "breath_gain",
    "hifigan_tension",
];

/**
 * 该参数当前是否因 **Compose 关闭**而不可用（应置灰、不可选中、不参与合成）。
 *
 * @param paramId 参数 id
 * @param composeEnabled 所属轨道组的 Compose 开关是否为开
 */
export function isGatedByCompose(paramId: string, composeEnabled: boolean): boolean {
    return !composeEnabled && COMPOSE_GATED_PARAMS.includes(paramId);
}

/**
 * 统一的效果参数门禁：Compose 关闭**或**气声分离开关闭时不可用。
 *
 * 后端在**同一处**（`gate_hifigan_effect_curves`）落地这两道门禁，
 * 前端也必须同源判断，否则会出现"UI 可编辑、后端已剥离"或反之。
 */
export function isEffectParamGated(
    paramId: string,
    separationEnabled: boolean,
    composeEnabled: boolean,
): boolean {
    return (
        isGatedBySeparation(paramId, separationEnabled) || isGatedByCompose(paramId, composeEnabled)
    );
}
