/**
 * 参数编辑器 · 主画布内容签名（纯函数）
 *
 * 【主要内容】
 * 定义主画布（曲线 / 选区框 / morph 手柄 / 剪贴板预览 / 中央提示文字）内容签名的
 * 类型与比较规则。
 *
 * 【作用：为什么要独立成模块】
 * 签名是"要不要重绘主画布"的**唯一判据**，漏项或误判的代价都是"图层停止更新"。
 * 而它此前在 `PianoRollPanel` 内以 `[...].join("|")` 构造，`join` 会把每个
 * 对象/数组元素串成字面量 `"[object Object]"`——于是
 *
 *     [{aBeat:4.1, bBeat:6.7}] 与 [{aBeat:40, bBeat:90}]  →  同一个签名
 *
 * 缓存命中、`drawPianoRoll` 在清屏前就 return，**新选区框画不出来、旧框不消失**；
 * 只有 `null ↔ object` 的转换能改变签名，这精确解释了"第一次选区可见、再次选择
 * 不更新，拖到边缘自动滚屏时才突然更新"（滚动量是真正的签名原语）。
 *
 * 顺带修掉的同类塌缩项：`paramMorphOverlay`（morph 手柄拖拽同样静默失效，实测
 * 3 点 → 9 点签名相同）、`paramViewsRef.current`、`liveEditOverrideRef.current`、
 * `detectedPitchCurves`、`referencePitchOverlays`、`secondaryParamViews`。
 *
 * 独立成模块的另一个原因：**本工程 vitest 跑在 node 环境且不允许 import `.tsx`**
 * （会经 barrel 间接引入 Redux / localStorage 而崩溃），放在面板里就等于不可测。
 *
 * 【与其他模块的关系】
 * - 上游：`PianoRollPanel` 每次绘制构造签名；`render.ts` 的 `drawPianoRoll` 消费它。
 * - 独立性：纯函数 + 类型，无 React / DOM / Redux 依赖，可在 node 环境单测。
 *
 * 【设计约束】
 * 1. 用 `Object.is` 逐项比较，**不是**字符串化：对象 / 数组按**引用**参与。
 * 2. 引用比较能逐帧失效，是因为选区在被拖拽时每次都被赋一个**新对象**
 *    （`usePianoRollInteractions` 的 `selectionRef.current = {…}`）。
 *    **原地更新**的数据源不得直接以对象引用参与签名 —— 引用不变则缓存永不失效
 *    （曲线停止刷新）。若要原地更新，必须改用一个**显式版本号**参与签名：
 *    绘制中的 live 覆盖即如此（`liveEditOverrideRef.current?.version ?? 0`，
 *    版本号全局单调递增，见 `useLiveParamEditing` 的 `LiveEditOverride`）。
 *    两类做法都成立，禁止的是"原地改字段却仍拿对象引用当签名项"。
 * 3. `Object.is` 下 `NaN` 与自身相等，正好避免"某输入恒为 NaN 时每帧刷帧"。
 */

/**
 * 主画布内容签名。
 *
 * 元素可以是原始值（数值 / 字符串 / 布尔）或**任意对象引用**；对象一律按引用
 * 参与比较，不做深比较（深比较会进入每帧热路径）。
 */
export type MainCanvasSignature = readonly unknown[];

/**
 * 比较两个主画布内容签名是否等价。
 *
 * 流程：任一侧缺失 → 不等价；长度不同 → 不等价；逐项 `Object.is` 比较。
 *
 * 特殊说明：用 `Object.is` 而不是 `===`，因为它对 `NaN` 的处理符合本场景需要
 * （`NaN` 视为与自身相同，避免某个输入恒为 NaN 时每帧刷帧；`-0` / `+0` 的区分
 * 在本场景无影响）。
 *
 * @param next 本帧签名。
 * @param previous 上一帧签名；`undefined` 表示尚未绘制过。
 * @returns 等价（可走缓存、跳过重绘）时为 true。
 */
export function isSameMainCanvasSignature(
    next: MainCanvasSignature | undefined,
    previous: MainCanvasSignature | undefined,
): boolean {
    if (next === undefined || previous === undefined) return false;
    if (next.length !== previous.length) return false;
    for (let index = 0; index < next.length; index += 1) {
        if (!Object.is(next[index], previous[index])) return false;
    }
    return true;
}
