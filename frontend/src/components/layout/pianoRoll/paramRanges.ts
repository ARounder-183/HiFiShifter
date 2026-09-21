/**
 * 参数值域的共享常量（前端侧）。
 *
 * 【为什么需要这个文件】参数值域的唯一真源在后端描述符
 * (`get_processor_params`)，前端绝大多数场景都应从描述符读取。
 * 但**动态（DYN）的"无数据"哨兵**必须在"没有描述符在手"的代码路径里
 * 也能表达（例如上下文菜单构造写回值、互转计划的归一化目标），
 * 因此把该常量固化在这里，与后端
 * `renderer::common_params::DYN_FOLLOW_ORIG` 一一对应。
 *
 * ⚠ 改动此值必须同步后端常量 —— 两者不一致会导致"初始化后响度异常"。
 */

import {
    CHILD_FORMANT_OFFSET_CENTS_RANGE,
    isChildFormantOffsetCentsParam,
} from "./childPitchOffsetParams";

/**
 * 动态（DYN）曲线的「沿用原声」哨兵值。
 *
 * 语义：该帧不做任何电平改变（增益 1.0）。值本身是负的 —— 因为 0..4 全部
 * 是合法目标电平（0 = 全静音，1.0 = 参考电平），"不改变"只能用值域外的
 * 负数表达。
 *
 * 后端在 `get_param_frames` 出口把它解析成真实原声电平，因此**前端读到的
 * 曲线永远不含负数**；只有在写回时才需要显式使用它。
 */
export const DYN_FOLLOW_ORIG = -1;

/** 动态参数 id。 */
export const DYN_PARAM_ID = "dyn";

/**
 * 动态参数的历史别名（旧会话持久化里可能出现）。
 *
 * 后端描述符只有 `"dyn"`；新代码一律用 {@link isDynParam} 判定，不要再扩散
 * 对别名的字面量比较。
 */
export const DYN_LEGACY_PARAM_ID = "dyn_edit";

/** 音量参数 id。 */
export const VOLUME_PARAM_ID = "volume";

/**
 * 判定参数是否为动态（含历史别名）。
 *
 * 【为什么收敛成一个函数】`param === "dyn" || param === "dyn_edit"` 这对字面量
 * 比较散布在面板 / 菜单 / 内核 / 渲染层至少五处；漏改一处（例如只加了 `dyn`）
 * 就会出现「动态面板显示成音量标尺」一类的分裂行为。判定收口在这里，
 * 后续若别名彻底退场也只改这一处。
 */
export function isDynParam(param: string | null | undefined): boolean {
    return param === DYN_PARAM_ID || param === DYN_LEGACY_PARAM_ID;
}

/**
 * 动态值的存储值域（与后端描述符一致：0..1 倍率）。
 *
 * 【倍率的锚点是绝对的】1.0 = 0 dBFS（数字满量程），0.5 = −6 dBFS，0 = 静音 ——
 * 与 DAW 的峰值电平表同一坐标系。因此别处测得 −4.7 dB 的一段，在这里就是
 * 0.582。**不做**任何"相对本轨道组最响段落"的归一化 —— 那会让倍率失去绝对
 * 意义（历史实现即把 0.582 读成 1.126）。
 *
 * 【为什么上限是 1】锚点是满量程，超过 1.0 的电平在播出去之前就会被削顶，
 * 画上去没有可兑现的物理意义 —— 那部分值域只会变成有效的无效编辑区。真正
 * 需要雕琢的是 0 dBFS 以下直到静音的整段空间，因此把值域收成 0..1，
 * 面板高度全部留给有意义的部分。
 *
 * 【与显示视口的区别】默认显示视口见 {@link DYN_DEFAULT_VIEW}（0..1）；
 * 本常量是缩放/写入的硬边界。
 */
export const DYN_VALUE_MIN = 0;
export const DYN_VALUE_MAX = 1;

/**
 * volume 的**默认值域视口**：显示 0..2 —— 与存储值域（描述符 0..2，±6 dB）
 * 一致，1.0（= 不增不减）落在面板垂直中线；>1 的提升与 <1 的衰减对称各占
 * 一半高度。只在用户未自定义过该参数视口时使用。
 */
export const VOLUME_DEFAULT_VIEW = { center: 1.0, span: 2.0 } as const;

/**
 * dyn 的**默认值域视口**：显示 0..1 —— 与存储值域完全一致。
 *
 * 锚点是满量程：0 dBFS（1.0）既是值域顶端也是音量天花板，居中偏上的留白
 * 没有物理意义（>1 会削顶）。因此默认视口直接铺满整个值域，既不裁掉可用
 * 区间，也不需要用户缩放才能看到全貌。
 *
 * 只在用户未自定义过该参数视口时使用。
 */
export const DYN_DEFAULT_VIEW = { center: 0.5, span: 1.0 } as const;

/**
 * dyn **增益档位**命令的翻倍距离：值域位移 0.5 个值单位 = ×2（+6 dB）。
 *
 * 【只服务于"命令"，不用于拖拽】`dynMultiplicativeFactor` 现在唯一的调用点是
 * `shiftParamUpSelection` / `shiftParamDownSelection` 这类**菜单/快捷键档位
 * 命令**（一次 = ×2 / ×0.5 = ±6 dB，与 DAW 的"增益 ±6 dB"同一语义）。
 *
 * 拖拽**不用**固定灵敏度：固定的 `k = 2^(Δ/0.5)` 与锚点位置无关，锚点位移
 * `A·(k−1)` 正比于 A —— 值越小动得越少（0.05 拖半屏只到 0.1），用户观感是
 * "不跟手"。拖拽改用由锚点导出的系数（见 {@link dynDragScaleFactor}），
 * 仍保持乘性语义。
 */
export const DYN_GAIN_STEP_DOUBLING_VALUE = 0.5;

/**
 * 把值域位移换算为 dyn 的**乘性系数**（供增益档位命令使用）。
 *
 * `factor = 2^(Δ / DYN_GAIN_STEP_DOUBLING_VALUE)` —— Δ=+0.5 → ×2，Δ=−0.5 → ×0.5。
 * 指数映射保证细调（小位移 → 接近 1 的系数）与大范围（连续翻倍）手感一致。
 *
 * ⚠ **不要用于拖拽**：见 {@link DYN_GAIN_STEP_DOUBLING_VALUE} 的说明。
 */
export function dynMultiplicativeFactor(linearDelta: number): number {
    const d = Number.isFinite(linearDelta) ? linearDelta : 0;
    return Math.pow(2, d / DYN_GAIN_STEP_DOUBLING_VALUE);
}

/**
 * 动态选区拖拽的**锚点缩放系数**：让「被抓取的那条线」恰好跟手。
 *
 * 【推导】设拖拽起点处被抓住的那条线的值为 `A`，指针在值域纵轴上的位移为 `Δ`。
 * 乘性缩放把选区整体乘以 `k`，于是锚点被移动到 `A · k`。要让它跟在光标下
 * （"跟手"），只需
 *
 *     A · k = A + Δ   ⟹   k = (A + Δ) / A
 *
 * 这个 `k` 就是**由拖拽参数线的位置决定的额外幅度逻辑**：
 *
 * - `k` 随锚点值 `A` 变化 —— 从很低的线（A 小）往上拖，`k` 自动变大，于是
 *   虽然乘性缩放本身按比例施加，锚点位移仍然**逐点等于指针位移**，不再出现
 *   "拉了半屏、线只挪一点点"；
 * - 选区的其它点按同一 `k` 缩放，倍率域的**相对关系保留**（0.05/0.10 → 0.10/0.20）；
 * - 静音帧（`0 × k = 0`）仍然保持静音 —— 乘性语义原有的"无声的地方仍然无声"
 *   不变。
 *
 * 【为什么不是"固定灵敏度"】旧实现用 `k = 2^(Δ / 0.5)`：系数只与位移有关、
 * 与锚点位置无关，于是锚点位移 `A · (k − 1) ∝ A` —— 值越小动得越少。本函数
 * 把 `A` 引进系数，正是补上这一项。
 *
 * 【下界】`Δ ≤ −A` 时锚点已到值域底部，`k` 取 0（整段缩到静音），不再继续变负
 * （负系数会把选区上下翻转）。这与"把锚点拖到 0"一致。
 *
 * @param anchorValue 拖拽起点处被抓住的那条线的值（动态值域）。
 * @param valueDelta 指针在值域上的位移（可为负）。
 * @returns 缩放系数；**锚点贴地**（`≤ DYN_SILENCE_FLOOR`）时返回 `null` ——
 *   此时 `0 × k = 0` 使乘性缩放**无论如何都动不了锚点**，语义上无解，
 *   调用方应退回值域内线性偏移（见 {@link shiftValueForDrag}）。
 */
export function dynDragScaleFactor(anchorValue: number, valueDelta: number): number | null {
    if (!Number.isFinite(anchorValue) || anchorValue <= DYN_SILENCE_FLOOR) return null;
    if (!Number.isFinite(valueDelta)) return 1;
    return Math.max(0, (anchorValue + valueDelta) / anchorValue);
}

/**
 * 动态选区拖拽的**逐帧结果**（完整法则，供调用方直接用）。
 *
 * 常规情况：以"被抓住那条线的值 `anchorValue`"为锚点做乘性缩放，锚点恰好跟手
 * （推导见 {@link dynDragScaleFactor}）；锚点贴地（抓住的本身就是静音）时乘性
 * 缩放对 `0` 无解，退回值域内线性偏移。
 *
 * 【为什么单独成一个函数】预览与提交必须逐值一致，而"系数由锚点导出 + 边界退回"
 * 这两步合起来才是完整法则。收在这里既保证两条路径同源，也让法则本身可单测
 * （见 paramRanges.test.ts）。
 *
 * @param orig 该帧原始值。
 * @param anchorValue 拖拽起点处被抓住的那条线的值。
 * @param valueDelta 指针在值域上的位移。
 * @returns 该帧的新值（已钳到动态值域）。
 */
export function shiftDynValueForDrag(
    orig: number,
    anchorValue: number,
    valueDelta: number,
): number {
    const scale = dynDragScaleFactor(anchorValue, valueDelta);
    if (scale !== null) return clampParamWriteValue(DYN_PARAM_ID, orig * scale);
    return shiftValueForDrag(DYN_PARAM_ID, orig, valueDelta);
}

/**
 * 拖拽的**逐帧值偏移**（值域内线性偏移），然后钳到后端会接受的值域。
 *
 * 【谁用它】音高 / 张力 / 各偏移量这类**非比值**参数 —— 它们的"跟手"就是
 * `原值 + Δ`；动态这种比值域参数走 {@link dynDragScaleFactor} 的锚点缩放，
 * 只有**锚点贴地**（被抓取那条线本身是静音，乘性无解）时才退回本函数。
 *
 * 【为什么线性就是"跟手"】拖拽位移取自指针在**值域纵轴**上的位移（Δ = 指针值
 * − 按下时的指针值），纵轴是线性刻度，故 `原值 + Δ` 恰好让被抓取的那一点停在
 * 光标下（直接操纵）。
 *
 * 【动态走本函数的两种情况】锚点贴地时退回这里；此时是线性偏移，静音帧会
 * 被抬起来（`0 + Δ = Δ`）——这是"线要跟着光标"与"静音保持静音"在零点的
 * 固有冲突，线性法则下只能选前者。常规情况（锚点 > 下限）走乘性缩放，
 * 静音保持静音。
 *
 * 【往下拖的边界】动态拖到 0 以下收敛到 0（"静音"），**不**套用后端写入入口的
 * "负值 = 沿用原声哨兵"规则 —— 否则往下拖会变成"什么都不改"，与手势意图相反。
 *
 * @param param 参数名（决定值域）。
 * @param orig 该帧的原始值。
 * @param valueDelta 指针在值域上的位移（可为负）。
 * @returns 偏移并钳制后的值。
 */
export function shiftValueForDrag(param: string, orig: number, valueDelta: number): number {
    if (!Number.isFinite(orig)) return orig;
    if (!Number.isFinite(valueDelta)) return clampParamWriteValue(param, orig);
    const shifted = orig + valueDelta;
    // 【拖拽的边界语义 ≠ 后端写入的边界语义】动态的负值在后端写入入口表示
    // 「沿用原声」**哨兵**（写回未编辑帧时用），但用户把动态值**拖到 0 以下**
    // 的意思是"这里静音" —— 若套用哨兵规则，往下拖会变成"什么都不改"，
    // 与手势意图正好相反。故拖拽在动态上把负值收敛到值域下限 0。
    // （拖拽的输入不会含哨兵：读回的曲线里哨兵已被后端解析成原声基线。）
    if (isDynParam(param) && shifted < 0) return DYN_VALUE_MIN;
    return clampParamWriteValue(param, shifted);
}

/**
 * 批量写回时的**哨兵保留**（dyn 专用，就地修改并返回）。
 *
 * 【为什么需要】dyn 的"未画"帧在 `get_param_frames` 出口被解析成原声基线，
 * 前端看到的 edit 无法区分"用户画的 0.8"与"未画、基线恰为 0.8"。平滑/量化/
 * 平均/拖拽提交这类"读-变换-写回"操作若直接写回，会把未画帧**物化**成显式
 * 目标电平 —— 当下听感不变（增益 = 基线/基线 = 1），但日后 clip 移动触发基线
 * 重分析时，这些帧不再跟随、响度静默漂移。写回前把位图标记的帧恢复成哨兵
 * （负值，`set_param_frames` 入口原样接受），"未画"语义得以跨操作存活。
 *
 * @param values 变换后的写回值（逐帧）。
 * @param sentinels 后端 `edit_sentinel` 位图（与 values 逐帧对齐；undefined =
 *   旧载荷/非 dyn，原样返回不做任何事）。
 */
export function restoreDynSentinels(
    values: number[],
    sentinels: readonly boolean[] | undefined,
): number[] {
    if (!sentinels) return values;
    const n = Math.min(values.length, sentinels.length);
    for (let i = 0; i < n; i += 1) {
        if (sentinels[i]) values[i] = DYN_FOLLOW_ORIG;
    }
    return values;
}

/**
 * 动态增益的**电平分母下限**：−60 dBFS（与后端 `DYN_SILENCE_FLOOR` 一一对应）。
 *
 * 增益按 `目标 / max(原声, 本下限) × 无内容淡出(原声)` 求得（见
 * `computeDynGain` / `noContentFade`）：
 * - **钳分母**：界定"无内容"（原声低于此值时增益有界）并让增益关于原声
 *   **处处连续** —— 后者是"近零原声处拉大动态 + 水平缩放出现随机伪影"的
 *   修复关键：若写成"低于门限就拒绝放大"，增益会在门限处阶跃，
 *   近零段的逐帧抖动会让相邻列的显示高度在"不可见"与"满高"之间随机切换；
 * - **无内容淡出**：仅有"有界"还不够 —— −90 dBFS 的抖动被 ×500 放大后仍达
 *   −36 dBFS（可闻嘶声）。下限的语义本就是"无内容"，故低于它时按 smoothstep
 *   淡出到静音（门限处导数也连续，不引入新的不连续）。
 *
 * 下限必须远低于常见内容电平：真实素材的轻声、气声、尾音普遍在 −34…−55 dBFS。
 */
export const DYN_SILENCE_FLOOR = 0.001;

/**
 * 无内容淡出的**下端点**：下限的 0.5 倍（−66 dBFS）。
 *
 * 【为什么不是"从 0 一路淡出"】淡出若跨越 [0, 下限]（线上式 ⇒ 对数轴上从 −∞ 到
 * −60 dB 的整段），中点落在 −66 dB 附近 ⇒ 而"看起来几乎是 0"的段落（线性值
 * 0.0005 量级）正好落在中点，淡出只压掉一半、仍会把它抬高到目标电平附近（用户
 * 报告的近零伪影）。真实的**有内容**素材在 −34…−55 dBFS（下限之上），与
 * −66…−90 dBFS 的抖动噪声底之间有 10 dB 以上的空档 —— 故把过渡带收紧到
 * `[−66, −60] dBFS`：空档之下的判据恒 0、真实内容恒 1，两者都不受影响。
 */
export const DYN_CONTENT_FADE_FLOOR_RATIO = 0.5;

/**
 * 动态增益上限：**从下限兑现到值域顶端**所需的倍数（= `DYN_VALUE_MAX / 下限`）。
 *
 * 与后端 `DYN_MAX_GAIN` 同一定义 —— 它不是独立的策略旋钮，而是
 * `DYN_SILENCE_FLOOR` 的推论：分母被钳到下限后，增益的上界就是
 * `目标/下限 ≤ 值域顶端/下限`。该值对正常输入不可达（精确兑现不会削顶：
 * 输出峰值 `= 原声 × (目标/原声) = 目标 ≤ 满量程`），只是数值兜底。
 *
 * 历史上限 ×4（仅 +12 dB）远不够把 −34 dBFS 提到 −4.7 dBFS 所需的 ×29，
 * 表现为"画了目标却达不到"。
 */
export const DYN_MAX_GAIN = DYN_VALUE_MAX / DYN_SILENCE_FLOOR;

/**
 * 动态增益的**电平对齐**部分（不含无内容淡出）：`目标 / max(原声, 下限)`，钳到上限。
 *
 * 与 {@link dynContentFade} 相乘即 {@link computeDynGain}（唯一真源仍是那个组合；
 * 后端 `common_params::compute_dyn_gain` 是同一份语义）。
 *
 * 【为什么单独导出】让**显示端**能把"淡出"的输入换成**桶级一致**的平滑原声，而
 * 电平对齐仍用点采样原声。原因：音频逐样本求值，点采样原声就够了；而波形是按
 * mipmap 桶绘制的（粗缩放 L2 桶 ≈ 85ms），若淡出也用桶内某一帧的点采样，那么
 * "桶峰"与"淡出"来自**不同时刻**，而淡出随原声陡变 ⇒ 显示随缩放随机起伏
 * （用户报告的"近零原声处缩小后出现随机伪影"）。
 */
export function dynLevelTargetingGain(target: number, orig: number): number {
    if (!Number.isFinite(target) || !Number.isFinite(orig)) return 1;
    if (target < 0) return 1; // 哨兵：沿用原声
    if (target <= 0) return 0; // 画静音：任何原声下都真静音
    // 分母钳到下限：界定"无内容" + 保证增益关于原声连续（见 DYN_SILENCE_FLOOR）。
    const denom = Math.max(orig, DYN_SILENCE_FLOOR);
    return Math.min(Math.max(target / denom, 0), DYN_MAX_GAIN);
}

/**
 * 动态增益（**唯一真源**）：电平对齐 × 无内容淡出。
 *
 * 【为什么要淡出】下限**以下**的原声只是抖动噪声底（16bit 抖动 ≈ −90 dBFS），
 * 按"目标电平"放大只会把噪声变成可听的嘶声（实测 −90 dB 原声 + 目标 0.5 →
 * 输出 −36 dBFS）。下限的语义本就是"无内容"，故低于它时按 smoothstep 平滑淡出到
 * 静音（门限处导数也连续，不引入阶跃伪影 —— 这正是更早的"低于门限就拒绝放大"
 * 被否掉的原因）。
 *
 * 【与后端的同构】后端 `common_params::compute_dyn_gain` 逐分支同构；后端改动时
 * **必须同步本函数**（这是"波形演示与实际播放一致"的前提）。
 */
export function computeDynGain(target: number, orig: number): number {
    return dynLevelTargetingGain(target, orig) * dynContentFade(orig);
}

/**
 * 「原声是否算作**有内容**」的平滑度（与后端同构）：下限之上恒 1，之下 smoothstep 淡出到 0。
 *
 * 【为什么不是硬门限】硬门限会在门限处产生阶跃（增益 1 → 目标/门限），近零原声
 * 段的逐帧基线抖动会让它变成随机咔哒 —— 这正是 `DYN_SILENCE_FLOOR` 当初拒绝
 * "低于门限就不再放大"的理由。smoothstep 在门限处**导数也连续**。
 *
 * 【为什么以原声（而非目标）为准】"有没有内容"是素材的属性，与用户画多高无关：
 * 同一段抖动噪声底，拉高目标不该变嘶声，拉低目标也不该变成"被压的嘶声"。
 *
 * 【对未画帧】未画帧的增益是 `原声 / max(原声, 下限)`：下限之上恰为 1；下限之下
 * < 1（"无内容处淡出到静音"，与下限"界定无内容"的语义一致）。真实素材的轻声/
 * 气声/尾音普遍在 −34…−55 dBFS，都在下限之上，不受影响。
 *
 * 【显示端为什么要喂它"平滑原声"】见 {@link dynLevelTargetingGain} 的说明。
 */
export function dynContentFade(orig: number): number {
    if (!(orig > 0)) return 0;
    const lo = DYN_SILENCE_FLOOR * DYN_CONTENT_FADE_FLOOR_RATIO;
    const x = Math.min(Math.max((orig - lo) / (DYN_SILENCE_FLOOR - lo), 0), 1);
    return x * x * (3 - 2 * x);
}

/**
 * 把**写回**的参数值钳制到后端会接受的值域（**与后端
 * `commands::params::set_param_frames` 的写入分支逐条同构**）。
 *
 * 【为什么前端也必须钳】后端在写入口按参数语义钳制（见 Rust 侧的同名分支，
 * 分支顺序与常量都与本函数一一对应），而前端过去对"用户输入 → 曲线值"这条
 * 路径**完全不钳** —— 只有值域视口的 `yToValue` 顺带夹了一下**指针位置**。
 * 于是拖拽预览可以画出后端不会接受的值：
 *
 * - 选区上拖时音量走 `orig + Δ`、动态走 `orig × 2^(Δ/0.5)`，两者都能跑出存储
 *   值域（音量 0..2、动态 0..1）。参数线因为画布裁切看着"停在顶端"，但**波形**
 *   按超出值放大（`volume × dynGain` 直接相乘），松手后后端把它们钳回来，
 *   波形于是跳回去 —— 用户表现为"拖的时候波形超了，松手又弹回"。
 *
 * 修复原则：**预览值与提交值必须逐值一致**。做法不是在提交前改写数据（后端已经
 * 钳过一次），而是在"用户输入 → live 覆盖"的写入点用同一份钳制函数，让预览从
 * 一开始就落在后端会接受的范围内。
 *
 * ⚠ 非有限值不在此处理：后端对非有限值有独立的兜底语义（额外曲线取参考值、
 * 音高/齿度取 0），前端没有等价信息（`param_reference_value` 在 Rust 侧解析），
 * 故原样透传交由后端决定。前端各写入路径本就会先过滤非有限值。
 *
 * ⚠ 后端改动本组分支时**必须同步本函数**（保持顺序与常量一一对应便于核对）。
 *
 * @param param 参数名（可含子轨前后缀）。
 * @param value 待写入的曲线值。
 * @returns 钳制后的值；参数不在这组分之内时原样返回。
 */
export function clampParamWriteValue(param: string, value: number): number {
    if (!Number.isFinite(value)) return value;
    let v = value;
    // 子轨共振峰偏移：与 `parse_child_pitch_offset_param` 的 Formant 分支同值域。
    if (isChildFormantOffsetCentsParam(param)) {
        v = clampTo(v, CHILD_FORMANT_OFFSET_CENTS_RANGE.min, CHILD_FORMANT_OFFSET_CENTS_RANGE.max);
    }
    if (param === "pitch") {
        // 0 是"未设置"哨兵，绝不能被钳进 1..127。
        if (v !== 0) v = clampTo(v, 1, 127);
        return v;
    }
    if (param === "tension") {
        return clampTo(v, -100, 100);
    }
    // 音量：乘性增益，负值对音量无意义（一并钳到 0 = 全静音）。
    if (param === "volume" || param === "hifigan_volume") {
        return clampTo(v, 0, 2);
    }
    if (isDynParam(param)) {
        // 负值统一收敛到「沿用原声」哨兵 —— 不允许写成 −0.7 之类的中间值；
        // 值本身无意义，只要符号为负就是同一个语义。上界与描述符值域逐字一致。
        if (v < 0) return DYN_FOLLOW_ORIG;
        return clampTo(v, DYN_VALUE_MIN, DYN_VALUE_MAX);
    }
    return v;
}

/** 局部 min/max 钳制（避免为一次钳制引入 utils 依赖）。 */
function clampTo(value: number, min: number, max: number): number {
    return value < min ? min : value > max ? max : value;
}
