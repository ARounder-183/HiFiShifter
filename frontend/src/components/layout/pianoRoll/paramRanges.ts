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
 * dyn 乘性拖拽的**翻倍距离**：线性等价位移 0.5 个值单位 = ×2（+6 dB）。
 *
 * 【为什么拖拽必须是乘性的】dyn 曲线是倍率域（0 = 静音，1.0 = 0 dB），且原始
 * 素材里被静音的段落值恒为 0。线性加法拖拽会把静音拖出响度（0 + 0.2 = 0.2）；
 * 乘性拖拽（0 × 任何倍率 = 0）才符合"无声的地方仍然无声"的直觉 ——
 * 例如 ×2：原 0 → 0，原 0.2 → 0.4，原 0.5 → 1.0。
 *
 * 0.5 个值单位在默认视口（0..1）下为面板高度的 50%，即拖半屏高度 = ±6 dB，
 * 与线性参数的拖拽速度感接近。
 */
export const DYN_DRAG_DOUBLING_VALUE = 0.5;

/**
 * 把线性等价的拖拽位移（值单位）换算为 dyn 的**乘性系数**。
 *
 * `factor = 2^(Δ / DYN_DRAG_DOUBLING_VALUE)` —— Δ=+0.5 → ×2，Δ=−0.5 → ×0.5。
 * 指数映射保证细调（小位移 → 接近 1 的系数）与大范围（连续翻倍）手感一致。
 */
export function dynMultiplicativeFactor(linearDelta: number): number {
    const d = Number.isFinite(linearDelta) ? linearDelta : 0;
    return Math.pow(2, d / DYN_DRAG_DOUBLING_VALUE);
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
 * 增益按 `目标 / max(原声, 本下限)` 求得：既界定"无内容"（原声低于此值时
 * 增益有界，不会把噪声底放大成嘶声），又让增益关于原声**处处连续** ——
 * 后者是"近零原声处拉大动态 + 水平缩放出现随机伪影"的修复关键：
 * 若写成"低于门限就拒绝放大"，增益会在门限处阶跃（1 → 目标/门限），
 * 近零段的逐帧抖动会让相邻列的显示高度在"不可见"与"满高"之间随机切换。
 *
 * 下限必须远低于常见内容电平：真实素材的轻声、气声、尾音普遍在 −34…−55 dBFS。
 */
export const DYN_SILENCE_FLOOR = 0.001;

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
 * 由「原声电平」与「目标电平」求该帧的动态增益（**与后端
 * `common_params::compute_dyn_gain` 逐分支同构**）。
 *
 * 波形预览与实际渲染必须走同一份语义，否则用户看到的与听到的会分叉 ——
 * 这里曾有一份独立的本地实现（含自己的门限常量），任何一侧调整都会让
 * 「波形演示能提升、实际播放不提升」这类问题重新出现。判定收口在此处，
 * 后端改动时**必须同步本函数**（分支结构保持一一对应便于核对）。
 */
export function computeDynGain(target: number, orig: number): number {
    if (!Number.isFinite(target) || !Number.isFinite(orig)) return 1;
    if (target < 0) return 1; // 哨兵：沿用原声
    if (target <= 0) return 0; // 画静音：任何原声下都真静音
    // 分母钳到下限：界定"无内容" + 保证增益关于原声连续（见 DYN_SILENCE_FLOOR）。
    const denom = Math.max(orig, DYN_SILENCE_FLOOR);
    return Math.min(Math.max(target / denom, 0), DYN_MAX_GAIN);
}
