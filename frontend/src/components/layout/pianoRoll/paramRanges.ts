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
 * 动态值的存储值域（与后端描述符一致：0..2 倍率）。
 *
 * 【为什么上限是 2】dyn 的值是**目标电平**，参考电平 = 轨道组最响的持续段落
 * （99 百分位）→ 99% 的帧天然 ≤ 1.0。画到 0 dB 以上 = 把某段抬得比全组最响
 * 段落还响，正常编辑几乎不会发生；用户真正需要雕琢的是 0 dB 以下直到静音
 * （−∞ dB）的空间。上限 2.0（+6 dB）只是极少见的提升余量；对安静段的更大
 * 提升由后端增益上限（`DYN_MAX_GAIN`，×4）实际接管，与目标值域无关。
 *
 * 【与显示视口的区别】默认显示视口见 {@link DYN_DEFAULT_VIEW}（0..1.25）；
 * 本常量是缩放/写入的硬边界。
 */
export const DYN_VALUE_MIN = 0;
export const DYN_VALUE_MAX = 2;

/**
 * volume 的**默认值域视口**：显示 0..2 —— 与存储值域（描述符 0..2，±6 dB）
 * 一致，1.0（= 不增不减）落在面板垂直中线；>1 的提升与 <1 的衰减对称各占
 * 一半高度。只在用户未自定义过该参数视口时使用。
 */
export const VOLUME_DEFAULT_VIEW = { center: 1.0, span: 2.0 } as const;

/**
 * dyn 的**默认值域视口**：显示 0..1.25，1.0（= 0 dB）位于面板上部 80% 处。
 *
 * 面板的绝大部分高度留给 0 dB 以下（−∞..0 dB）—— 这正是用户需要着重编辑的
 * 区间；顶部 20% 只是极少量"比最响段落还响"的余量。用户仍可缩放看到全值域
 * 0..2。只在用户未自定义过该参数视口时使用。
 */
export const DYN_DEFAULT_VIEW = { center: 0.625, span: 1.25 } as const;

/**
 * dyn 乘性拖拽的**翻倍距离**：线性等价位移 0.5 个值单位 = ×2（+6 dB）。
 *
 * 【为什么拖拽必须是乘性的】dyn 曲线是倍率域（0 = 静音，1.0 = 0 dB），且原始
 * 素材里被静音的段落值恒为 0。线性加法拖拽会把静音拖出响度（0 + 0.2 = 0.2）；
 * 乘性拖拽（0 × 任何倍率 = 0）才符合"无声的地方仍然无声"的直觉 ——
 * 例如 ×2：原 0 → 0，原 0.2 → 0.4，原 0.5 → 1.0。
 *
 * 0.5 个值单位在默认视口（0..1.25）下约为面板高度的 40%，即拖 40% 面板
 * 高度 = ±6 dB，与线性参数的拖拽速度感接近。
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

/** 动态增益的静音保护下限（与后端 `DYN_MIN_REF` 对应，仅用于 UI 提示）。 */
export const DYN_MIN_REF = 0.05;

/** 动态增益上限（与后端 `DYN_MAX_GAIN` 对应，仅用于 UI 提示）。 */
export const DYN_MAX_GAIN = 4.0;
