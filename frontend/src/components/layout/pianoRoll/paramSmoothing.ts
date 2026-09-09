// 底部参数面板「平滑度」的统一算法层。
//
// 背景
// ----
// 「平滑度」滑杆（edgeSmoothnessPercent，0-100）原本同时驱动三套互不一致的实现：
//   1. 选区编辑（移调/设音高等）的边缘淡化 —— PianoRollPanel 内联的「锚点直线滑移」；
//   2. 选区拖拽路径的边缘淡化 —— usePianoRollInteractions 里逐行重复的第二份实现；
//   3. 「平滑化」op / 右拖下压 / 绘制后自动平滑 —— 矩形核（box）多遍均值。
// 三者在量纲（% 选区长度 vs 帧数）、核形状（直线/矩形）、边界行为（截断窗、
// 跨未浊桥接）上互不一致，且存在「把未浊帧 pitch=0 混成非零」的哨兵破坏 bug。
//
// 本模块收敛为两个原语：
//   - applyEdgeBlend：delta 空间的边缘交叉淡化（曲线×曲线，C¹/C² 权重剖面）；
//   - smoothCurveGaussian：单次高斯核卷积（镜像边界、未浊分段、中值预清）。
// 所有时间参数以毫秒定义，经 framePeriodMs 换算为帧，与工程/模型的帧周期解耦。
//
// 组合性
// ------
// 边缘淡化在 delta 空间表达：out(f) = base(f) + w(f)·(edited(f) − base(f))，
// 其中 w 是以边界为中心、两侧各 halfSpan 帧的单调权重剖面（选区内侧 → 1，
// 外侧 → 0）。重复对同一选区做相同编辑时，各次的 w·delta 线性叠加 —— k 次 +2
// 移调与 1 次 +2k 移调在选区外产生完全相同的过渡带（旧锚点滑移会累积失真）。

/** 每侧过渡带最大宽度（毫秒）。100% 强度时选区边界两侧各 60ms 过渡（总带宽 ≤120ms）。 */
export const EDGE_HALF_SPAN_MAX_MS = 60;
/** 手绘结束后自动平滑的最大 σ（毫秒）。 */
export const DRAW_SMOOTH_SIGMA_MAX_MS = 40;
/** 「平滑化」op 的最小 σ（毫秒）。 */
export const SMOOTH_SIGMA_MIN_MS = 2;
/** 「平滑化」op 的最大 σ（毫秒）——对应强度 100%（u=1）。 */
export const SMOOTH_SIGMA_MAX_MS = 60;
/**
 * σ 的硬上限（毫秒）：右键下拖不设强度上限（与弹窗百分比显示一致，随拖拽
 * 距离按同一幂曲线继续增长），σ 达到该上限后不再增加。仅为数值/性能兜底 ——
 * 典型选区长度下，曲线在此之前早已视觉饱和（完全展平为趋势线）。
 */
export const SMOOTH_SIGMA_HARD_MAX_MS = 500;
/** 未浊缺口桥接上限（毫秒）：pitch 曲线中连续未浊段超过该时长时不跨段平滑。 */
export const GAP_BRIDGE_MAX_MS = 50;
/** 高斯核的截断半径（σ 的倍数）。 */
export const GAUSSIAN_CUTOFF_SIGMAS = 3;

/** 边缘淡化权重剖面形状。smoothstep 为 C¹ 连续（默认），quintic 为 C²。 */
export type EdgeShape = "smoothstep" | "quintic" | "linear";

export function clampToUnit(value: number): number {
    return Math.min(1, Math.max(0, Number(value) || 0));
}

/** pitch 参数的「可编辑值」判定：0 为未浊哨兵，任何平滑都不得改写/越过它。 */
export function editablePitchValue(v: number): boolean {
    return Number.isFinite(v) && v !== 0;
}

/**
 * 边缘淡化权重剖面：u∈[0,1]，0 = 完全保持原值，1 = 完全采用编辑值。
 * 单调递增（u=0.5 恒为 0.5，与边界帧折半的旧几何一致）。
 */
export function edgeBlendWeight(shape: EdgeShape, u: number): number {
    const t = clampToUnit(u);
    switch (shape) {
        case "linear":
            return t;
        case "quintic": {
            return t * t * t * (t * (t * 6 - 15) + 10);
        }
        case "smoothstep":
        default:
            return t * t * (3 - 2 * t);
    }
}

/**
 * 强度（0-100）→ 每侧过渡带半宽（帧）。
 * 时间尺度固定：每侧 ≤ strength% × EDGE_HALF_SPAN_MAX_MS；同时不超过 maxFrames
 * （调用方传 ⌊选区长度/4⌋，保证左右过渡带不重叠、短选区自动按比例收缩）。
 */
export function strengthToHalfSpanFrames(
    strengthPercent: number,
    framePeriodMs: number,
    maxFrames: number,
): number {
    const strength = Math.min(100, Math.max(0, Number(strengthPercent) || 0));
    if (strength <= 0) return 0;
    const fp = Math.max(1e-6, Number(framePeriodMs) || 5);
    const byTime = Math.round((strength / 100) * (EDGE_HALF_SPAN_MAX_MS / fp));
    return Math.max(0, Math.min(Math.max(0, Math.floor(maxFrames)), byTime));
}

/**
 * 选区编辑的每侧过渡带半宽（帧）。
 * 时间尺度固定：每侧 ≤ strength% × EDGE_HALF_SPAN_MAX_MS，同时不超过
 * maxFrames（调用方传 ⌊选区长度/4⌋，保证左右过渡带不重叠、短选区自动按
 * 比例收缩）。
 *
 * 注意：dense 索引可能经过降采样（stride > 1 时每个索引跨 stride 个音频帧），
 * 调用方应传 fpMs × stride 作为 framePeriodMs。
 */
export function edgeHalfSpanFramesForSelection(args: {
    strengthPercent: number;
    framePeriodMs: number;
    editedLen: number;
}): number {
    const editedLen = Math.max(0, Math.floor(Number(args.editedLen) || 0));
    const strength = Math.min(100, Math.max(0, Number(args.strengthPercent) || 0));
    if (strength <= 0 || editedLen <= 0) return 0;
    return strengthToHalfSpanFrames(
        strength,
        args.framePeriodMs,
        Math.floor(editedLen / 4),
    );
}

/**
 * 「平滑化」op / 右拖下压：强度单位 → 高斯 σ（毫秒）。
 *
 * 强度单位不限于 0-1：右键下拖随拖拽距离线性增大（-50px = 1.0，继续下拖
 * 按同一幂曲线继续加深），与 `formatRightDragMorphPercent` 的无限百分比显示
 * 一致（units = -percent/100）。u=1 → 60ms；u>1 沿同一曲线继续（C¹ 连续），
 * 直至 SMOOTH_SIGMA_HARD_MAX_MS 兜底。
 */
export function smoothSigmaMsFromUnits(units: number): number {
    const u = Math.max(0, Number(units) || 0);
    if (u <= 0) return 0;
    return Math.min(
        SMOOTH_SIGMA_HARD_MAX_MS,
        SMOOTH_SIGMA_MIN_MS + Math.pow(u, 1.4) * (SMOOTH_SIGMA_MAX_MS - SMOOTH_SIGMA_MIN_MS),
    );
}

/** 手绘后自动平滑：强度（0-100）→ 高斯 σ（毫秒）。 */
export function drawSmoothSigmaMsFromStrength(strengthPercent: number): number {
    const s = Math.min(100, Math.max(0, Number(strengthPercent) || 0)) / 100;
    if (s <= 0) return 0;
    return s * DRAW_SMOOTH_SIGMA_MAX_MS;
}

export type EdgeBlendArgs = {
    /** 编辑后的 dense（选区内已写入新值，选区外 = 原值）；本函数原地修改它。 */
    dense: number[];
    /** 编辑前的 dense（与 dense 同索引）。 */
    base: number[];
    editedStartIdx: number;
    editedLen: number;
    /** 每侧过渡带半宽（dense 索引；0 = 不淡化）。 */
    halfSpanFrames: number;
    /**
     * 编辑曲线延拓：返回任意帧（含选区外）的“编辑后期望值”。
     * 语义示例（setPitch）：常量 () => target —— 选区外向目标自然滑移。
     */
    editedAt?: (frame: number, baseValue: number) => number;
    /**
     * 编辑 delta 延拓：返回任意帧的“编辑增量”，目标 = base + delta。
     * 语义示例（常数移调）：常量 () => delta。
     * 两者都缺省时：取边界帧的实际增量常数延拓（量化类编辑的合理默认）。
     */
    editedDeltaAt?: (frame: number, baseValue: number) => number;
    shape?: EdgeShape;
    /** 哨兵判定（pitch: v!==0）。返回 false 的帧不做淡化（P0 修复）。 */
    isEditable?: (v: number) => boolean;
};

/**
 * 边缘交叉淡化（delta 空间，原地修改 dense）。
 *
 * 对左右边界各取带宽 [b−half, b+half]：选区内侧 w→1（保持编辑值），外侧 w→0
 * （保持原值），中间按权重剖面平滑过渡。目标曲线在选区内 = 已写入的编辑值；
 * 选区外 = editedAt / editedDeltaAt / 边界增量常数延拓（缺省）。
 * 任一侧为哨兵值（isEditable=false）的帧完全不写。
 */
export function applyEdgeBlend(args: EdgeBlendArgs): void {
    const half = Number(args.halfSpanFrames) || 0;
    if (!(half > 0) || args.editedLen <= 0) return;
    const dense = args.dense;
    const base = args.base;
    const n = Math.min(dense.length, base.length);
    if (n <= 0) return;
    const shape = args.shape ?? "smoothstep";
    const isEditable = args.isEditable ?? (() => true);
    const maxIdx = n - 1;
    const startIdx = Math.min(maxIdx, Math.max(0, Math.floor(args.editedStartIdx)));
    const endIdx = Math.min(maxIdx, Math.max(startIdx, Math.floor(args.editedStartIdx + args.editedLen - 1)));

    // 选区内已写入的编辑值快照（先取后写，避免读写交错）
    const edited = dense.slice(startIdx, endIdx + 1);
    // 边界帧的实际编辑增量（缺省延拓规则用）
    const leftDelta = edited[0] - base[startIdx];
    const rightDelta = edited[edited.length - 1] - base[endIdx];

    const targetAt = (f: number): number => {
        if (f >= startIdx && f <= endIdx) return edited[f - startIdx];
        if (args.editedAt) return args.editedAt(f, base[f]);
        if (args.editedDeltaAt) return base[f] + args.editedDeltaAt(f, base[f]);
        return base[f] + (f < startIdx ? leftDelta : rightDelta);
    };

    // 带宽 [b−half, b+half]；ascending：w 从外侧 0 升到内侧 1（左边界），
    // 否则从内侧 1 降到外侧 0（右边界）。
    const blendBand = (b: number, ascending: boolean) => {
        const lo = Math.max(0, Math.floor(b - half));
        const hi = Math.min(maxIdx, Math.ceil(b + half));
        for (let f = lo; f <= hi; f += 1) {
            const u = ascending
                ? (f - (b - half)) / (2 * half)
                : (b + half - f) / (2 * half);
            const w = edgeBlendWeight(shape, u);
            if (w <= 0) continue;
            if (!isEditable(base[f])) continue;
            const baseV = base[f];
            const target = targetAt(f);
            if (!Number.isFinite(target)) continue;
            dense[f] = baseV + (target - baseV) * w;
        }
    };

    if (startIdx > 0) blendBand(startIdx, true);
    if (endIdx < maxIdx) blendBand(endIdx, false);
}

export type GaussianSmoothOptions = {
    sigmaMs: number;
    framePeriodMs: number;
    /**
     * 样本有效性判定（如 pitch: v!==0 视为有声）。返回 false 的样本：
     * 输出保持原值，且不参与邻居的加权均值。
     */
    valueFilter?: (v: number) => boolean;
    /** 跨连续无效段的最大桥接长度（帧）；超过则硬切分段。缺省按 GAP_BRIDGE_MAX_MS 换算。 */
    gapBridgeFrames?: number;
    /**
     * 3 点中值预清孤立毛刺（默认**关闭**）。开启后孤立的单帧尖峰会被直接
     * 去除（去噪语义），而不是被高斯平均摊开 —— 只在明确想要“去毛刺”
     * 的场景（如共振峰轨迹）显式打开。
     */
    medianPrepass?: boolean;
    /**
     * 边界延拓：trend（默认，关于端点值的反对称镜像 —— 线性/二次趋势在端点
     * 精确保持，斜坡不内拉也不过冲）| hold（常数延拓）。两者仅在没有提供
     * leftContext/rightContext（真实上下文）时才参与。
     */
    boundary?: "trend" | "hold";
    /** 选区边界外侧的紧邻上下文值（leftContext 末位对应 values[−1]）。 */
    leftContext?: number[];
    /** 选区边界外侧的紧邻上下文值（rightContext 首位对应 values[n]）。 */
    rightContext?: number[];
};

/**
 * 单次高斯核曲线平滑（替代旧的矩形核多遍均值）。
 *
 * - 高斯核无副瓣：对周期成分（颤音）的衰减单调平坦，不会像 box 那样在部分
 *   频率几乎全消、部分频率只削 13dB；
 * - trend 反对称延拓 / 真实上下文：消除旧“截断窗”在端点把值拉向内部均值的
 *   偏置（斜坡端点精确保持）；
 * - 有效样本加权重归一：哨兵帧（pitch=0）既不参与均值也不被改写；
 * - 长未浊缺口硬切分段：不把两个短语桥接在一起。
 *
 * O(n·(2r+1))，r = ⌈3σ⌉；典型 σ ≤ 60ms/5ms = 12 帧 → 全曲 200k 帧约 15M 次乘加。
 */
export function smoothCurveGaussian(values: number[], opts: GaussianSmoothOptions): number[] {
    const n = values.length;
    if (n === 0) return [];
    const fp = Math.max(1e-6, Number(opts.framePeriodMs) || 5);
    const sigmaFrames = Math.max(0, Number(opts.sigmaMs) || 0) / fp;
    if (!(sigmaFrames >= 0.5)) return values.slice();

    const boundary = opts.boundary ?? "trend";
    const radius = Math.max(1, Math.ceil(GAUSSIAN_CUTOFF_SIGMAS * sigmaFrames));
    const valueFilter = opts.valueFilter;

    // 有效掩码
    const valid = new Array<boolean>(n);
    for (let i = 0; i < n; i += 1) {
        const v = values[i];
        valid[i] = valueFilter ? valueFilter(v) : true;
    }

    // 中值预清（3 点，仅有效帧参与）——只影响卷积输入，不改写无效帧
    let work = values.slice();
    if (opts.medianPrepass === true) {
        const med = values.slice();
        for (let i = 0; i < n; i += 1) {
            if (!valid[i]) continue;
            const a = i > 0 && valid[i - 1] ? work[i - 1] : work[i];
            const b = work[i];
            const c = i < n - 1 && valid[i + 1] ? work[i + 1] : work[i];
            med[i] = median3(a, b, c);
        }
        work = med;
    }

    // 分段：连续无效帧长度 > gapBridge 处硬切（不跨段取均值）
    const gapBridge = Math.max(
        0,
        Math.round(opts.gapBridgeFrames ?? GAP_BRIDGE_MAX_MS / fp),
    );
    const segmentOf = new Array<number>(n).fill(0);
    let seg = 0;
    let invalidRun = 0;
    for (let i = 0; i < n; i += 1) {
        if (!valid[i]) {
            invalidRun += 1;
            continue;
        }
        if (invalidRun > gapBridge) seg += 1;
        invalidRun = 0;
        segmentOf[i] = seg;
    }

    // 预构建延拓缓冲：[左 pad | 数据 | 右 pad]
    // 规则（按深度 d=1..radius，j=−d / n+d−1…）：
    //   1. 提供了真实上下文 → 直接用（有效性经 valueFilter 过滤）；
    //   2. 否则 trend：关于**边缘有效样本值**的反对称镜像，镜像源只沿
    //      “向内的有效样本序列”取（无效样本不进入镜像，避免把哨兵/缺口
    //      反射进均值）；源不足时退化为 hold；
    //   3. 边缘样本本身无效 → trend/hold 无意义，pad 判无效（该方向不参与）。
    const padded = new Array<number>(radius + n + radius);
    const paddedValid = new Array<boolean>(padded.length);
    const leftCtx = opts.leftContext;
    const rightCtx = opts.rightContext;
    const leftCtxLen = leftCtx ? leftCtx.length : 0;
    const rightCtxLen = rightCtx ? rightCtx.length : 0;

    const leftTrendSources: number[] = [];
    if (valid[0]) {
        for (let i = 1; i < n && leftTrendSources.length < radius; i += 1) {
            if (!valid[i]) break; // 不跨缺口镜像
            leftTrendSources.push(work[i]);
        }
    }
    const rightTrendSources: number[] = [];
    if (valid[n - 1]) {
        for (let i = n - 2; i >= 0 && rightTrendSources.length < radius; i -= 1) {
            if (!valid[i]) break; // 不跨缺口镜像
            rightTrendSources.push(work[i]);
        }
    }

    // trend/hold 延拓说明：反对称镜像对局部多项式趋势（线性/二次）精确，
    // 但在「边界旁有孤立尖峰且无真实上下文」的病态输入下会把尖峰镜像成
    // 反向瓣。平滑化 op 总是提供真实上下文（见调用方），右拖路径的选区
    // 数据也极少出现该形态，故不做值域钳制（钳制会杀死单调趋势延拓）。
    for (let d = 1; d <= radius; d += 1) {
        const p = radius - d; // j = -d
        let v = leftCtx && d <= leftCtxLen ? leftCtx[leftCtxLen - d] : undefined;
        let vValid = true;
        if (v !== undefined) {
            vValid = valueFilter ? valueFilter(v) : true;
        } else if (!valid[0]) {
            v = work[0];
            vValid = false;
        } else {
            const src = d <= leftTrendSources.length ? leftTrendSources[d - 1] : work[0];
            v = boundary === "hold" ? work[0] : 2 * work[0] - src;
        }
        padded[p] = v;
        paddedValid[p] = vValid;
    }
    for (let i = 0; i < n; i += 1) {
        padded[radius + i] = work[i];
        paddedValid[radius + i] = valid[i];
    }
    for (let d = 1; d <= radius; d += 1) {
        const p = radius + n + d - 1; // j = n + d - 1
        let v = rightCtx && d <= rightCtxLen ? rightCtx[d - 1] : undefined;
        let vValid = true;
        if (v !== undefined) {
            vValid = valueFilter ? valueFilter(v) : true;
        } else if (!valid[n - 1]) {
            v = work[n - 1];
            vValid = false;
        } else {
            const src = d <= rightTrendSources.length ? rightTrendSources[d - 1] : work[n - 1];
            v = boundary === "hold" ? work[n - 1] : 2 * work[n - 1] - src;
        }
        padded[p] = v;
        paddedValid[p] = vValid;
    }

    // 归一化高斯核
    const kernel = new Array<number>(2 * radius + 1);
    {
        const twoSigmaSq = 2 * sigmaFrames * sigmaFrames;
        let sum = 0;
        for (let k = -radius; k <= radius; k += 1) {
            const w = Math.exp(-(k * k) / twoSigmaSq);
            kernel[k + radius] = w;
            sum += w;
        }
        for (let i = 0; i < kernel.length; i += 1) kernel[i] /= sum;
    }

    const out = values.slice();
    for (let i = 0; i < n; i += 1) {
        if (!valid[i]) continue;
        const segId = segmentOf[i];
        const center = radius + i;
        let acc = 0;
        let wsum = 0;
        for (let k = -radius; k <= radius; k += 1) {
            const p = center + k;
            if (p < 0 || p >= padded.length) continue;
            if (p >= radius && p < radius + n) {
                const j = p - radius;
                if (!paddedValid[p] || segmentOf[j] !== segId) continue;
            } else if (!paddedValid[p]) {
                continue;
            }
            const w = kernel[k + radius];
            acc += w * padded[p];
            wsum += w;
        }
        out[i] = wsum > 0 ? acc / wsum : work[i];
    }
    return out;
}

function median3(a: number, b: number, c: number): number {
    if (a > b) {
        if (b > c) return b;
        return a > c ? c : a;
    }
    if (a > c) return a;
    return b > c ? c : b;
}
