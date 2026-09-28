import { test } from "vitest";

import { buildWaveformGeometry } from "./geometry.ts";
import type { WaveformScene } from "./sceneBuilder.ts";
import { makeLoudnessAmplitudeMap } from "../components/layout/pianoRoll/PianoRollWaveformSurface.tsx";

/**
 * 近零原声 + 大目标电平：**幻峰**回归（粗缩放下的随机伪影）。
 *
 * ## 故障形态
 *
 * 把"原始动态值几乎接近 0"处的动态拉大，再水平缩小视图，波形在该处出现
 * 随机伪影（满高尖刺与空洞交替，位置随缩放改变）。
 *
 * ## 根因（与渲染逻辑直接相关）
 *
 * 波形每列的峰值取自 **mipmap 桶**：粗缩放走 L2（div=4096 ≈ **85 ms**），
 * 而动态的原声基线是 **20 ms** 窗的估计 —— **分子窗口比分母窗口宽 4 倍**。
 * 于是"桶内最大样本"可以大于"用来求增益的那一点的原声基线"，显示值
 * `桶峰 × 目标/基线` 会**超过目标电平**。
 *
 * 近零区把目标拉大后，`目标/基线` 极大（基线 0.0002 时 ×1000，直接撞上限），
 * 显示值对桶峰变得极度敏感 —— 哪个桶"侥幸"含稍大的样本，哪一列就成满高
 * 幻峰；水平缩放改变桶的划分方式，幻峰位置随之**随机变化**。
 *
 * ## 物理界（本测试的判据）
 *
 * 逐样本地 `|x(s)| ≤ 原声基线(t_s)`，因此
 *   真实输出 `= |x(s)| × vol × 目标/基线 ≤ vol(t_s) × 目标(t_s)`。
 * **超过该值的显示成分不可能被听到**，必然是幻峰。本测试即以
 * `max over 列窗 (vol × 目标)`（× clip 增益）为上界逐列断言。
 *
 * ## 修复为何不影响真实内容
 *
 * 未编辑区 `目标 = 原声`，上界 = `vol × max(原声)`，而显示用的桶峰
 * ≤ 该窗内的原声基线最大 ≤ 上界 ⇒ 钳制恒不触发（下方第二条断言钉住这点：
 * 未编辑区在**任何**缩放下都必须逐像素等于源包络）。
 */
const SR = 48000;
const FRAME_MS = 5;
const DIV = 4096; // 粗缩放命中的 L2 桶宽（≈85ms）——伪影的触发条件
const DUR = 60;
const T0 = 20; // 20s 起进入近零段
const T1 = 40; // 40s 回到稳态
/**
 * 近零段里用户画的目标电平。
 *
 * 取 0.3（中等值）而非 1.0：满量程目标会让幻峰撞上波形矩形的削顶而被掩盖，
 * 观测不到越界（见 buildScenario 内的说明）。
 */
const NEAR_ZERO_TARGET = 0.3;

/** 场景：稳态(0-20) → 近零噪声(20-40) → 稳态(40-60)。 */
function buildScenario(): {
    bMax: Float32Array;
    bMin: Float32Array;
    baseline: Float32Array;
    target: Float32Array;
    nFrames: number;
} {
    const total = Math.round(DUR * SR);
    let seed = 1234567;
    const rnd = (): number => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
    };
    const audio = new Float32Array(total);
    const frameSamples = Math.round((FRAME_MS / 1000) * SR);
    let amp = 1e-4;
    for (let i = 0; i < total; i += 1) {
        const t = i / SR;
        if (t < T0) {
            audio[i] = 0.25 * Math.sin(2 * Math.PI * 220 * t);
        } else if (t < T1) {
            // 近零噪声：振幅随机（10^-3.5 … 10^-2.7），白噪
            if ((i - Math.round(T0 * SR)) % frameSamples === 0) {
                amp = Math.pow(10, -3.5 + 0.8 * rnd());
            }
            audio[i] = amp * (rnd() * 2 - 1);
        } else {
            audio[i] = 0.3 * Math.sin(2 * Math.PI * 180 * t);
        }
    }

    // 原声基线：20ms 窗峰值、帧中心锚定（与后端 compute_frame_levels 同口径）。
    const nFrames = Math.round((DUR * 1000) / FRAME_MS);
    const baseline = new Float32Array(nFrames);
    const half = Math.round(0.01 * SR);
    for (let h = 0; h < nFrames; h += 1) {
        const c = Math.round(((h + 0.5) * FRAME_MS * SR) / 1000);
        let p = 1e-9;
        for (let i = Math.max(0, c - half); i < Math.min(total, c + half); i += 1) {
            const a = Math.abs(audio[i]);
            if (a > p) p = a;
        }
        baseline[h] = p;
    }

    // 目标电平：未编辑区由后端把哨兵解析成基线（= base）；
    // 近零区（20..40s）是用户画的目标。
    //
    // 【必须用**中等**目标值（0.3），不能用 1.0】目标取满量程时幻峰会撞上
    // 波形矩形的削顶（±1 钳制）而被掩盖，观测不到 —— 本测试的判据是
    // "显示不得超过可达上限"，若上限恰好等于矩形边界就失去判别力。
    const target = new Float32Array(nFrames);
    for (let h = 0; h < nFrames; h += 1) {
        const t = ((h + 0.5) * FRAME_MS) / 1000;
        target[h] = t >= T0 && t < T1 ? NEAR_ZERO_TARGET : baseline[h];
    }

    // mipmap 桶峰值（div 粒度），与显示管线一致。
    const bucketCount = Math.ceil(total / DIV);
    const bMax = new Float32Array(bucketCount);
    const bMin = new Float32Array(bucketCount);
    for (let b = 0; b < bucketCount; b += 1) {
        let mx = 0;
        let mn = 0;
        for (let i = b * DIV; i < Math.min(total, (b + 1) * DIV); i += 1) {
            if (audio[i] > mx) mx = audio[i];
            if (audio[i] < mn) mn = audio[i];
        }
        bMax[b] = mx;
        bMin[b] = mn;
    }
    return { bMax, bMin, baseline, target, nFrames };
}

function buildMap(baseline: Float32Array, target: Float32Array, nFrames: number, volume: number) {
    return makeLoudnessAmplitudeMap(
        {
            startFrame: 0,
            stride: 1,
            framePeriodMs: FRAME_MS,
            volume: new Float32Array(nFrames).fill(volume) as unknown as number[],
            dynTarget: target as unknown as number[],
            dynBaseline: baseline as unknown as number[],
        },
        { volume: () => null, dyn: () => null },
        () => 0,
    );
}

function envelopeAt(
    widthPx: number,
    bMax: Float32Array,
    bMin: Float32Array,
    map: ReturnType<typeof buildMap>,
): number[] {
    const scene: WaveformScene = {
        segments: [
            {
                clipId: "c",
                sourcePath: "x.wav",
                sourceSampleRate: SR,
                sourceStartSec: 0,
                sourceEndSec: DUR,
                clipStartSec: 0,
                clipLocalStartSec: 0,
                clipLocalEndSec: DUR,
                clipTotalDurationSec: DUR,
                screenRect: { x: 0, y: 0, width: widthPx, height: 100 },
                reversed: false,
                gain: 1,
                fadeInSec: 0,
                fadeOutSec: 0,
                fadeInShape: 0,
                fadeInDir: 0,
                fadeOutShape: 0,
                fadeOutDir: 0,
                alpha: 1,
                channelMode: 0,
                sourceChannels: 0,
            },
        ],
        markers: [],
    };
    const geo = buildWaveformGeometry({
        scene,
        color: "#fff",
        getPeaks: () => ({ min: bMin, max: bMax, dataStartSec: 0, dataDurationSec: DUR }),
        amplitudeMap: map,
    });
    // 顶点语义：上沿 y = 50 − mappedMax·50、下沿 y = 50 − mappedMin·50。
    const out: number[] = [];
    for (let v = 0; v < geo.vertices.length; v += 12) {
        const yT = Math.min(geo.vertices[v + 1] ?? 50, geo.vertices[v + 7] ?? 50);
        out.push((50 - yT) / 50);
    }
    return out;
}

const sliceOf = (a: number[], from: number, to: number): number[] =>
    a.slice(Math.floor(a.length * from), Math.floor(a.length * to));

test("near-zero dyn gain: no phantom peaks above the reachable level (coarse zoom)", () => {
    const { bMax, bMin, baseline, target, nFrames } = buildScenario();
    const map = buildMap(baseline, target, nFrames, 1);

    // 允许 4% 数值裕度（切片离散化 + 浮点）。
    const MARGIN = 1.04 + 1e-3;

    for (const widthPx of [600, 300, 150, 75, 40, 20]) {
        const heights = envelopeAt(widthPx, bMax, bMin, map);
        const spc = DUR / widthPx;
        for (let c = 0; c < heights.length; c += 1) {
            const centerSec = ((c + 0.5) / widthPx) * DUR;
            const loSec = Math.max(0, centerSec - spc / 2);
            const hiSec = Math.min(DUR, centerSec + spc / 2);
            // 上界 = 该列窗内 max(vol × 目标)。动态区里就是"用户画的目标电平"。
            const f0 = Math.max(0, Math.floor((loSec * 1000) / FRAME_MS));
            const f1 = Math.min(nFrames - 1, Math.ceil((hiSec * 1000) / FRAME_MS));
            let ceiling = 0;
            for (let f = f0; f <= f1; f += 1) {
                const lv = Math.max(target[f], 0) * 1; // vol = 1
                if (lv > ceiling) ceiling = lv;
            }
            if (heights[c] > ceiling * MARGIN) {
                throw new Error(
                    `cols=${widthPx} 列 ${c}: 包络 ${heights[c].toFixed(4)} ` +
                        `超过可达上限 ${ceiling.toFixed(4)}（幻峰）`,
                );
            }
        }
    }
});

test("near-zero dyn gain: unedited regions stay pixel-identical at every zoom", () => {
    const { bMax, bMin, baseline, target, nFrames } = buildScenario();
    // 未编辑基准：因子恒 1（目标=原声、音量=1）→ 显示即源包络本身。
    const linearMap = makeLoudnessAmplitudeMap(
        {
            startFrame: 0,
            stride: 1,
            framePeriodMs: FRAME_MS,
            volume: new Float32Array(nFrames).fill(1) as unknown as number[],
            dynTarget: baseline.slice() as unknown as number[],
            dynBaseline: baseline.slice() as unknown as number[],
        },
        { volume: () => null, dyn: () => null },
        () => 0,
    );
    const withDyn = buildMap(baseline, target, nFrames, 1);

    for (const widthPx of [600, 300, 75, 20]) {
        const ref = sliceOf(envelopeAt(widthPx, bMax, bMin, linearMap), 0.02, 0.32);
        const got = sliceOf(envelopeAt(widthPx, bMax, bMin, withDyn), 0.02, 0.32);
        for (let i = 0; i < ref.length; i += 1) {
            if (Math.abs(ref[i] - got[i]) > 1e-6) {
                throw new Error(
                    `cols=${widthPx} 列 ${i}: 未编辑区被改动 ${ref[i].toFixed(6)} → ` +
                        `${got[i].toFixed(6)}（幻峰钳制不得压缩真实波形）`,
                );
            }
        }
    }
});

/**
 * 「几乎为 0 的原声」+ 大目标 + **水平缩放**：波形高度必须又低又**与缩放无关**。
 *
 * ## 故障形态（用户报告）
 *
 * 原声"几乎接近 0"的段落（线性值 0.0005 量级、肉眼即 0）把动态值拉高后，水平缩小
 * 到一定程度，该段出现**随机伪影** —— 同一段素材在不同缩放下画出的高度截然不同。
 *
 * ## 根因（两件事叠加）
 *
 * 1. **判据的输入抖动**：波形按 mipmap 桶绘制，粗缩放命中的 L2 桶宽 ≈ 85ms；而
 *    无内容判据（`dynContentFade`）的输入是 5ms 帧栅格、20ms 窗的逐帧原声估计，
 *    在"接近 0"的段落逐帧抖动若干 dB。判据随原声陡变 ⇒ 桶峰（一个时刻）与判据
 *    （另一个时刻）失配 ⇒ 命中哪些列随桶的划分（= 缩放）变化。
 * 2. **过渡带落点**：淡出若跨越 [0, 下限]，对数轴上的中点落在 −66 dB 附近 ——
 *    正是"看起来是 0"的量级，于是该段只被压掉一半、仍被抬到目标附近。
 *
 * ## 判据
 *
 * 对"接近 0"的段落（−66 / −80 dBFS 抖动）：任何缩放下的显示高度都必须很低，
 * 且**各缩放之间高度差极小**（与缩放无关）；有内容的段落（下限之上）不受影响。
 */
test("near-zero dyn gain: displayed height must be low and zoom-independent", () => {
    const nFrames = Math.round((DUR * 1000) / FRAME_MS);
    const widths = [600, 300, 150, 75, 40, 20];

    const scenario = (meanDb: number, spreadDb: number) => {
        const target: number[] = new Array(nFrames).fill(0.5);
        const baseline: number[] = new Array(nFrames);
        let seed = 20240922;
        const rnd = (): number => {
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            return seed / 0x7fffffff;
        };
        for (let i = 0; i < nFrames; i += 1) {
            baseline[i] = Math.pow(10, (meanDb + spreadDb * (rnd() - 0.5)) / 20);
        }
        const total = Math.round(DUR * SR);
        const bucketCount = Math.ceil(total / DIV);
        const bMax = new Float32Array(bucketCount);
        const bMin = new Float32Array(bucketCount);
        for (let b = 0; b < bucketCount; b += 1) {
            const f0 = Math.floor((b * DIV) / SR / (FRAME_MS / 1000));
            const f1 = Math.min(nFrames - 1, Math.ceil(((b + 1) * DIV) / SR / (FRAME_MS / 1000)));
            let mx = 0;
            for (let f = Math.max(0, f0); f <= f1; f += 1) mx = Math.max(mx, baseline[f] ?? 0);
            bMax[b] = mx;
            bMin[b] = -mx;
        }
        const map = makeLoudnessAmplitudeMap(
            {
                startFrame: 0,
                stride: 1,
                framePeriodMs: FRAME_MS,
                volume: new Float32Array(nFrames).fill(1) as unknown as number[],
                dynTarget: target as unknown as number[],
                dynBaseline: baseline as unknown as number[],
            },
            { volume: () => null, dyn: () => null },
            () => 0,
        );
        return { bMax, bMin, map };
    };

    // 「几乎为 0」：−66 dBFS（线性 5e-4）与 −80 dBFS 抖动。
    for (const [label, meanDb, spreadDb] of [
        ["−66 dBFS ± 8", -66, 8],
        ["−80 dBFS ± 10", -80, 10],
    ] as Array<[string, number, number]>) {
        const { bMax, bMin, map } = scenario(meanDb, spreadDb);
        const maxima = widths.map((w) => Math.max(...envelopeAt(w, bMax, bMin, map)));
        for (let i = 0; i < widths.length; i += 1) {
            if (maxima[i] > 0.05) {
                throw new Error(
                    `${label} @ ${widths[i]}px：显示高度 ${maxima[i].toFixed(4)} 应远低于目标 0.5`,
                );
            }
        }
        // 与缩放无关：各缩放下最大高度之差必须极小（旧的随机伪影会在此露馅）。
        const lo = Math.min(...maxima);
        const hi = Math.max(...maxima);
        // 允许 ≤ 2% 高度的缩放差异（桶/切片聚合本身带来的微小起伏），
        // 远小于"随机伪影"时代（该段能达目标的 40% 以上）。
        if (hi - lo > 0.02) {
            throw new Error(
                `${label}：显示高度随缩放变化过大（${lo.toFixed(4)}…${hi.toFixed(4)}）`,
            );
        }
    }

    // 有内容（下限之上，−40 dBFS）的段落仍须照常达到目标 —— 不得误压。
    const { bMax, bMin, map } = scenario(-40, 2);
    const contentMax = Math.max(...envelopeAt(75, bMax, bMin, map));
    if (contentMax < 0.2) {
        throw new Error(`下限之上的段落被过度压制：实测 ${contentMax.toFixed(4)}`);
    }
});
