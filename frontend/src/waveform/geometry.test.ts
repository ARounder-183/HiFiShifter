import { test } from "vitest";

import { buildWaveformGeometry, parseWaveformColor } from "./geometry.ts";
import type { WaveformScene, WaveformSceneSegment } from "./sceneBuilder.ts";

test("waveform/geometry dual-band samples each channel plane independently", () => {
    // 回归：双带布局下两条带的逐像素采样必须各自读取本带的声道平面。
    // 此前误读 peaks.min/max（恒为 ch0），导致两条带画出同一个左声道 ——
    // 用户看到"双声道波形上下两条都是左声道"。
    // 用例：ch0 满幅、ch1 静音。修复后：上带（ch0）应跨满半带高度，
    // 下带（ch1）应收敛在自身中心线上。
    const segment = {
        clipId: "clip",
        sourcePath: "/stereo.wav",
        sourceSampleRate: 4,
        sourceStartSec: 0,
        sourceEndSec: 1,
        clipLocalStartSec: 0,
        clipLocalEndSec: 1,
        clipTotalDurationSec: 1,
        screenRect: { x: 0, y: 0, width: 4, height: 100 },
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
        sourceChannels: 2,
    };
    const scene: WaveformScene = { segments: [segment], markers: [] };

    const result = buildWaveformGeometry({
        scene,
        color: "rgba(255,255,255,1)",
        getPeaks: () => ({
            min: new Float32Array([-1, -1, -1, -1]),
            max: new Float32Array([1, 1, 1, 1]),
            dataStartSec: 0,
            dataDurationSec: 1,
            channels: 2,
            ch1Min: new Float32Array([0, 0, 0, 0]),
            ch1Max: new Float32Array([0, 0, 0, 0]),
        }),
    });

    // 顶点布局：先 band0 的全部列（每列 2 顶点），再 band1 的全部列。
    // 4 列 × 2 带 × 2 顶点 = 16 顶点；band0 第 0 列 = 顶点 0/1，band1 第 0 列 = 顶点 8/9。
    const vertices = result.vertices;
    if (vertices.length < 16 * 6) {
        throw new Error(`expected 16 vertices (96 floats), got ${vertices.length / 6}`);
    }
    const yAt = (vertexIndex: number): number => vertices[vertexIndex * 6 + 1];
    const upperTop = Math.min(yAt(0), yAt(1));
    const upperBottom = Math.max(yAt(0), yAt(1));
    if (Math.abs(upperTop - 0) > 1 || Math.abs(upperBottom - 50) > 1) {
        throw new Error(`upper band (ch0) should span y 0..50, got ${upperTop}..${upperBottom}`);
    }
    // 下带（ch1，中心 y=75、半高 25）：静音应收敛在 y≈75。
    const lowerTop = Math.min(yAt(8), yAt(9));
    const lowerBottom = Math.max(yAt(8), yAt(9));
    if (Math.abs(lowerTop - 75) > 1 || Math.abs(lowerBottom - 75) > 1) {
        throw new Error(
            `lower band (ch1, silent) should collapse to y≈75, got ${lowerTop}..${lowerBottom}`,
        );
    }
});

/**
 * ★ 近零原声幻峰回归：列包络**不得超出逐样本真值**。
 *
 * 【故障形态】把"原始动态值几乎接近 0"处的动态拉大后缩小视图，波形在该处
 * 出现随机伪影（幻峰 / 空洞，随缩放位置改变）。
 *
 * 【根因】几何层原先把"片内最大峰值"（按**窗口**聚合）乘上"片**中心**时刻的
 * 增益"（**点采样**）。两个口径不同，当增益在片内剧烈变化时（近零原声下
 * `目标/max(原声,下限)` 逐帧可跨数个数量级）乘积会系统性超出真实包络，且
 * 超出量随片宽增长 —— 一列覆盖的桶越多（缩得越小）幻峰越高。
 *
 * 【判据】以逐样本真值 `max|sample| × gain(该样本时刻)` 为基准：几何输出的
 * 每一列都不得显著超过它所覆盖区间内的真值最大值。（面积/形状可以不同，
 * 但"可听高度"不可能超过该区间内真实出现过的最大可听幅度。）
 */
test("near-zero dyn gain: envelope never exceeds per-sample ground truth", async () => {
    const SR = 48000;
    const FRAME_MS = 5;
    const DIV = 16; // 桶粒度（L0）；窗口内桶数 ≫ 16 片上限，使上限真正生效
    const DUR = 12;
    const T0 = 3.5;
    const total = Math.round(DUR * SR);

    // 近零噪声尾音：每 5ms 帧一个随机幅度（10^-3.6 … 10^-2.7），帧内白噪。
    // 帧间幅度剧烈波动 ⇒ 20ms 峰（基线）与粗略桶峰不成比例 —— 触发失配。
    let seed = 987654321;
    const rnd = () => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
    };
    const audio = new Float32Array(total);
    const frameSamples = Math.round((FRAME_MS / 1000) * SR);
    let amp = 0.25;
    for (let i = 0; i < total; i += 1) {
        const t = i / SR;
        if (t < T0) {
            audio[i] = 0.25 * Math.sin(2 * Math.PI * 220 * t);
        } else {
            const off = i - Math.round(T0 * SR);
            if (off % frameSamples === 0) amp = Math.pow(10, -3.6 + 0.9 * rnd());
            audio[i] = amp * (rnd() * 2 - 1);
        }
    }

    // 基线：20ms 窗峰值、帧中心锚定（与后端 compute_frame_levels 同口径）。
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

    const FLOOR = 0.001;
    const MAX_GAIN = 1000;
    const gainAt = (sec: number): number => {
        const f = (sec * 1000) / FRAME_MS;
        const i0 = Math.max(0, Math.min(nFrames - 1, Math.floor(f)));
        const i1 = Math.min(nFrames - 1, i0 + 1);
        const fr = Math.min(1, Math.max(0, f - i0));
        const b = Math.max(baseline[i0] + (baseline[i1] - baseline[i0]) * fr, FLOOR);
        return Math.min(1 / b, MAX_GAIN);
    };

    // 幅度映射：目标 1.0 的 dyn 增益（与 paramRanges::computeDynGain 同构）。
    const map = ((value: number, gain: number, timeSec: number | null) =>
        timeSec === null ? value * gain : value * gain * gainAt(timeSec)) as ((
        v: number,
        g: number,
        t: number | null,
    ) => number) & { factorAt(t: number | null): number | null };
    map.factorAt = (t) => (t === null ? 1 : gainAt(t));

    // 桶峰值（div 粒度），与显示管线一致。
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

    let observableCount = 0;
    let totalObservableSlots = 0;
    for (const widthPx of [600, 300, 150, 75, 40, 20]) {
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

        // 逐列取出映射后的上下包络。顶点语义：上顶点 y = centerY − mappedMax·halfHeight，
        // 下顶点 y = centerY − mappedMin·halfHeight（命中 50 / 半高 50）⇒
        // mappedMax = (50 − yTop)/50、mappedMin = (50 − yBottom)/50。
        // 【必须分别比较】把两者合成"高度"会掩盖单侧的幻峰（上侧越界与下侧
        // 收缩可以互相抵消），只有逐侧比较才钉得住。
        const cols: { max: number; min: number }[] = [];
        for (let v = 0; v < geo.vertices.length; v += 12) {
            const yTop = Math.min(geo.vertices[v + 1] ?? 50, geo.vertices[v + 7] ?? 50);
            const yBottom = Math.max(geo.vertices[v + 1] ?? 50, geo.vertices[v + 7] ?? 50);
            cols.push({ max: (50 - yTop) / 50, min: (50 - yBottom) / 50 });
        }

        // 参考口径 = **逐桶配对**，正负两侧各自求：
        //   refMax = max over buckets( bMax[b] × gain(bucketTime) )
        //   refMin = min over buckets( bMin[b] × gain(bucketTime) )
        //
        // 这是渲染语义要求的正确值：每个峰值桶只有乘上"它自己时刻"的增益才是
        // 它真实的可听高度。几何输出恒 ≤ 该参考（切片取片内 argmax 并配以其
        // 自身时刻增益，是参考项的一个子集；窗口 ≤ 16 桶时精确相等）——
        // 少算是安全的（略低的包络），**多算即为幻峰**。
        //
        // 旧实现（片内 max × 片**中心**增益）会超过该参考，且越界量随缩放
        // 增大：实测 20 列时达 1.88×（15/20 列越界）= 用户报告的随机伪影。
        //
        // 【参考窗口必须比几何窗口宽 1 桶】几何的 indexStart/indexEnd 由
        // 浮点乘除取整得出，窗口边界处 ±1 桶的差别取决于运算顺序（本测试
        // 重算时曾因此比几何少含一个边界桶，把几何的合法值误判为幻峰）。
        // 向两侧各放宽一桶使参考窗口**包含**几何窗口，判据才严格成立。
        const spc = DUR / widthPx;
        for (let c = 0; c < cols.length; c += 1) {
            const centerSec = ((c + 0.5) / widthPx) * DUR;
            const loSec = Math.max(0, centerSec - spc / 2);
            const hiSec = Math.min(DUR, centerSec + spc / 2);
            const idxStart = Math.max(0, Math.floor((loSec / DUR) * bucketCount));
            const idxEnd = Math.min(
                bucketCount - 1,
                Math.max(idxStart, Math.ceil((hiSec / DUR) * bucketCount) - 1),
            );
            const refStart = Math.max(0, idxStart - 1);
            const refEnd = Math.min(bucketCount - 1, idxEnd + 1);
            let refMax = 0;
            let refMin = 0;
            for (let b = refStart; b <= refEnd; b += 1) {
                const g = gainAt(((b + 0.5) * DIV) / SR);
                const vMax = bMax[b] * g;
                const vMin = bMin[b] * g;
                if (vMax > refMax) refMax = vMax;
                if (vMin < refMin) refMin = vMin;
            }
            // 顶点语义（已由 probe 确认）：顶点0 上沿 = centerY − mappedMax·half、
            // 顶点1 下沿 = centerY − mappedMin·half。故 cols[].max = mappedMax、
            // cols[].min = mappedMin，参考按同口径钳到矩形。
            const wantMax = Math.min(Math.max(refMax, 0), 1);
            const wantMin = Math.max(Math.min(refMin, 0), -1);
            // 统计"非饱和"侧：饱和侧被矩形钳到 ±1，越界不可观测。
            if (wantMax < 1 - 1e-9) observableCount += 1;
            if (wantMin > -1 + 1e-9) observableCount += 1;
            totalObservableSlots += 2;
            // 允许 2% + 1e-3 数值裕度（几何不得**超过**参考）。
            if (cols[c].max > wantMax * 1.02 + 1e-3) {
                throw new Error(
                    `cols=${widthPx} 列 ${c}: 上包络 ${cols[c].max.toFixed(4)} ` +
                        `超过逐桶参考 ${wantMax.toFixed(4)}（幻峰）`,
                );
            }
            if (cols[c].min < wantMin * 1.02 - 1e-3) {
                throw new Error(
                    `cols=${widthPx} 列 ${c}: 下包络 ${cols[c].min.toFixed(4)} ` +
                        `低于逐桶参考 ${wantMin.toFixed(4)}（幻峰）`,
                );
            }
        }
    }
    // 保证本用例真的覆盖了可观测区间（否则断言会退化成"恒真"）。
    if (observableCount < totalObservableSlots * 0.1) {
        throw new Error(
            `可观测（非饱和）列过少：${observableCount}/${totalObservableSlots}，用例失去判别力`,
        );
    }
});

test("waveform/geometry.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    assertEqual(
        parseWaveformColor("rgba(246,250,255,0.92)"),
        [246 / 255, 250 / 255, 1, 0.92],
        "rgba colors become normalized GPU channels",
    );
    assertEqual(
        parseWaveformColor("#804020"),
        [128 / 255, 64 / 255, 32 / 255, 1],
        "hex colors become normalized GPU channels",
    );

    const scene: WaveformScene = {
        segments: [
            {
                clipId: "clip",
                sourcePath: "/tone.wav",
                sourceSampleRate: 4,
                sourceStartSec: 0,
                sourceEndSec: 1,
                clipLocalStartSec: 0,
                clipLocalEndSec: 1,
                clipTotalDurationSec: 1,
                screenRect: { x: 0, y: 0, width: 4, height: 100 },
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

    const result = buildWaveformGeometry({
        scene,
        color: "rgba(255,255,255,0.5)",
        getPeaks: () => ({
            min: new Float32Array([-1, -0.5, -0.25, 0]),
            max: new Float32Array([1, 0.5, 0.25, 0]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
    });

    assertEqual(result.complete, true, "geometry is complete when every segment has peaks");
    assertEqual(result.lineCount, 4, "one GPU line is emitted per visible pixel column");
    assertEqual(
        Array.from(result.vertices.slice(0, 12)),
        [0.5, 0, 1, 1, 1, 0.5, 0.5, 100, 1, 1, 1, 0.5],
        "first peak maps to the full waveform height with normalized color",
    );

    const missing = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => null,
    });
    assertEqual(missing.complete, false, "missing peaks defer frame presentation");
    assertEqual(missing.vertices.length, 0, "missing data renders no replacement geometry");

    const partialScene: WaveformScene = {
        segments: [
            scene.segments[0],
            {
                ...scene.segments[0],
                clipId: "missing",
                sourcePath: "/missing.wav",
                screenRect: { x: 4, y: 0, width: 4, height: 100 },
            },
        ],
        markers: [],
    };

    const partial = buildWaveformGeometry({
        scene: partialScene,
        color: "#ffffff",
        getPeaks: (sourcePath: string) =>
            sourcePath === "/tone.wav"
                ? {
                      min: new Float32Array([-1, 1]),
                      max: new Float32Array([1, -1]),
                      dataStartSec: 0,
                      dataDurationSec: 1,
                  }
                : null,
    });
    assertEqual(partial.complete, false, "partial availability is reported as incomplete");
    assertEqual(partial.lineCount, 4, "available segments still render while missing data loads");

    const dimmed = buildWaveformGeometry({
        scene: { segments: [{ ...scene.segments[0], inactive: true }], markers: [] },
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-1]),
            max: new Float32Array([1]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
    });
    assertEqual(
        Array.from(dimmed.vertices.slice(2, 6)),
        [Math.fround(0.42), Math.fround(0.42), Math.fround(0.42), Math.fround(0.78)],
        "inactive take lanes darken rgb and color alpha",
    );

    const dimmedMarker = buildWaveformGeometry({
        scene: {
            segments: [],
            markers: [
                {
                    clipId: "clip",
                    timelineSec: 1,
                    xPx: 10,
                    yPx: 20,
                    heightPx: 100,
                    kind: "loop",
                    inactive: true,
                },
            ],
        },
        color: "#ffffff",
        getPeaks: () => null,
    });
    assertEqual(
        dimmedMarker.vertices.length,
        84,
        "inactive marker still emits its scanline-filled triangle",
    );
    assertEqual(
        Array.from(dimmedMarker.vertices.slice(2, 6)),
        [Math.fround(0.42), Math.fround(0.42), Math.fround(0.42), Math.fround(0.78)],
        "inactive markers darken too",
    );

    // 音量增益 > 1 时包络按 gain 放大，必须被钳制在波形矩形内（削顶显示），
    // 不能溢出 clip 上下边界。
    const boosted = buildWaveformGeometry({
        scene: { segments: [{ ...scene.segments[0], gain: 3 }], markers: [] },
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-1]),
            max: new Float32Array([1]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
    });
    // 高度 100，中心 50，±1 峰值 × gain 3 → 名义 ±150，钳制后恰好 0 / 100。
    assertEqual(
        [boosted.vertices[1], boosted.vertices[7]],
        [0, 100],
        "gain-boosted envelope clamps to the waveform rect (flat-top display)",
    );

    // gain = 1 时钳制不影响正常包络（回归保护）。峰值取 ±0.5 保证精确：
    // 中心 50 ± 0.5×50 → 25 / 75。
    const unity = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-0.5]),
            max: new Float32Array([0.5]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
    });
    assertEqual(
        [unity.vertices[1], unity.vertices[7]],
        [25, 75],
        "unity gain envelope is untouched by the clamp",
    );
});

/**
 * 逐时间幅度映射（动态面板）。
 *
 * 【为什么必须单测这一层】"编辑动态时波形不重绘"的根因是映射拿不到时间：
 * 它只能施加一个全局系数，画多少曲线波形都不变。这里钉住"同一峰值在不同
 * 时间可以映射出不同高度"，以及映射时刻的**源锚定**语义：每次调用拿到的
 * timeSec 是切片中心**峰值桶**的时间轴绝对时间 —— 峰值与自身时刻的增益配对。
 */
test("amplitude map receives slice-anchored timeline time", async () => {
    const scene: WaveformScene = {
        segments: [
            {
                clipId: "clip",
                sourcePath: "/tone.wav",
                sourceSampleRate: 4,
                sourceStartSec: 0,
                sourceEndSec: 1,
                // Clip 从时间轴 10s 开始，本段覆盖它的第 0..1s。
                clipStartSec: 10,
                clipLocalStartSec: 0,
                clipLocalEndSec: 1,
                clipTotalDurationSec: 1,
                screenRect: { x: 0, y: 0, width: 4, height: 100 },
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

    const seen: number[] = [];
    const result = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-1, -1, -1, -1]),
            max: new Float32Array([1, 1, 1, 1]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
        amplitudeMap: (value, gain, timeSec) => {
            if (timeSec !== null) seen.push(timeSec);
            return value * gain;
        },
    });

    // dpr=1、宽 4 列、4 个峰值桶：列窗口（±半列 = ±0.125s）恰好各覆盖一个桶
    // → 每列 1 个切片、min/max 各一次 → 共 4×2 = 8 次。
    if (seen.length !== 8) {
        throw new Error(`expected 8 samples, received ${seen.length}`);
    }
    // 切片中心 = 峰值桶中心（源锚定）：桶 k 中心 = (k+0.5)/4 秒 + Clip 起点 10s。
    // 注意这不是列中心的 t（列中心 0.125/0.375/... 在 1 桶/列的巧合下与桶中心
    // 重合；dpr=1.25 的用例里两者分离，见下一条测试）。
    const expectedSlicesPerColumn = [[10.125], [10.375], [10.625], [10.875]];
    let cursor = 0;
    for (let column = 0; column < 4; column += 1) {
        const expected = expectedSlicesPerColumn[column];
        for (let slice = 0; slice < expected.length; slice += 1) {
            const minTime = seen[cursor];
            const maxTime = seen[cursor + 1];
            cursor += 2;
            if (minTime === undefined || maxTime === undefined) {
                throw new Error(`column ${column} slice ${slice}: missing samples`);
            }
            // min 与 max 必须用同一个时间（同一切片内增益恒定，包络不失真）。
            if (minTime !== maxTime) {
                throw new Error(
                    `column ${column} slice ${slice}: min/max disagree on time (${minTime} vs ${maxTime})`,
                );
            }
            if (Math.abs(minTime - expected[slice]) > 1e-9) {
                throw new Error(
                    `column ${column} slice ${slice}: expected t=${expected[slice]}, received ${minTime}`,
                );
            }
        }
    }
    if (result.lineCount !== 4) {
        throw new Error(`expected 4 lines, received ${result.lineCount}`);
    }
});

/**
 * 缩放一致性回归（用户报告：音量/动态缩放后的波形"峰值高度在水平缩放过程中
 * 乱跳"、不同缩放等级画出截然不同的波形）。
 *
 * 根因：旧实现整列共用**列中心**时刻的一次增益 —— 峰值被乘上"别的时刻"的
 * 音量/动态值；列中心随缩放在时间轴上扫动，峰值高度随之跳变，粗缩放还会
 * 画出幻峰。修复后每列按峰值索引切片、峰值与**自身时刻**的增益配对：
 * 同一峰值在任何缩放等级下高度一致，粗列包络 = 细列包络的逐段 max。
 *
 * 本用例构造一个"音量凹陷 + 大峰值落在凹陷里"的场景：旧实现会在粗缩放下
 * 画出一根满高的幻峰（列中心在凹陷外），修复后粗 / 细两种缩放的最高峰
 * 都是 0.1 —— 完全一致。
 */
test("volume-scaled envelope is zoom-stable (no phantom peaks at coarse zoom)", async () => {
    const makeScene = (widthPx: number): WaveformScene => ({
        segments: [
            {
                clipId: "clip",
                sourcePath: "/tone.wav",
                sourceSampleRate: 4,
                sourceStartSec: 0,
                sourceEndSec: 1,
                clipStartSec: 0,
                clipLocalStartSec: 0,
                clipLocalEndSec: 1,
                clipTotalDurationSec: 1,
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
    });
    // 桶 1（源 0.25..0.5s，中心 t=0.375）是满幅大峰值；其余桶 0.1。
    const peaks = {
        min: new Float32Array([-0.1, -1, -0.1, -0.1]),
        max: new Float32Array([0.1, 1, 0.1, 0.1]),
        dataStartSec: 0,
        dataDurationSec: 1,
    };
    // 音量自动化：t < 0.25 为 1，其后凹陷到 0.1 —— 大峰值（t=0.375）落在
    // 凹陷内，可听高度 = 1 × 0.1 = 0.1。
    const volumeDipMap = (value: number, gain: number, timeSec: number | null) =>
        value * gain * (timeSec !== null && timeSec < 0.25 ? 1 : 0.1);

    const build = (widthPx: number) =>
        buildWaveformGeometry({
            scene: makeScene(widthPx),
            color: "#ffffff",
            getPeaks: () => peaks,
            amplitudeMap: volumeDipMap,
        });

    // 高度 100、中心 50、半高 50：可听峰值 0.1 → 包络顶 = 50 − 0.1×50 = 45。
    const tallestTop = (geometry: { vertices: Float32Array }): number => {
        let minY = Number.POSITIVE_INFINITY;
        for (let v = 0; v < geometry.vertices.length; v += 12) {
            const y = geometry.vertices[v + 1] ?? 0;
            if (y < minY) minY = y;
        }
        return minY;
    };

    const coarse = build(4); // 4 列粗缩放：列 0/1 的窗口横跨大峰值桶
    const fine = build(40); // 40 列细缩放：大峰值独占若干列
    const coarseTop = tallestTop(coarse);
    const fineTop = tallestTop(fine);
    if (Math.abs(coarseTop - 45) > 1e-6) {
        throw new Error(
            `coarse zoom draws a phantom peak: expected tallest envelope top 45 (0.1), got ${coarseTop}`,
        );
    }
    if (Math.abs(fineTop - 45) > 1e-6) {
        throw new Error(
            `fine zoom envelope changed: expected tallest envelope top 45 (0.1), got ${fineTop}`,
        );
    }
});

test("amplitude map can vary per pixel column (dynamic gain)", async () => {
    const scene: WaveformScene = {
        segments: [
            {
                clipId: "clip",
                sourcePath: "/tone.wav",
                sourceSampleRate: 4,
                sourceStartSec: 0,
                sourceEndSec: 1,
                clipStartSec: 0,
                clipLocalStartSec: 0,
                clipLocalEndSec: 1,
                clipTotalDurationSec: 1,
                screenRect: { x: 0, y: 0, width: 4, height: 100 },
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

    // 模拟动态增益：前半段 ×2，后半段 ×0.5。
    const result = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-0.5, -0.5, -0.5, -0.5]),
            max: new Float32Array([0.5, 0.5, 0.5, 0.5]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
        amplitudeMap: (value, gain, timeSec) =>
            value * gain * (timeSec !== null && timeSec < 0.5 ? 2 : 0.5),
    });

    // 中心 50、半高 50：×2 → 峰值 0.5×2=1 → top 0；×0.5 → 0.25 → top 37.5。
    const topOf = (col: number) => result.vertices[col * 12 + 1];
    if (topOf(0) !== 0) {
        throw new Error(`column 0 should be boosted to the top, received ${topOf(0)}`);
    }
    if (Math.abs(topOf(3) - 37.5) > 1e-6) {
        throw new Error(`column 3 should be attenuated, received ${topOf(3)}`);
    }
    // 逐列不同 —— 这正是"波形按动态值重绘"的可观测判据。
    if (topOf(0) === topOf(3)) {
        throw new Error("columns must differ when the dynamic gain varies over time");
    }
});

/**
 * 时域因子视图（`WaveformAmplitudeFactors`）的等价性与调用次数契约。
 *
 * 【为什么必须钉住】几何层在映射声明了 `factorAt` 时改走"每切片求值一次因子
 * + 两次乘法"，不再逐值调用 `amplitudeMap`。这条快路径是拖动音量/动态时
 * 不卡顿的关键（逐值调用要在一次重建里做数万次曲线取样与 live 覆盖解析），
 * 但它必须与逐值路径**逐像素等价** —— 否则同一份数据会因为"映射有没有声明
 * 因子"而画出不同的波形。
 */
test("amplitude factor view is pixel-identical to per-value calls", async () => {
    const scene: WaveformScene = {
        segments: [
            {
                clipId: "clip",
                sourcePath: "/tone.wav",
                sourceSampleRate: 4,
                sourceStartSec: 0,
                sourceEndSec: 1,
                clipStartSec: 5,
                clipLocalStartSec: 0,
                clipLocalEndSec: 1,
                clipTotalDurationSec: 1,
                screenRect: { x: 0, y: 0, width: 4, height: 100 },
                reversed: false,
                gain: 0.8,
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
    const peaks = {
        min: new Float32Array([-0.4, -0.9, -0.2, -0.6]),
        max: new Float32Array([0.4, 0.9, 0.2, 0.6]),
        dataStartSec: 0,
        dataDurationSec: 1,
    };
    /** 与 `timeSec` 相关的纯乘性因子（复刻响度映射的形态）。 */
    const factorOf = (timeSec: number | null): number =>
        timeSec === null ? 1 : 0.25 + 1.5 * Math.abs(Math.sin(timeSec * 3.1));

    // ① 逐值路径（不声明 factorAt）。
    const perValue = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => peaks,
        amplitudeMap: (value, gain, timeSec) => value * gain * factorOf(timeSec),
    });
    // ② 因子路径（声明 factorAt）。
    let factorCalls = 0;
    let perValueCalls = 0;
    const amplitudeFactorMap = ((value: number, gain: number, timeSec: number | null) => {
        perValueCalls += 1;
        return value * gain * factorOf(timeSec);
    }) as ((value: number, gain: number, timeSec: number | null) => number) & {
        factorAt(t: number | null): number | null;
    };
    amplitudeFactorMap.factorAt = (timeSec) => {
        factorCalls += 1;
        return factorOf(timeSec);
    };
    const viaFactor = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => peaks,
        amplitudeMap: amplitudeFactorMap,
    });

    // 【等价性】顶点逐值相同（含颜色 / alpha 分量）。
    const expected = Array.from(perValue.vertices);
    const actual = Array.from(viaFactor.vertices);
    if (JSON.stringify(expected) !== JSON.stringify(actual)) {
        throw new Error(
            `factor path must be pixel-identical: expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`,
        );
    }

    // 【调用次数】因子路径不得落到逐值调用；每切片恰好一次因子求值。
    if (perValueCalls !== 0) {
        throw new Error(`factor path must not fall back to per-value calls, got ${perValueCalls}`);
    }
    // 4 列 × 1 桶/列 = 4 个切片 → 4 次因子求值（逐值路径会是 8 次）。
    if (factorCalls !== 4) {
        throw new Error(`expected 4 factor evaluations (one per slice), received ${factorCalls}`);
    }
});

/**
 * `factorAt` 返回 null 时必须回落到逐值调用（非乘性 / 数据异常的时刻）。
 */
test("amplitude factor view falls back to per-value calls when factor is null", async () => {
    const scene: WaveformScene = {
        segments: [
            {
                clipId: "clip",
                sourcePath: "/tone.wav",
                sourceSampleRate: 4,
                sourceStartSec: 0,
                sourceEndSec: 1,
                clipLocalStartSec: 0,
                clipLocalEndSec: 1,
                clipTotalDurationSec: 1,
                screenRect: { x: 0, y: 0, width: 4, height: 100 },
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
    let perValueCalls = 0;
    const map = ((value: number, gain: number) => {
        perValueCalls += 1;
        return value * gain * 0.5;
    }) as unknown as ((value: number, gain: number, timeSec: number | null) => number) & {
        factorAt(t: number | null): number | null;
    };
    map.factorAt = () => null;

    const result = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-0.5, -0.5, -0.5, -0.5]),
            max: new Float32Array([0.5, 0.5, 0.5, 0.5]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
        amplitudeMap: map,
    });

    // 4 列 × 2（min/max）= 8 次逐值调用（回落后仍按切片契约调用）。
    if (perValueCalls !== 8) {
        throw new Error(`expected 8 per-value fallback calls, received ${perValueCalls}`);
    }
    // 0.5 × 0.5 = 0.25 → 中心 50、半高 50 → top = 50 − 0.25×50 = 37.5。
    const topOf = (col: number) => result.vertices[col * 12 + 1];
    if (Math.abs((topOf(0) ?? 0) - 37.5) > 1e-6) {
        throw new Error(`fallback result must still be correct, received ${topOf(0)}`);
    }
});

/**
 * 设备像素网格枚举（用户报告的"缩放率 > 1 时线宽 1~2px 抖动"的几何侧回归）。
 *
 * 旧实现按 CSS 像素枚举包络列、列宽恒 1 CSS px：dpr=1.25/1.5 这类非整数比下
 * 列的**设备**覆盖在 1~2 物理像素之间随位置跳变。修复后列枚举走设备像素网格
 * （每列恰好 `round(dpr)` 物理像素、列边落在设备像素边界），任何 dpr 下线宽
 * 恒定。渲染端展开宽度由 surfaceRenderer.test.ts 的同族用例钉住。
 */
test("device-pixel column enumeration (dpr grid alignment)", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const baseSegment: WaveformSceneSegment = {
        clipId: "clip",
        sourcePath: "/tone.wav",
        sourceSampleRate: 4,
        sourceStartSec: 0,
        sourceEndSec: 1,
        clipLocalStartSec: 0,
        clipLocalEndSec: 1,
        clipTotalDurationSec: 1,
        screenRect: { x: 0, y: 0, width: 4, height: 100 },
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
    };
    const scene: WaveformScene = { segments: [baseSegment], markers: [] };
    const build = (dpr?: number) =>
        buildWaveformGeometry({
            scene,
            color: "#ffffff",
            getPeaks: () => ({
                min: new Float32Array([-1, -1, -1, -1]),
                max: new Float32Array([1, 1, 1, 1]),
                dataStartSec: 0,
                dataDurationSec: 1,
            }),
            ...(dpr === undefined ? {} : { dpr }),
        });

    // dpr=1.25（Windows 125%）：W=1 → 每设备列一包络样本。4 CSS px 宽 →
    // ceil(4·1.25/1) = 5 列；列中心 CSS x = (k+0.5)·1/1.25，对应设备中心 k+0.5。
    const dpr125 = build(1.25);
    assertEqual(dpr125.lineCount, 5, "dpr=1.25 enumerates one column per device pixel");
    const centers125 = Array.from({ length: 5 }, (_, k) => dpr125.vertices[k * 12]);
    assertEqual(
        centers125.map((x) => Math.round(x * 1.25 * 2) / 2),
        [0.5, 1.5, 2.5, 3.5, 4.5],
        "dpr=1.25 column centers land on device pixel centers",
    );

    // dpr=1.5：W = round(1.5) = 2 → 4·1.5/2 = 3 列；中心 x = (k+0.5)·2/1.5 CSS。
    // 顶点经 Float32Array 存储会舍入到 f32 —— 与 Math.fround(期望) 逐位相等。
    const dpr15 = build(1.5);
    assertEqual(dpr15.lineCount, 3, "dpr=1.5 uses W=2 device px columns");
    assertEqual(
        Array.from({ length: 3 }, (_, k) => dpr15.vertices[k * 12]),
        [0.5, 1.5, 2.5].map((center) => Math.fround((center * 2) / 1.5)),
        "dpr=1.5 column centers sit mid-column in device space",
    );

    // dpr=2：W = 2 → 4·2/2 = 4 列，中心 x = k+0.5 CSS —— 与 dpr=1 的旧实现
    // 逐值一致（契约：整数 dpr 的画面不因本次修复而变）。
    const dpr2 = build(2);
    const dpr1 = build(1);
    assertEqual(dpr2.lineCount, 4, "dpr=2 keeps the legacy 1-CSS-px column pitch");
    assertEqual(
        Array.from({ length: 4 }, (_, k) => dpr2.vertices[k * 12]),
        Array.from({ length: 4 }, (_, k) => dpr1.vertices[k * 12]),
        "dpr=2 column centers equal the legacy dpr=1 centers",
    );

    // 缺省 dpr=1：与显式 dpr=1 一致（既有调用方 / 测试行为不变）。
    const defaulted = build();
    assertEqual(defaulted.lineCount, dpr1.lineCount, "omitted dpr defaults to 1 (legacy behavior)");
});

test("device-grid columns widen the sample window per column (dpr-aware)", async () => {
    // 每列取峰窗口 = 时长 × W / (宽·dpr)：dpr=1.25、W=1、宽 4、时长 1 → 0.2s。
    // 用一个阶梯峰值序列检验窗口按列推进（旧 CSS 列口径是 0.25s/列）。
    const seen: Array<[number, number]> = [];
    buildWaveformGeometry({
        scene: {
            segments: [
                {
                    clipId: "clip",
                    sourcePath: "/tone.wav",
                    sourceSampleRate: 4,
                    sourceStartSec: 0,
                    sourceEndSec: 1,
                    clipLocalStartSec: 0,
                    clipLocalEndSec: 1,
                    clipTotalDurationSec: 1,
                    screenRect: { x: 0, y: 0, width: 4, height: 100 },
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
        },
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-1, -1, -1, -1]),
            max: new Float32Array([1, 1, 1, 1]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
        dpr: 1.25,
        amplitudeMap: (value, _gain, timeSec) => {
            if (timeSec !== null) seen.push([timeSec, value]);
            return value;
        },
    });
    // dpr=1.25、W=1 → 5 个设备列。列窗口（±半列 = ±0.1s）与 4 桶的覆盖关系：
    // 列 0 → 桶 0；列 1 → 桶 0..1；列 2 → 桶 1..2；列 3 → 桶 2..3；列 4 → 桶 3
    // → 共 1+2+2+2+1 = 8 个切片，min/max 各一次 = 16 次调用。
    if (seen.length !== 16) {
        throw new Error(`expected 16 samples, received ${seen.length}`);
    }
    // 每切片时刻 = 其覆盖桶的中心（源锚定）：桶 k 中心 = (k+0.5)/4 s。
    // 注意列中心 t（0.1/0.3/0.5/...）与此**不同** —— 旧实现按列中心采样增益，
    // 峰值高度会随缩放扫动；现在锚定到峰值桶自身。
    const expectedSliceTimes = [0.125, 0.125, 0.375, 0.375, 0.625, 0.625, 0.875, 0.875];
    let cursor = 0;
    for (let slice = 0; slice < expectedSliceTimes.length; slice += 1) {
        const minTime = seen[cursor]?.[0];
        const maxTime = seen[cursor + 1]?.[0];
        cursor += 2;
        if (minTime === undefined || maxTime === undefined) {
            throw new Error(`slice ${slice}: missing samples`);
        }
        if (minTime !== maxTime) {
            throw new Error(`slice ${slice}: min/max disagree on time (${minTime} vs ${maxTime})`);
        }
        if (Math.abs(minTime - expectedSliceTimes[slice]) > 1e-9) {
            throw new Error(
                `slice ${slice}: expected t=${expectedSliceTimes[slice]}, received ${minTime}`,
            );
        }
    }
});
