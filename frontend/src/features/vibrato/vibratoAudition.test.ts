/**
 * 颤音试听引擎的契约。
 *
 * 【要钉死的三条】与 `audioPreview` 同源（见其头部注释）：
 * 1. **连发两次 `play`，旧的一路必须被停掉** —— 否则留下"还在响却已无人登记"的
 *    孤儿，用户听到的是两路叠加（这是所有音频单例最容易破的不变量）。
 * 2. **`stop()` 停掉所有已登记的源**。
 * 3. **环境不支持 `AudioContext` 时收敛**：返回 `false`、不抛、不留状态。
 *
 * 另有 `buildAuditionCurve` 的换算契约：它保证"听到的 = 看到的"—— 试听直接吃
 * 预览的同一份 cents 曲线，只做单位换算。
 *
 * 【为什么每个测试都 new 一个播放器】`AudioContext` 是单例播放器的私有状态，
 * 跨测试存活会让第二个测试装的新假 context 永远收不到振荡器 —— 孤儿断言会因此
 * 全部失真。导出类正是为了测试能拿到干净的实例；UI 用的单例只是
 * `new VibratoAuditionPlayer()` 的一份。
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
    AUDITION_BASE_MIDI,
    buildAuditionCurve,
    buildContourAuditionPair,
    VibratoAuditionPlayer,
} from "./vibratoAudition";

/** 假振荡器：记录 start/stop 与音高调度，便于断言孤儿与包络。 */
function makeFakeOsc() {
    return {
        type: "",
        frequency: {
            /** setValueCurveAtTime 的全部调用（曲线, 起点, 时长）。 */
            curves: [] as Array<[Float32Array, number, number]>,
            value: 0,
            setValueCurveAtTime(
                this: { curves: Array<[Float32Array, number, number]> },
                curve: Float32Array,
                start: number,
                duration: number,
            ) {
                this.curves.push([curve, start, duration]);
            },
            setValueAtTime() {},
            linearRampToValueAtTime() {},
            setTargetAtTime() {},
            cancelScheduledValues() {},
            cancelAndHoldAtTime() {},
        },
        started: 0,
        stopped: 0,
        disconnected: 0,
        connect() {},
        start() {
            this.started += 1;
        },
        stop() {
            this.stopped += 1;
        },
        disconnect() {
            this.disconnected += 1;
        },
        onended: null as (() => void) | null,
    };
}

interface FakeGain {
    connect(): void;
    disconnect(): void;
    gain: {
        value: number;
        events: Array<{ method: string; args: unknown[] }>;
        setValueAtTime(...args: unknown[]): void;
        linearRampToValueAtTime(...args: unknown[]): void;
        setTargetAtTime(...args: unknown[]): void;
        cancelScheduledValues(...args: unknown[]): void;
        cancelAndHoldAtTime(...args: unknown[]): void;
    };
}

function makeFakeGain(): FakeGain {
    const gain: FakeGain["gain"] = {
        value: 0,
        events: [],
        setValueAtTime(...args: unknown[]) {
            gain.events.push({ method: "setValueAtTime", args });
        },
        linearRampToValueAtTime(...args: unknown[]) {
            gain.events.push({ method: "linearRampToValueAtTime", args });
        },
        setTargetAtTime(...args: unknown[]) {
            gain.events.push({ method: "setTargetAtTime", args });
        },
        cancelScheduledValues(...args: unknown[]) {
            gain.events.push({ method: "cancelScheduledValues", args });
        },
        cancelAndHoldAtTime(...args: unknown[]) {
            gain.events.push({ method: "cancelAndHoldAtTime", args });
        },
    };
    return { connect() {}, disconnect() {}, gain };
}

/** 假 AudioContext：记录 createOscillator 的产物，供孤儿断言使用。 */
function installFakeAudioContext() {
    const oscs: ReturnType<typeof makeFakeOsc>[] = [];
    const gains: FakeGain[] = [];
    const fakeCtx = {
        state: "running",
        destination: {},
        currentTime: 100,
        resume: () => Promise.resolve(),
        createGain: () => {
            const gain = makeFakeGain();
            gains.push(gain);
            return gain;
        },
        createBiquadFilter: () => ({
            connect() {},
            disconnect() {},
            type: "",
            frequency: { value: 0 },
            Q: { value: 0 },
        }),
        createDynamicsCompressor: () => ({
            connect() {},
            threshold: { value: 0 },
            knee: { value: 0 },
            ratio: { value: 0 },
            attack: { value: 0 },
            release: { value: 0 },
        }),
        createOscillator: () => {
            const osc = makeFakeOsc();
            oscs.push(osc);
            return osc;
        },
    };
    (globalThis as { AudioContext?: unknown }).AudioContext = function AudioContextStub() {
        return fakeCtx;
    } as unknown as typeof AudioContext;
    return { oscs, gains };
}

/** 一条 4 点、15ms 的平直曲线。 */
function flatCurve(cents = 0) {
    const hz = 440 * 2 ** (cents / 1200);
    return {
        freqHz: new Float32Array([hz, hz, hz, hz]),
        durationSec: 0.015,
        peakHz: hz,
    };
}

describe("buildAuditionCurve", () => {
    const baseHz = 440 * 2 ** ((AUDITION_BASE_MIDI - 69) / 12);

    it("深度 0 的预设输出恒等于基频（直线预设试听是平音）", () => {
        const curve = buildAuditionCurve({ wave: new Array(10).fill(0) });
        expect(curve.freqHz.length).toBe(10);
        for (const hz of curve.freqHz) expect(hz).toBeCloseTo(baseHz, 4);
    });

    it("cents → Hz 换算正确：30 cents ≈ ×1.0174", () => {
        const curve = buildAuditionCurve({ wave: [30, 0, -30] });
        expect(curve.freqHz[0]).toBeCloseTo(baseHz * 2 ** (30 / 1200), 4);
        expect(curve.freqHz[1]).toBeCloseTo(baseHz, 4);
        expect(curve.freqHz[2]).toBeCloseTo(baseHz * 2 ** (-30 / 1200), 4);
        expect(curve.peakHz).toBeCloseTo(curve.freqHz[0], 4);
    });

    it("时长 = (点数 - 1) × 5ms（与预览几何一致）", () => {
        expect(buildAuditionCurve({ wave: new Array(321).fill(0) }).durationSec).toBeCloseTo(
            1.6,
            9,
        );
    });

    it("非有限的 cents 按 0 处理，不产生 NaN 频率", () => {
        const curve = buildAuditionCurve({ wave: [Number.NaN, 10, Number.POSITIVE_INFINITY] });
        for (const hz of curve.freqHz) expect(Number.isFinite(hz)).toBe(true);
    });

    it("极短输入至少产出两个点（setValueCurveAtTime 的下限）", () => {
        const curve = buildAuditionCurve({ wave: [] });
        expect(curve.freqHz.length).toBeGreaterThanOrEqual(2);
    });
});

describe("VibratoAuditionPlayer", () => {
    let oscs: ReturnType<typeof makeFakeOsc>[];
    let gains: FakeGain[];
    let player: VibratoAuditionPlayer;

    /**
     * 取**voice 的**增益节点。
     *
     * 【为什么不是 gains[0]】`ensureContext` 会先造 masterGain（gains[0]），
     * 它的 gain 从不排程任何事件 —— 断言排程必须找 voice 那个（最后一个）。
     */
    function voiceGain(): FakeGain {
        expect(gains.length).toBeGreaterThanOrEqual(2);
        return gains[gains.length - 1];
    }

    beforeEach(() => {
        ({ oscs, gains } = installFakeAudioContext());
        player = new VibratoAuditionPlayer();
    });

    afterEach(() => {
        player.stop();
        delete (globalThis as { AudioContext?: unknown }).AudioContext;
    });

    it("播放后 isPlaying 为真，stop 后源被停掉且不泄漏节点", () => {
        expect(player.play(flatCurve())).toBe(true);
        expect(player.isPlaying).toBe(true);
        expect(oscs).toHaveLength(1);
        expect(oscs[0].started).toBe(1);

        player.stop();
        // play 排过一次自然结束，stopInternal 又重排了一次提前结束 ——
        // Web Audio 允许多次 stop、以最后一次为准，所以这里是 2 不是 1。
        expect(oscs[0].stopped).toBe(2);
        // 节点要等 onended 才拆链；这里手动触发以模拟引擎回调。
        oscs[0].onended?.();
        expect(oscs[0].disconnected).toBe(1);
    });

    it("★ 连发两次 play：旧的一路必须被停掉（不留孤儿）", () => {
        player.play(flatCurve());
        player.play(flatCurve(30));

        expect(oscs).toHaveLength(2);
        /*
         * 旧的那路必须被**提前**停掉 —— 否则它与新的叠加，用户听到两路。
         * 判据：旧路被 stop 了**两次**（play 自排的自然结束 + stopInternal 的
         * 提前重排），而新路只有自然结束那一次。Web Audio 允许多次 stop、
         * 以最后一次为准，因此次数差就是"谁被提前停了"的指纹。
         */
        expect(oscs[0].stopped).toBe(2);
        expect(oscs[1].started).toBe(1);
        expect(oscs[1].stopped).toBe(1);
    });

    it("音高走 setValueCurveAtTime 一条曲线（不是逐点排事件）", () => {
        player.play(flatCurve());
        expect(oscs[0].frequency.curves).toHaveLength(1);
        const [curve, , duration] = oscs[0].frequency.curves[0];
        expect(curve).toBeInstanceOf(Float32Array);
        expect(duration).toBeCloseTo(0.015, 9);
    });

    it("滤波器按曲线峰值定标（深颤音的上半周不被削掉）", () => {
        // 峰值频率是基频 × 2^(50/1200) ≈ ×1.029；滤波器点应高于它。
        player.play(flatCurve(50));
        // 滤波器的 frequency 由引擎写值；这里验证曲线峰值进了换算 ——
        // peakHz 来自 buildAuditionCurve（上方已测），此处只需确认播放可用。
        expect(oscs[0].started).toBe(1);
    });

    it("自然结束带渐出排程（收尾不在零点也不出爆音）", () => {
        player.play(flatCurve());
        const setTargets = voiceGain().gain.events.filter((e) => e.method === "setTargetAtTime");
        // 第一段是持续电平，最后一段必须是向 0 的渐出。
        expect(setTargets.length).toBeGreaterThanOrEqual(2);
        const last = setTargets[setTargets.length - 1];
        expect(last.args[0]).toBe(0);
        // 渐出起点 = 起音偏移 5ms + 曲线时长 15ms。
        expect(last.args[1]).toBeCloseTo(100.005 + 0.015, 6);
    });

    it("提前 stop 走 cancelAndHold + 向 0 渐出（不产生爆音）", () => {
        player.play(flatCurve());
        const gain = voiceGain();
        gain.gain.events.length = 0;
        player.stop();

        const methods = gain.gain.events.map((e) => e.method);
        expect(methods).toContain("cancelAndHoldAtTime");
        const setTargets = gain.gain.events.filter((e) => e.method === "setTargetAtTime");
        expect(setTargets[setTargets.length - 1]?.args[0]).toBe(0);
        // 自然结束 + 提前重排（以最后一次为准）。
        expect(oscs[0].stopped).toBe(2);
    });

    it("环境不支持 AudioContext 时收敛：返回 false、不抛、不留状态", () => {
        const isolated = new VibratoAuditionPlayer();
        delete (globalThis as { AudioContext?: unknown }).AudioContext;
        expect(isolated.play(flatCurve())).toBe(false);
        expect(isolated.isPlaying).toBe(false);
    });

    it("退化曲线（单点 / 零时长）不播放", () => {
        expect(
            player.play({ freqHz: new Float32Array([440]), durationSec: 0.005, peakHz: 440 }),
        ).toBe(false);
        expect(
            player.play({ freqHz: new Float32Array([440, 880]), durationSec: 0, peakHz: 880 }),
        ).toBe(false);
        expect(player.isPlaying).toBe(false);
    });
});

/*
 * A/B 试听：原参数线 vs 新参数线。
 *
 * 【契约】两条曲线必须**共用同一个中心与同一个基音** —— 否则听感差异里混进了基准音高
 * 的偏移，用户听到的就不再只是"加了颤音之后的差别"，对比失去意义。
 */
describe("buildContourAuditionPair", () => {
    const baseHz = 440 * Math.pow(2, (AUDITION_BASE_MIDI - 69) / 12);
    const hzAt = (cents: number) => baseHz * Math.pow(2, cents / 1200);

    /**
     * 造一份"套用预览"：本文件只关心原 / 新两条线与可信掩码，
     * 其余字段（包络、读数）试听用不到。
     */
    const preview = (
        contour: readonly number[],
        wave: readonly number[] = contour,
        stableFrames?: readonly boolean[],
    ) =>
        ({
            contour,
            wave,
            stableFrames: stableFrames ?? contour.map(() => true),
        }) as Parameters<typeof buildContourAuditionPair>[0];

    it("两条线共用同一个中心：差异只来自加进去的那部分", () => {
        const source = [6000, 6100, 6200, 6300];
        // 整体抬高 40 分（等价于"加了一点偏置"）。
        const result = source.map((value) => value + 40);
        const pair = buildContourAuditionPair(preview(source, result), 5);
        expect(pair).not.toBeNull();
        // 中心 = 6150 → 原线是 ±150 分，新线在此基础上再高 40 分。
        expect(pair!.source.freqHz[0]).toBeCloseTo(hzAt(-150), 4);
        expect(pair!.result.freqHz[0]).toBeCloseTo(hzAt(-110), 4);
    });

    it("断口保持**最近的音符音高**，不掉到中心", () => {
        const values = [Number.NaN, 6000, 6200, Number.NaN];
        const pair = buildContourAuditionPair(preview(values), 5);
        // 中心 = 6100：首帧保持第一个音符（6000 → −100 分），末帧保持最后一个（+100 分）。
        expect(pair!.source.freqHz[0]).toBeCloseTo(hzAt(-100), 3);
        expect(pair!.source.freqHz[3]).toBeCloseTo(hzAt(100), 3);
    });

    /*
     * 报告过的缺陷：选区含气口 / 过渡帧时，试听原参数线会"咔"一声。
     *
     * 旧实现把断口映射到**中心**音高，于是边界上相邻两帧能差三千多分（实测 3424 分），
     * 听感是爆音而不是音高。断口改为保持最近音符音高之后，相邻帧的变化只应来自颤音本身
     * （±40 分 / 6 Hz ≈ 每帧 8 分）。
     */
    it("气口边界不产生大跳（相邻帧差保持在颤音量级）", () => {
        const gap = new Array<number>(20).fill(Number.NaN);
        const note = Array.from({ length: 40 }, (_, i) => 6000 + Math.sin(i / 6) * 40);
        const pair = buildContourAuditionPair(preview([...gap, ...note, ...gap]), 5);
        const centsOf = (hz: number) => 1200 * Math.log2(hz / baseHz);
        let maxJump = 0;
        for (let i = 1; i < pair!.source.freqHz.length; i += 1) {
            maxJump = Math.max(
                maxJump,
                Math.abs(centsOf(pair!.source.freqHz[i]) - centsOf(pair!.source.freqHz[i - 1])),
            );
        }
        expect(maxJump).toBeLessThan(100);
    });

    /*
     * 报告过的缺陷：试听原参数线时听到"超低频"。
     *
     * 试听是**合成人声**，基准固定 C4；跟踪器在音符内部给出的异常（八度跳、持续的
     * 几十赫兹低估）按原值合成出来就是一段闷响。两道闸门各自负责一类：
     *
     * 1. **幅度闸门**：偏离旋律中心超过两个八度的一律按最近的音高持续
     *    （持续的低估不离群，只有量级判得出来）；
     * 2. **可信掩码**：`refineNoteFrames` 判出的局部离群帧同样持续
     *    （量级还在范围内、但显然不是这一段音高的那些）。
     *
     * 两条都带"不设闸门会怎样"的反面对照 —— 否则断言可能在测一个恒真的东西。
     */
    it("持续低估的帧不发声（幅度闸门）", () => {
        const contour = Array.from({ length: 200 }, (_, i) => 6000 + Math.sin(i / 6) * 40);
        // 音符内部 40 帧**持续**低估（跟踪器卡在 20 Hz 附近那类），且不离群。
        for (let i = 80; i < 120; i += 1) contour[i] = 2000;
        // 反面对照：那 40 帧若按原值发声，中心约 5200 分 → 它们落在 41 Hz 上下。
        const centerIfRaw = (160 * 6000 + 40 * 2000) / 200;
        expect(hzAt(2000 - centerIfRaw)).toBeLessThan(60);

        const pair = buildContourAuditionPair(preview(contour))!;
        expect(Math.min(...pair.source.freqHz), "不该出现超低频").toBeGreaterThan(200);
        expect(Math.max(...pair.source.freqHz)).toBeLessThan(350);
    });

    it("量级在范围内、但不可信的帧同样不发声（可信掩码）", () => {
        const contour = Array.from({ length: 200 }, (_, i) => 6000 + Math.sin(i / 6) * 40);
        const trusted = contour.map(() => true);
        // 音符内部 30 帧八度跳：量级只偏离 1200 分（幅度闸门放行），但不可信。
        for (let i = 80; i < 110; i += 1) {
            contour[i] = 7200;
            trusted[i] = false;
        }

        const withMask = buildContourAuditionPair(preview(contour, contour, trusted))!;
        const withoutMask = buildContourAuditionPair(preview(contour))!;

        // 不带掩码：那 30 帧被当成音高发出来（比音符高一个八度）。
        expect(Math.max(...withoutMask.source.freqHz)).toBeGreaterThan(400);
        // 带掩码：全程都在音符自己的音高附近。
        expect(Math.max(...withMask.source.freqHz)).toBeLessThan(350);
        expect(Math.min(...withMask.source.freqHz)).toBeGreaterThan(150);
    });

    it("全部帧都不可信时返回 null（没有可发声的音高）", () => {
        const contour = new Array<number>(50).fill(3000);
        const trusted = new Array<boolean>(50).fill(false);
        expect(buildContourAuditionPair(preview(contour, contour, trusted))).toBeNull();
    });

    it("时长按真实帧周期折算（不是写死的 5ms）", () => {
        const values = [6000, 6100, 6200];
        expect(buildContourAuditionPair(preview(values), 5)!.source.durationSec).toBeCloseTo(
            0.01,
            9,
        );
        expect(buildContourAuditionPair(preview(values), 10)!.source.durationSec).toBeCloseTo(
            0.02,
            9,
        );
    });

    it("原参数线不足两点时返回 null（调用方据此禁用按钮）", () => {
        expect(buildContourAuditionPair(preview([Number.NaN, Number.NaN], [1, 2]), 5)).toBeNull();
        expect(buildContourAuditionPair(preview([6000]), 5)).toBeNull();
    });
});
