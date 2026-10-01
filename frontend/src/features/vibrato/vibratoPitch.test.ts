import { describe, expect, test } from "vitest";

import {
    MIN_NOTE_MS,
    PLAUSIBLE_PITCH_MAX,
    PLAUSIBLE_PITCH_MIN,
    isUnsetValue,
    planVibratoTarget,
    suppressNonTargetFrames,
    usesUnsetValue,
} from "./vibratoPitch";

/*
 * 音高哨兵适配的纯逻辑。
 *
 * 【为什么值得单独钉】这一层错了不会抛错，只会**静默改坏曲线**：锚点被气口或浊清
 * 边界的低值抹痕带偏，`baseline: "line"` 就把整段真实音高拉向那个错值；少了一次
 * 掩蔽，颤音就被写进根本不是音符的帧里。两者都只能靠断言才看得见。
 *
 * 默认帧周期 5 ms ⇒ 一个音符段至少要 20 帧（`MIN_NOTE_MS`）。
 */
const FP = 5;
const MIN_FRAMES = MIN_NOTE_MS / FP;

const zeros = (count: number) => new Array<number>(count).fill(0);
const flat = (count: number, value: number) => new Array<number>(count).fill(value);
/** 一段真实音高：从 `from` 起每帧升 0.05 半音（幅度足够小，仍是一个音符段）。 */
const glide = (count: number, from: number) =>
    Array.from({ length: count }, (_, i) => from + i * 0.05);

describe("usesUnsetValue / isUnsetValue", () => {
    test("只有音高用 0 当「无数据」哨兵", () => {
        expect(usesUnsetValue("pitch")).toBe(true);
        for (const param of ["dyn", "volume", "tension", "breath_gain"]) {
            expect(usesUnsetValue(param)).toBe(false);
        }
    });

    test("音高：判据是 > 0，0 / 负值 / 非有限值都算未检测", () => {
        expect(isUnsetValue("pitch", 0)).toBe(true);
        expect(isUnsetValue("pitch", -3)).toBe(true);
        expect(isUnsetValue("pitch", Number.NaN)).toBe(true);
        expect(isUnsetValue("pitch", Number.POSITIVE_INFINITY)).toBe(true);
        expect(isUnsetValue("pitch", 0.5)).toBe(false);
        expect(isUnsetValue("pitch", 60)).toBe(false);
    });

    test("其他参数：0 是合法值，不是哨兵", () => {
        expect(isUnsetValue("volume", 0)).toBe(false);
        expect(isUnsetValue("dyn", 0)).toBe(false);
    });
});

describe("planVibratoTarget：非哨兵参数", () => {
    test("整段可调制，锚点原样取首末帧（与既有行为逐字一致）", () => {
        const plan = planVibratoTarget("volume", [0, 1, 2], FP);
        expect(plan).not.toBeNull();
        expect(plan!.anchors).toEqual({ startValue: 0, endValue: 2 });
        expect(plan!.modulatable).toEqual([true, true, true]);
    });

    test("空数组返回 null", () => {
        expect(planVibratoTarget("volume", [], FP)).toBeNull();
        expect(planVibratoTarget("pitch", [], FP)).toBeNull();
    });
});

describe("planVibratoTarget：音高按音符段切分", () => {
    /*
     * 报告场景：选区首尾是气口，紧挨着气口还有几帧**低而非零**的过渡帧
     * （音高跟踪器在浊清边界给出的 20~40 Hz 低估 → MIDI 3~22）。
     *
     * 旧实现取"首个非零帧"当锚点，于是锚到 3；再往前的版本取"首帧"，锚到 0。
     * 两者都会把中间真实唱出来的 60 拖下去。
     */
    test("报告场景：边界上的低值过渡帧既不当锚点，也不被写入颤音", () => {
        const blip = [3, 8, 15, 20, 22];
        const values = [...zeros(10), ...blip, ...zeros(5), ...glide(60, 60), ...zeros(10)];
        const plan = planVibratoTarget("pitch", values, FP);
        expect(plan, "有一段真实音高，应当规划出落点").not.toBeNull();

        // 锚点必须是真实音高的代表值，而不是 3/8/15/20/22 里的任何一个。
        expect(plan!.anchors.startValue).toBeGreaterThan(59);
        expect(plan!.anchors.endValue).toBeGreaterThan(59);

        // 过渡帧与气口都不参与；真实音高段参与。
        for (let i = 0; i < 20; i += 1) expect(plan!.modulatable[i]).toBe(false);
        expect(plan!.modulatable[20]).toBe(true);
        expect(plan!.modulatable[79]).toBe(true);
        for (let i = 80; i < 90; i += 1) expect(plan!.modulatable[i]).toBe(false);
    });

    test("够长才算音符：19 帧（95ms）不成段，20 帧（100ms）成段", () => {
        const short = [...zeros(5), ...flat(MIN_FRAMES - 1, 60), ...zeros(5)];
        expect(planVibratoTarget("pitch", short, FP)).toBeNull();
        const long = [...zeros(5), ...flat(MIN_FRAMES, 60), ...zeros(5)];
        expect(planVibratoTarget("pitch", long, FP)).not.toBeNull();
    });

    test("门槛按实际帧周期折算，不写死帧数", () => {
        // 帧周期 10ms：9 帧 = 90ms 不够，10 帧 = 100ms 够。
        expect(planVibratoTarget("pitch", flat(9, 60), 10)).toBeNull();
        expect(planVibratoTarget("pitch", flat(10, 60), 10)).not.toBeNull();
    });

    /*
     * 值域闸门专治"够长但荒谬"的那种：跟踪器把一段 30~40 Hz 的低估坚持上百毫秒时，
     * 时长这道关拦不住它，而 MIDI 24（C1，32.7 Hz）以下不可能是人声基频。
     */
    test("够长但落在音高值域之外的低值段不算音符", () => {
        expect(planVibratoTarget("pitch", flat(40, 20), FP)).toBeNull();
        expect(planVibratoTarget("pitch", flat(40, 3), FP)).toBeNull();
        // 边界值本身算音符（值域是闭区间，与参数编辑器一致）。
        expect(planVibratoTarget("pitch", flat(40, PLAUSIBLE_PITCH_MIN), FP)).not.toBeNull();
        expect(planVibratoTarget("pitch", flat(40, PLAUSIBLE_PITCH_MAX), FP)).not.toBeNull();
        expect(planVibratoTarget("pitch", flat(40, PLAUSIBLE_PITCH_MAX + 1), FP)).toBeNull();
    });

    test("锚点取音符段的稳健代表值（中位数），不是边界那一帧", () => {
        // 起声处 5 帧衰减尾巴（仍然有声、也在值域内），随后是稳定的 60。
        const values = [45, 48, 52, 56, 58, ...flat(40, 60)];
        const plan = planVibratoTarget("pitch", values, FP)!;
        expect(plan.anchors.startValue).toBe(60);
        expect(plan.anchors.endValue).toBe(60);
    });

    test("内部空隙把音符段切开：锚点取首末段，空隙不受影响", () => {
        const values = [...flat(30, 60), ...zeros(10), ...flat(30, 64)];
        const plan = planVibratoTarget("pitch", values, FP)!;
        expect(plan.anchors.startValue).toBe(60);
        expect(plan.anchors.endValue).toBe(64);
        expect(plan.modulatable[29]).toBe(true);
        for (let i = 30; i < 40; i += 1) expect(plan.modulatable[i]).toBe(false);
        expect(plan.modulatable[40]).toBe(true);
    });

    test("整段未检测 / 没有任何够长的段 → null（没有可调制的对象）", () => {
        expect(planVibratoTarget("pitch", flat(50, 0), FP)).toBeNull();
        expect(planVibratoTarget("pitch", [Number.NaN, 0, Number.NaN], FP)).toBeNull();
    });
});

describe("suppressNonTargetFrames", () => {
    test("非目标帧写回 0，目标帧原样保留", () => {
        const values = [...zeros(10), ...flat(30, 60), ...zeros(10)];
        const plan = planVibratoTarget("pitch", values, FP)!;
        const result = new Array<number>(values.length).fill(99);
        suppressNonTargetFrames(result, plan);
        expect(result[0]).toBe(0);
        expect(result[20]).toBe(99);
        expect(result[39]).toBe(99);
        expect(result[45]).toBe(0);
    });

    test("非哨兵参数一个字节都不改", () => {
        const plan = planVibratoTarget("volume", [0, 1, 2], FP)!;
        const result = [7, 8, 9];
        suppressNonTargetFrames(result, plan);
        expect(result).toEqual([7, 8, 9]);
    });

    test("幂等：已还原过的结果再还原一次不变", () => {
        const values = [...zeros(10), ...flat(30, 60)];
        const plan = planVibratoTarget("pitch", values, FP)!;
        const once = new Array<number>(values.length).fill(42);
        suppressNonTargetFrames(once, plan);
        const twice = once.slice();
        suppressNonTargetFrames(twice, plan);
        expect(twice).toEqual(once);
    });

    test("长度不一致时按较短者对齐，不越界", () => {
        const plan = planVibratoTarget("pitch", [...zeros(10), ...flat(30, 60)], FP)!;
        const result = [5, 6];
        suppressNonTargetFrames(result, plan);
        expect(result).toEqual([0, 0]);
    });
});
