/**
 * 颤音试听播放器。
 *
 * 【为什么不用钢琴卷帘的 `PianoKeySound`】它的 API 是为琴键设计的：
 * `osc.frequency.value` 是静态的，塞不进一条随时间变化的音高曲线；`activeOscillators`
 * 以 `midiNote` 为键、`play` 对同一键直接 `return`，生命周期是"按住即响、松手才停"。
 * 试听要的是"一次调度、自动结束、可被下一次播放打断"——硬扩展会破坏那两条语义。
 * 因此这里**新建播放器、照抄它的音色配方**（triangle + lowpass + 压缩器 + 起音 ramp），
 * 听感与琴键一致 —— 正是"像钢琴卷帘的 MIDI 声"的要求。
 *
 * 【三条不变量】照抄 `features/fileBrowser/audioPreview.ts` 头部注释（那是本仓库
 * 音频单例的最好先例，注释明说"每次改动都要守住"）：
 *
 * 1. **任何 `osc.start()` 之前必须先停掉上一路**。`play()` 里虽无 `await`，但
 *    "连点两下"仍会让第二次播放与第一次叠在一起 —— 旧的必须原地放弃。否则会留下
 *    "还在响却已无人登记"的孤儿：`stop()` 停不掉它，用户听到的是两路叠加。
 * 2. **所有已 start 的源都必须登记**，`stop()` 逐个停，杜绝孤儿。
 * 3. **失败必须收敛**：`AudioContext` 不存在（jsdom 测试、无声卡）时返回 `false`
 *    静默退出，不把异常抛给调用方 —— 试听是锦上添花，不能因为它报错。
 */

import type { VibratoPreviewSamples } from "../../components/layout/vibrato/vibratoDialogLogic";
import { DEFAULT_FRAME_PERIOD_MS } from "./vibratoCurve";

/** 试听的基音：C4。 */
export const AUDITION_BASE_MIDI = 60;

/** 与 `PianoKeySound` 相同的起音 / 释放整形参数（防咔哒）。 */
const ATTACK_SEC = 0.008;
const SUSTAIN_TAU = 0.05;
const RELEASE_TAU = 0.015;
/** 自然结束与提前停止共用的渐出尾巴。 */
const STOP_TAIL_SEC = 0.1;
/** 试听音量。琴键用 0.25；试听独占一条链路，略抬高以盖过编辑时的视觉专注。 */
const VELOCITY = 0.3;

/** 试听曲线：逐点频率（Hz）+ 总时长 + 曲线峰值频率（供低通滤波器定标）。 */
export interface VibratoAuditionCurve {
    /** 逐点频率（Hz）。长度与预览采样点数一致。 */
    freqHz: Float32Array;
    /** 总时长（秒）。 */
    durationSec: number;
    /** 曲线上的最高频率（Hz）。 */
    peakHz: number;
}

/**
 * 由预览采样构建试听曲线。
 *
 * 【听到的 = 看到的】`preview.wave` 就是画布上画的那条曲线（已含渐入 / 渐强 /
 * 渐出 / 基线偏移 / 不规则度），这里只做 cents → Hz 的换算，**不另写一份公式**。
 *
 * 【基音为什么固定】预设管理器是全局预设库，与任何工程选区无关；跟着选区走会让
 * 同一个预设在不同上下文听感不同，试听就失去了"这个预设听起来什么样"的稳定含义。
 * 30 cents 在 C4（≈262Hz）上是 ±7.6Hz，听感明确。
 *
 * @param baseMidi 基音（MIDI 音高）。默认 C4。
 */
export function buildAuditionCurve(
    preview: Pick<VibratoPreviewSamples, "wave">,
    baseMidi: number = AUDITION_BASE_MIDI,
): VibratoAuditionCurve {
    const baseHz = 440 * Math.pow(2, (baseMidi - 69) / 12);
    const wave = preview.wave;
    const freqHz = new Float32Array(Math.max(2, wave.length));
    let peakHz = 0;
    for (let i = 0; i < freqHz.length; i += 1) {
        const cents = Number.isFinite(wave[i]) ? (wave[i] as number) : 0;
        freqHz[i] = baseHz * Math.pow(2, cents / 1200);
        peakHz = Math.max(peakHz, freqHz[i]);
    }
    const durationSec = ((freqHz.length - 1) * DEFAULT_FRAME_PERIOD_MS) / 1000;
    return { freqHz, durationSec, peakHz };
}

export class VibratoAuditionPlayer {
    private ctx: AudioContext | null = null;
    private compressor: DynamicsCompressorNode | null = null;
    private masterGain: GainNode | null = null;
    /** 最后一次 `play` 登记的结束回调（全部声音结束后触发一次）。 */
    private onEndedCallback: (() => void) | null = null;
    /** 所有仍在发声的源（正常情况至多一个；登记表用于兜底清理孤儿）。 */
    private liveVoices = new Set<{
        osc: OscillatorNode;
        filter: BiquadFilterNode;
        gain: GainNode;
    }>();

    private ensureContext(): AudioContext | null {
        const Ctor = (globalThis as { AudioContext?: typeof AudioContext }).AudioContext;
        if (!Ctor) return null;
        if (!this.ctx) {
            this.ctx = new Ctor();
            // 与 PianoKeySound 相同的链路尾端：压缩器防削波，主增益抬到听感一致。
            this.compressor = this.ctx.createDynamicsCompressor();
            this.compressor.threshold.value = -8;
            this.compressor.knee.value = 20;
            this.compressor.ratio.value = 6;
            this.compressor.attack.value = 0.005;
            this.compressor.release.value = 0.1;
            this.masterGain = this.ctx.createGain();
            this.masterGain.gain.value = 1.3;
            this.compressor.connect(this.masterGain);
            this.masterGain.connect(this.ctx.destination);
        }
        // 自动播放策略：play 只由按钮点击触发（用户手势），挂起时恢复即可。
        if (this.ctx.state === "suspended") {
            void this.ctx.resume();
        }
        return this.ctx;
    }

    get isPlaying(): boolean {
        return this.liveVoices.size > 0;
    }

    /**
     * 播放一段试听曲线；再次调用会先停掉上一路（约 0.1s 交叉渐出）。
     *
     * @param onEnded 最后一路声音结束（自然结束或被 `stop()` 打断）后的回调。
     *   只在**再无任何在响的声音**时触发 —— 供 UI 把按钮从「停止」切回「播放」，
     *   与 `audioPreview.play` 的 `onEnd` 同构。失败时不回调。
     * @returns 是否真的开始播放。环境不支持 `AudioContext` 时为 `false`
     *   （静默退出 —— 试听不能因为环境不支持而报错）。
     */
    play(curve: VibratoAuditionCurve, onEnded?: () => void): boolean {
        // 不变量 3：失败收敛。取不到 AudioContext 就什么都不做。
        const ctx = this.ensureContext();
        if (!ctx || !this.compressor) return false;
        // `setValueCurveAtTime` 至少要两个点；时长非正没有可听内容。
        if (curve.freqHz.length < 2 || !(curve.durationSec > 0)) return false;

        // 不变量 1：先停掉上一路再起这一路（停是渐出的，两路短暂交叉，不出爆音）。
        this.stopInternal();

        const osc = ctx.createOscillator();
        const filter = ctx.createBiquadFilter();
        const gain = ctx.createGain();

        osc.type = "triangle";
        // 一条曲线调度整个颤音：Web Audio 在点之间线性插值。预览是 5ms 一点
        // （200Hz 控制率），对音高变化来说完全听不出阶梯。提前 stop 时**不需要**
        // 取消这条曲线 —— gain 渐出 + osc.stop 之后它已无关紧要。
        const now = ctx.currentTime;
        const startTime = now + 0.005;
        const naturalEnd = startTime + curve.durationSec;
        osc.frequency.setValueCurveAtTime(curve.freqHz, startTime, curve.durationSec);

        filter.type = "lowpass";
        // 与 PianoKeySound 同式，但基准取**曲线峰值**而不是单一音高：
        // 颤音在扫，滤波器若按起点定标会把深颤音的上半周削掉。
        filter.frequency.value = Math.min(curve.peakHz * 4.5, 15000);
        filter.Q.value = 1.5;

        /*
         * 增益包络：起音 → 持续 → **自然结束的渐出**。
         * 最后一段不是可选项：曲线末端一般不在零点（相位任意的波形），若让
         * 振荡器在持续增益上直接 stop，收尾就是一声"咔"。提前停止走
         * `stopInternal()` 的 cancelAndHold + 渐出，与这条互不干扰。
         */
        gain.gain.value = 0;
        gain.gain.setValueAtTime(0, startTime);
        gain.gain.linearRampToValueAtTime(VELOCITY, startTime + ATTACK_SEC);
        gain.gain.setTargetAtTime(VELOCITY * 0.75, startTime + ATTACK_SEC, SUSTAIN_TAU);
        gain.gain.setTargetAtTime(0, naturalEnd, RELEASE_TAU);

        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this.compressor);

        osc.start(startTime);
        // 渐出尾巴之后再停，保证自然收尾无声。
        osc.stop(naturalEnd + STOP_TAIL_SEC);

        const voice = { osc, filter, gain };
        this.liveVoices.add(voice);
        this.onEndedCallback = onEnded ?? null;
        // 不变量 2：离开登记表并拆链。这里只清理自己的 voice —— 新会话有
        // 自己的 voice，互不触碰。
        osc.onended = () => {
            this.liveVoices.delete(voice);
            gain.disconnect();
            filter.disconnect();
            osc.disconnect();
            // 全部结束才通知：上一路被抢占时的 onended 不得把新一路的
            // 「正在播放」状态误清掉。
            if (this.liveVoices.size === 0) this.onEndedCallback?.();
        };
        return true;
    }

    /** 停止当前试听（渐出，与琴键松手同一手法，不产生爆音）。 */
    stop(): void {
        this.stopInternal();
    }

    private stopInternal(): void {
        const ctx = this.ctx;
        if (!ctx) return;
        const now = ctx.currentTime;
        for (const { osc, gain } of this.liveVoices) {
            // cancelAndHold 让"已排程的渐强"停在当前值再渐出；老引擎没有它时
            // 退化为取消排程 + 就地取值（与 PianoKeySound.stop 相同的兜底）。
            if (typeof gain.gain.cancelAndHoldAtTime === "function") {
                gain.gain.cancelAndHoldAtTime(now);
            } else {
                gain.gain.cancelScheduledValues(now);
                gain.gain.setValueAtTime(gain.gain.value, now);
            }
            gain.gain.setTargetAtTime(0, now, RELEASE_TAU);
            // stop 可重复调用，最后一次生效 —— 把排程的结束点提前到渐出尾巴末端。
            osc.stop(now + STOP_TAIL_SEC);
        }
    }
}

export const vibratoAudition = new VibratoAuditionPlayer();
