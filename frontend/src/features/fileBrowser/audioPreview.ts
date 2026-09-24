import { fileBrowserApi, type AudioPreviewData } from "../../services/api/fileBrowser";
import { reportFrontendError } from "../../services/frontendErrorLog";

/**
 * 音频预览播放引擎（单例）
 * 基于 Web Audio API，用于文件浏览器中的音频文件预览播放。
 *
 * 【不变量（每次改动都要守住）】
 * 1. **任何 `source.start()` 之前必须紧邻一次会话检查**：`play()` 在 `await` 取
 *    数据期间可能被新的 `play()` / `stop()` 抢占，抢占后旧的那次必须原地放弃。
 *    否则会出现"第二路音源已经在响、却不在 `this.source` 里"的孤儿：`stop()`
 *    停不掉它，它的 `onended` 也因为 `currentFile` 已变而不回调 —— 用户听到的是
 *    "重复播放并且和旧的叠加"。
 * 2. **所有已 start 的源都必须被登记**，`stop()` 逐个停止，杜绝孤儿。
 * 3. **失败必须收敛**：解码/取数失败时不能把状态留在"正在播放"，也不能把
 *    rejection 抛给调用方（调用方无从处理）。失败静默退出，只上报诊断。
 */
class AudioPreviewEngine {
    private ctx: AudioContext | null = null;
    private gainNode: GainNode | null = null;
    private source: AudioBufferSourceNode | null = null;
    /** 所有仍在发声的源（正常情况下至多一个；登记表用于兜底清理孤儿）。 */
    private liveSources = new Set<AudioBufferSourceNode>();
    private currentFile: string | null = null;
    private cache = new Map<string, AudioBuffer>();
    private onEndCallback: (() => void) | null = null;
    private playSessionId = 0;

    private ensureContext(): { ctx: AudioContext; gain: GainNode } {
        if (!this.ctx) {
            this.ctx = new AudioContext();
            this.gainNode = this.ctx.createGain();
            this.gainNode.connect(this.ctx.destination);
        }
        // 如果 AudioContext 被挂起（autoplay policy），恢复它
        if (this.ctx.state === "suspended") {
            void this.ctx.resume();
        }
        return { ctx: this.ctx, gain: this.gainNode! };
    }

    /**
     * 播放指定文件的预览，始终从头开始。
     *
     * @param onEnd 播放结束（自然结束或被 `stop()` 打断）后的回调；**失败时不回调**
     *   —— 失败由本函数内部收敛并返回 `false`，调用方据此回滚 UI 状态。
     * @returns 是否真的开始播放。取数/解码失败或中途被抢占时为 `false`。
     */
    async play(filePath: string, onEnd?: () => void): Promise<boolean> {
        this.stopInternal();
        this.onEndCallback = onEnd ?? null;
        const session = ++this.playSessionId;

        try {
            const { ctx, gain } = this.ensureContext();

            let buffer = this.cache.get(filePath);
            if (!buffer) {
                const data: AudioPreviewData = await fileBrowserApi.readAudioPreview(
                    filePath,
                    480_000,
                );
                // await 期间被新的 play / stop 抢占：放弃，且**不要**碰当前状态。
                if (session !== this.playSessionId) return false;
                buffer = this.decodePreviewData(ctx, data);
                this.cache.set(filePath, buffer);
                // 限制缓存数量为 20 个，超出则淘汰最旧的
                if (this.cache.size > 20) {
                    const oldestKey = this.cache.keys().next().value;
                    if (oldestKey) this.cache.delete(oldestKey);
                }
            }

            // 缓存命中路径没有 await，但仍要再查一次：调用方可能在同一次事件里
            // 连续调用了 play()（例如"点了两下"）。
            if (session !== this.playSessionId) return false;

            const source = ctx.createBufferSource();
            source.buffer = buffer;
            source.connect(gain);
            source.onended = () => {
                this.liveSources.delete(source);
                if (this.currentFile === filePath && this.source === source) {
                    this.currentFile = null;
                    this.source = null;
                    this.onEndCallback?.();
                }
            };
            source.start();
            this.liveSources.add(source);
            this.source = source;
            this.currentFile = filePath;
            return true;
        } catch (error) {
            // 失败即退出播放：不提示用户（按需求），只上报诊断。
            if (session === this.playSessionId) {
                this.stopInternal();
            }
            reportFrontendError(
                `[audioPreview] 预览失败：${filePath} — ${
                    error instanceof Error ? error.message : String(error)
                }`,
            );
            return false;
        }
    }

    /** 停止当前播放（公开入口：同时作废所有在飞的 `play()`）。 */
    stop(): void {
        this.playSessionId += 1;
        this.stopInternal();
    }

    /**
     * 内部停止：停掉**所有**已登记的源。
     *
     * 不自增会话号（`play()` 内部先调用它，会话号由调用方推进）。
     */
    private stopInternal(): void {
        for (const source of this.liveSources) {
            try {
                source.onended = null;
                source.stop();
            } catch {
                /* 已经停止 */
            }
            try {
                source.disconnect();
            } catch {
                /* 已断开 */
            }
        }
        this.liveSources.clear();
        this.source = null;
        this.currentFile = null;
        this.onEndCallback = null;
    }

    /** 设置预览音量 (0~1) */
    setVolume(v: number): void {
        const { gain } = this.ensureContext();
        gain.gain.value = Math.max(0, Math.min(1, v));
    }

    /** 当前是否正在播放 */
    isPlaying(): boolean {
        return this.source !== null && this.currentFile !== null;
    }

    /** 获取正在播放的文件路径 */
    getCurrentFile(): string | null {
        return this.currentFile;
    }

    /** 清除缓存 */
    clearCache(): void {
        this.cache.clear();
    }

    /**
     * 将后端返回的 base64 编码 f32 LE interleaved PCM 数据
     * 解码为 Web Audio AudioBuffer
     */
    private decodePreviewData(ctx: AudioContext, data: AudioPreviewData): AudioBuffer {
        const binaryStr = atob(data.pcmBase64);
        const bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
            bytes[i] = binaryStr.charCodeAt(i);
        }
        const floats = new Float32Array(bytes.buffer);
        const channels = Math.max(1, data.channels);
        const frames = Math.floor(floats.length / channels);
        const audioBuffer = ctx.createBuffer(channels, frames, data.sampleRate);

        // 反交错到各声道
        for (let ch = 0; ch < channels; ch++) {
            const channelData = audioBuffer.getChannelData(ch);
            for (let f = 0; f < frames; f++) {
                channelData[f] = floats[f * channels + ch];
            }
        }

        return audioBuffer;
    }
}

/** 全局单例 */
export const audioPreview = new AudioPreviewEngine();
