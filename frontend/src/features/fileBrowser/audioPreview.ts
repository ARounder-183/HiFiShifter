import { fileBrowserApi, type AudioPreviewData } from "../../services/api/fileBrowser";

/**
 * 音频预览播放引擎（单例）
 * 基于 Web Audio API，用于文件浏览器中的音频文件预览播放。
 */
class AudioPreviewEngine {
    private ctx: AudioContext | null = null;
    private gainNode: GainNode | null = null;
    private source: AudioBufferSourceNode | null = null;
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
     */
    async play(filePath: string, onEnd?: () => void): Promise<void> {
        this.stop();
        this.onEndCallback = onEnd ?? null;
        const { ctx, gain } = this.ensureContext();

        // 记录当前的播放会话 ID
        const currentSession = ++this.playSessionId;

        let buffer = this.cache.get(filePath);
        if (!buffer) {
            // 从后端获取 PCM 数据
            const data: AudioPreviewData = await fileBrowserApi.readAudioPreview(filePath, 480_000);

            // 如果 await 期间用户点击了其他音频，直接放弃当前操作
            if (currentSession !== this.playSessionId) return;

            buffer = this.decodePreviewData(ctx, data);
            this.cache.set(filePath, buffer);

            // 限制缓存数量为 20 个，超出则淘汰最旧的
            if (this.cache.size > 20) {
                const oldestKey = this.cache.keys().next().value;
                if (oldestKey) this.cache.delete(oldestKey);
            }
        }

        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(gain);
        source.onended = () => {
            if (this.currentFile === filePath) {
                this.currentFile = null;
                this.source = null;
                this.onEndCallback?.();
            }
        };
        source.start();
        this.source = source;
        this.currentFile = filePath;
    }

    /** 停止当前播放 */
    stop(): void {
        if (this.source) {
            try {
                this.source.stop();
            } catch {
                /* 已停止 */
            }
            this.source.disconnect();
            this.source = null;
        }
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
