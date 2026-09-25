/**
 * 波形二进制协议解析器
 *
 * 解析后端 get_waveform_mipmap_binary 返回的二进制数据。
 * 后端以 Base64 编码传输，前端解码后按以下协议解析：
 *
 * 协议格式 v2（当前）：[Header 28B] [ch0_min f32[]] [ch0_max f32[]] [ch1_min] [ch1_max] ...
 *
 * Header:
 *   bytes 0-3:   magic "WFPK" (4 bytes)
 *   bytes 4-7:   format_version = 2 (u32, little-endian)
 *   bytes 8-11:  sample_rate (u32, little-endian)
 *   bytes 12-15: division_factor (u32, little-endian)
 *   bytes 16-19: peak_count (u32, little-endian)
 *   bytes 20-23: level (u32, little-endian)
 *   bytes 24-27: channels (u32, little-endian)
 *
 * 峰值数据按**声道分块**排列（ch0_min → ch0_max → ch1_min → ch1_max …），
 * 每块 peak_count × f32。
 *
 * 协议格式 v1（旧后端兜底）：[Header 20B] [min f32[]] [max f32[]]，无
 * format_version/channels 字段；数据是跨声道合并的单组包络，按 channels=1 解析。
 */

/** v2 Header 字节数 */
const HEADER_SIZE_V2 = 28;

/** v1 Header 字节数 */
const HEADER_SIZE_V1 = 20;

/** v2 协议版本号 */
const FORMAT_VERSION_V2 = 2;

/** 解码后的波形 mipmap 二进制数据 */
export interface WaveformMipmapBinary {
    /** 采样率 */
    sampleRate: number;
    /** 该级别的除数因子（L0=16, L1=512, L2=4096） */
    divisionFactor: number;
    /** 峰值数据点数量 */
    peakCount: number;
    /** mipmap 级别 (0/1/2) */
    level: number;
    /** 声道数（显示语义只区分 1/2；>2 声道源的第 3+ 声道仅参与包络合并） */
    channels: 1 | 2;
    /**
     * 跨声道合并包络（逐峰值取各声道 min 的最小值 / max 的最大值）。
     * 兼容所有既有消费者；v1 数据即原始数组本身（零拷贝）。
     */
    min: Float32Array;
    max: Float32Array;
    /** 声道 0 的 min/max 视图（零拷贝） */
    ch0Min: Float32Array;
    ch0Max: Float32Array;
    /** 声道 1 的 min/max 视图；单声道时指向 ch0（同一数据） */
    ch1Min: Float32Array;
    ch1Max: Float32Array;
}

/**
 * 将 Base64 字符串解码为 ArrayBuffer
 *
 * 使用 atob() + Uint8Array 一次性解码，
 * 替代旧版逐字节 number[] → Uint8Array 拷贝（性能提升 5-10x）。
 */
export function base64ToArrayBuffer(base64: string): ArrayBuffer {
    const binary = atob(base64);
    const len = binary.length;
    const buffer = new ArrayBuffer(len);
    const view = new Uint8Array(buffer);
    for (let i = 0; i < len; i++) {
        view[i] = binary.charCodeAt(i);
    }
    return buffer;
}

function isMagic(view: DataView): boolean {
    return (
        view.getUint8(0) === 0x57 && // W
        view.getUint8(1) === 0x46 && // F
        view.getUint8(2) === 0x50 && // P
        view.getUint8(3) === 0x4b // K
    );
}

/**
 * 解码波形 mipmap 二进制数据（自动识别 v2 / v1 布局）
 *
 * @param buffer - 二进制数据（ArrayBuffer）
 * @returns 解码后的数据，或 null（数据无效时）
 */
export function decodeWaveformBinary(buffer: ArrayBuffer): WaveformMipmapBinary | null {
    if (buffer.byteLength < HEADER_SIZE_V1) return null;

    const view = new DataView(buffer);
    if (!isMagic(view)) return null;

    // v1 头部没有 format_version 字段：bytes 4-7 直接是 sample_rate。
    // 合法采样率不可能把 4 字节任一高位字节摆成 2 —— 用它区分两代布局。
    const v2FormatVersion = view.getUint32(4, true);
    const isV2 = v2FormatVersion === FORMAT_VERSION_V2;

    if (isV2) {
        if (buffer.byteLength < HEADER_SIZE_V2) return null;
        const sampleRate = view.getUint32(8, true);
        const divisionFactor = view.getUint32(12, true);
        const peakCount = view.getUint32(16, true);
        const level = view.getUint32(20, true);
        const rawChannels = view.getUint32(24, true);
        const displayChannels: 1 | 2 = rawChannels >= 2 ? 2 : 1;

        const blockCount = Math.max(1, rawChannels);
        const expectedSize = HEADER_SIZE_V2 + peakCount * blockCount * 4 * 2;
        if (buffer.byteLength < expectedSize) return null;

        // 逐声道零拷贝视图（>2 声道源只保留前两声道的视图，其余仅参与包络）。
        const ch0Min = new Float32Array(buffer, HEADER_SIZE_V2, peakCount);
        const ch0Max = new Float32Array(buffer, HEADER_SIZE_V2 + peakCount * 4, peakCount);
        const ch1Min =
            blockCount >= 2
                ? new Float32Array(buffer, HEADER_SIZE_V2 + peakCount * 8, peakCount)
                : ch0Min;
        const ch1Max =
            blockCount >= 2
                ? new Float32Array(buffer, HEADER_SIZE_V2 + peakCount * 12, peakCount)
                : ch0Max;

        if (blockCount <= 1) {
            return {
                sampleRate,
                divisionFactor,
                peakCount,
                level,
                channels: 1,
                min: ch0Min,
                max: ch0Max,
                ch0Min,
                ch0Max,
                ch1Min,
                ch1Max,
            };
        }

        // 跨声道合并包络：min 取各声道最小、max 取各声道最大（含第 3+ 声道）。
        const min = new Float32Array(peakCount);
        const max = new Float32Array(peakCount);
        min.set(ch0Min);
        max.set(ch0Max);
        for (let ch = 1; ch < blockCount; ch++) {
            const base = HEADER_SIZE_V2 + ch * peakCount * 8;
            const blockMin = new Float32Array(buffer, base, peakCount);
            const blockMax = new Float32Array(buffer, base + peakCount * 4, peakCount);
            for (let i = 0; i < peakCount; i++) {
                const bmin = blockMin[i];
                if (bmin < min[i]) min[i] = bmin;
                const bmax = blockMax[i];
                if (bmax > max[i]) max[i] = bmax;
            }
        }

        return {
            sampleRate,
            divisionFactor,
            peakCount,
            level,
            channels: displayChannels,
            min,
            max,
            ch0Min,
            ch0Max,
            ch1Min,
            ch1Max,
        };
    }

    // ── v1 兜底：单组跨声道包络 ──
    const sampleRate = view.getUint32(4, true);
    const divisionFactor = view.getUint32(8, true);
    const peakCount = view.getUint32(12, true);
    const level = view.getUint32(16, true);

    const expectedSize = HEADER_SIZE_V1 + peakCount * 4 * 2;
    if (buffer.byteLength < expectedSize) return null;

    const min = new Float32Array(buffer, HEADER_SIZE_V1, peakCount);
    const max = new Float32Array(buffer, HEADER_SIZE_V1 + peakCount * 4, peakCount);

    return {
        sampleRate,
        divisionFactor,
        peakCount,
        level,
        channels: 1,
        min,
        max,
        ch0Min: min,
        ch0Max: max,
        ch1Min: min,
        ch1Max: max,
    };
}

/**
 * 从 Base64 编码字符串直接解码波形 mipmap 数据
 *
 * 便捷方法，合并 base64ToArrayBuffer + decodeWaveformBinary。
 * 替代旧版 decodeWaveformFromNumberArray（基于 JSON number[] 的低效方案）。
 */
export function decodeWaveformFromBase64(base64: string): WaveformMipmapBinary | null {
    if (!base64 || base64.length < HEADER_SIZE_V1) return null;
    let buffer: ArrayBuffer;
    try {
        buffer = base64ToArrayBuffer(base64);
    } catch {
        // 非法 base64（atob 对字母表外的字符抛 InvalidCharacterError）：按本
        // 函数与 decodeWaveformBinary 的"无效数据返回 null"契约兜住，不能让
        // 单个损坏载荷把批量预载等调用方炸掉。
        return null;
    }
    return decodeWaveformBinary(buffer);
}
