/**
 * 波形二进制协议解析器
 *
 * 解析后端 get_waveform_mipmap_binary 返回的二进制数据。
 * 协议格式：[Header 20B] [min f32[]] [max f32[]]
 *
 * Header:
 *   bytes 0-3:   magic "WFPK" (4 bytes)
 *   bytes 4-7:   sample_rate (u32, little-endian)
 *   bytes 8-11:  division_factor (u32, little-endian)
 *   bytes 12-15: peak_count (u32, little-endian)
 *   bytes 16-19: level (u32, little-endian)
 */

/** Header 字节数 */
const HEADER_SIZE = 20;

/** 魔数 "WFPK" */
const MAGIC = "WFPK";

/** 解码后的波形 mipmap 二进制数据 */
export interface WaveformMipmapBinary {
    /** 采样率 */
    sampleRate: number;
    /** 该级别的除数因子（L0=64, L1=512, L2=4096） */
    divisionFactor: number;
    /** 峰值数据点数量 */
    peakCount: number;
    /** mipmap 级别 (0/1/2) */
    level: number;
    /** 最小值数组（Float32Array，零拷贝视图） */
    min: Float32Array;
    /** 最大值数组（Float32Array，零拷贝视图） */
    max: Float32Array;
}

/**
 * 将 Tauri 返回的 number[] 转为 ArrayBuffer
 *
 * Tauri v2 对 Vec<u8> 返回类型会序列化为 number[]，
 * 需要手动转换为 ArrayBuffer 以获得 Float32Array 视图。
 */
export function numberArrayToArrayBuffer(arr: number[]): ArrayBuffer {
    const buffer = new ArrayBuffer(arr.length);
    const view = new Uint8Array(buffer);
    for (let i = 0; i < arr.length; i++) {
        view[i] = arr[i];
    }
    return buffer;
}

/**
 * 解码波形 mipmap 二进制数据
 *
 * @param buffer - 二进制数据（ArrayBuffer）
 * @returns 解码后的数据，或 null（数据无效时）
 */
export function decodeWaveformBinary(
    buffer: ArrayBuffer,
): WaveformMipmapBinary | null {
    if (buffer.byteLength < HEADER_SIZE) return null;

    const view = new DataView(buffer);

    // 验证魔数
    const magic = String.fromCharCode(
        view.getUint8(0),
        view.getUint8(1),
        view.getUint8(2),
        view.getUint8(3),
    );
    if (magic !== MAGIC) return null;

    const sampleRate = view.getUint32(4, true);
    const divisionFactor = view.getUint32(8, true);
    const peakCount = view.getUint32(12, true);
    const level = view.getUint32(16, true);

    const expectedSize = HEADER_SIZE + peakCount * 4 * 2;
    if (buffer.byteLength < expectedSize) return null;

    // Float32Array 视图（零拷贝，直接引用原始 buffer）
    const min = new Float32Array(buffer, HEADER_SIZE, peakCount);
    const max = new Float32Array(
        buffer,
        HEADER_SIZE + peakCount * 4,
        peakCount,
    );

    return { sampleRate, divisionFactor, peakCount, level, min, max };
}

/**
 * 从 Tauri 返回的 number[] 直接解码
 *
 * 便捷方法，合并 numberArrayToArrayBuffer + decodeWaveformBinary。
 */
export function decodeWaveformFromNumberArray(
    arr: number[],
): WaveformMipmapBinary | null {
    if (!arr || arr.length < HEADER_SIZE) return null;
    const buffer = numberArrayToArrayBuffer(arr);
    return decodeWaveformBinary(buffer);
}
