/**
 * 参数曲线二进制协议解析器
 *
 * 解析后端 `get_param_frames`（`binary=true`）返回的曲线数据。
 * 后端以 Base64 编码传输，本模块解码后按以下协议解析：
 *
 * 协议格式：[Header 8B] [orig f32[count]] [edit f32[count]]，小端序
 *
 * Header:
 *   bytes 0-3:  magic "PFB1" (4 bytes)
 *   bytes 4-7:  count (u32, little-endian) — orig/edit 各自的采样数
 *
 * 与后端 `commands/params.rs::encode_param_frames_binary` 配套，
 * 改动任一侧必须同步另一侧。
 *
 * 为什么用二进制
 * --------------
 * 参数曲线按帧存储（fp 约 5~6ms），长音频低缩放下一次取数可达几十万帧。
 * JSON number[] 方案下这是几十万个数字字面量：序列化慢、体积约为二进制的
 * 4 倍、解析还会长时间阻塞主线程。改为 Base64 + Float32Array 后：
 *   · 体积约缩小 4 倍（f32 定点 vs 数字字面量）；
 *   · 解码用 DataView 一次性建 Float32Array 视图，接近零拷贝。
 */

/** Header 字节数 */
const HEADER_SIZE = 8;

/** 魔数 "PFB1"（Param Frames Binary v1） */
const MAGIC = "PFB1";

/** 解码后的参数曲线二进制数据 */
export interface ParamFramesBinary {
    /** orig/edit 各自的采样数 */
    count: number;
    /** 原始曲线（Float32Array，零拷贝视图） */
    orig: Float32Array;
    /** 编辑曲线（Float32Array，零拷贝视图） */
    edit: Float32Array;
}

/**
 * 将 Base64 字符串解码为 ArrayBuffer。
 *
 * 使用 atob() + Uint8Array 一次性解码（与 waveformBinaryCodec 保持一致）。
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

/**
 * 解析参数曲线二进制数据。
 *
 * @param buffer 二进制数据（ArrayBuffer）
 * @returns 解码后的数据，或 null（数据无效时）
 */
export function decodeParamFramesBinary(buffer: ArrayBuffer): ParamFramesBinary | null {
    if (buffer.byteLength < HEADER_SIZE) return null;

    const view = new DataView(buffer);

    const magic = String.fromCharCode(
        view.getUint8(0),
        view.getUint8(1),
        view.getUint8(2),
        view.getUint8(3),
    );
    if (magic !== MAGIC) return null;

    const count = view.getUint32(4, true);
    const expectedSize = HEADER_SIZE + count * 4 * 2;
    if (count === 0 || buffer.byteLength < expectedSize) return null;

    // Float32Array 视图（零拷贝，直接引用原始 buffer）
    const orig = new Float32Array(buffer, HEADER_SIZE, count);
    const edit = new Float32Array(buffer, HEADER_SIZE + count * 4, count);

    return { count, orig, edit };
}

/**
 * 从 Base64 编码字符串直接解码参数曲线数据。
 *
 * 便捷方法，合并 base64ToArrayBuffer + decodeParamFramesBinary。
 */
export function decodeParamFramesFromBase64(base64: string): ParamFramesBinary | null {
    if (!base64 || base64.length < HEADER_SIZE) return null;
    let buffer: ArrayBuffer;
    try {
        buffer = base64ToArrayBuffer(base64);
    } catch {
        // 非法 Base64（理论上后端不会产生）→ 视为无效数据
        return null;
    }
    return decodeParamFramesBinary(buffer);
}

/**
 * 把解码结果转成普通 number[]，供既有渲染/编辑路径直接使用。
 *
 * 返回的两个数组是新建的（拷贝），不共享底层 buffer。
 */
export function paramFramesBinaryToArrays(decoded: ParamFramesBinary): {
    orig: number[];
    edit: number[];
} {
    const orig = new Array<number>(decoded.count);
    const edit = new Array<number>(decoded.count);
    for (let i = 0; i < decoded.count; i += 1) {
        orig[i] = decoded.orig[i];
        edit[i] = decoded.edit[i];
    }
    return { orig, edit };
}
