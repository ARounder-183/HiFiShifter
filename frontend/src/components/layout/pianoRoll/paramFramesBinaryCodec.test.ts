import { test } from "vitest";

import {
    base64ToArrayBuffer,
    decodeParamFramesBinary,
    decodeParamFramesFromBase64,
    paramFramesBinaryToArrays,
} from "./paramFramesBinaryCodec.js";

/**
 * 这里的断言锁住的是前后端的二进制协议契约：
 *   [Header 8B: magic "PFB1" + count u32LE][orig f32[count]][edit f32[count]]
 *
 * 最关键的是**平面布局**（先整个 orig 再整个 edit），而不是交错。
 * 平面布局才允许前端用 `new Float32Array(buffer, offset, count)` 建零拷贝视图；
 * 若后端改成交错布局，解码器会拿到完全错误的数值且不报错。
 */
test("components/layout/pianoRoll/paramFramesBinaryCodec.test.ts scripted checks", async () => {
    // 与后端 encode_param_frames_binary 一致的编码器（测试侧自建，避免依赖 Rust）
    function encodePlanar(orig: number[], edit: number[]): string {
        const bytes: number[] = [];
        const pushF32 = (v: number) => {
            const b = new Uint8Array(4);
            new DataView(b.buffer).setFloat32(0, v, true);
            bytes.push(...b);
        };
        const pushU32 = (v: number) => {
            const b = new Uint8Array(4);
            new DataView(b.buffer).setUint32(0, v, true);
            bytes.push(...b);
        };
        bytes.push(0x50, 0x46, 0x42, 0x31); // "PFB1"
        pushU32(orig.length);
        for (const v of orig) pushF32(v);
        for (const v of edit) pushF32(v);
        let bin = "";
        for (const b of bytes) bin += String.fromCharCode(b);
        return btoa(bin);
    }

    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    // ── 正常往返 ─────────────────────────────────────────────────────────
    {
        const orig = [0, 1, -2.5, 127];
        const edit = [0, 0.5, 60, 0];
        const decoded = decodeParamFramesFromBase64(encodePlanar(orig, edit));
        if (!decoded) throw new Error("decoded should not be null");
        assertEqual(decoded.count, 4, "count");
        // f32 精度：用容差比较
        for (let i = 0; i < orig.length; i += 1) {
            assertEqual(Math.abs(decoded.orig[i] - orig[i]) < 1e-5, true, `orig[${i}]`);
            assertEqual(Math.abs(decoded.edit[i] - edit[i]) < 1e-5, true, `edit[${i}]`);
        }
        // 转数组后长度一致
        const { orig: o2, edit: e2 } = paramFramesBinaryToArrays(decoded);
        assertEqual(o2.length, 4, "toArrays orig length");
        assertEqual(e2.length, 4, "toArrays edit length");
        assertEqual(o2[2] === decoded.orig[2], true, "toArrays passthrough");
    }

    // ── 布局必须为平面：交错数据会被解码成错误数值 ────────────────────────
    {
        // 构造交错数据（orig[0],edit[0],orig[1],edit[1]...）
        const interleaved = encodePlanar([0, 0], [1, 1]);
        const decoded = decodeParamFramesFromBase64(interleaved);
        if (!decoded) throw new Error("decoded should not be null");
        // 若解码器误按交错读，orig 会是 [0,1]；按平面读应为 [0,0]
        assertEqual(decoded.orig[0], 0, "planar orig[0]");
        assertEqual(decoded.orig[1], 0, "planar orig[1]");
        assertEqual(decoded.edit[0], 1, "planar edit[0]");
    }

    // ── 非法输入必须返回 null，不能抛异常 ────────────────────────────────
    assertEqual(decodeParamFramesFromBase64(""), null, "empty base64");
    assertEqual(decodeParamFramesFromBase64("!!!not-base64!!!"), null, "bad base64");

    {
        // 魔数错误
        const bytes = new Uint8Array(16);
        new DataView(bytes.buffer).setUint32(4, 1, true);
        let bin = "";
        for (const b of bytes) bin += String.fromCharCode(b);
        assertEqual(decodeParamFramesFromBase64(btoa(bin)), null, "wrong magic");
    }

    {
        // count=0 → 无有效数据
        const bytes = new Uint8Array(8);
        new DataView(bytes.buffer).setUint32(4, 0, true);
        let bin = "";
        for (const b of bytes) bin += String.fromCharCode(b);
        assertEqual(decodeParamFramesFromBase64(btoa(bin)), null, "zero count");
    }

    {
        // 声明 count=4 但数据被截断
        const good = decodeParamFramesFromBase64(encodePlanar([1, 2, 3, 4], [5, 6, 7, 8]));
        if (!good) throw new Error("good should not be null");
        // 手工截掉尾部：只剩 header + 3 个 orig
        const raw = new Uint8Array(8 + 4 * 4 + 4);
        new DataView(raw.buffer).setUint32(4, 4, true);
        let bin = "";
        for (const b of raw) bin += String.fromCharCode(b);
        const truncated = decodeParamFramesBinary(base64ToArrayBuffer(btoa(bin)));
        assertEqual(truncated, null, "truncated payload");
    }
});
