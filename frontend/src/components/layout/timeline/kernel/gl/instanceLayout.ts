/**
 * 时间轴渲染内核 · 实例缓冲布局（唯一来源）
 *
 * 【主要内容】
 * 定义 25 float/实例的字段偏移、实例模式常量，以及「平面矩形实例」的写入函数。
 *
 * 【作用】
 * 实例缓冲是裸 `Float32Array`：字段偏移一旦在写入端与读取端（GL 属性绑定）之间漂移，
 * 不会报错、只会画错。因此偏移必须只有一个来源——本文件。偏移值与既有
 * `runtime/timelineClipGlRenderer` 的 `OFF_*` 常量**逐项一致**（Spike 阶段复制，
 * 阶段 1 合并为单一来源）。
 *
 * 【与其他模块的关系】
 * - 上游：`scene/gridInstances` 产出 `FlatInstance`，本文件负责写入缓冲；
 *   `scene/clipInstances` 复用既有的 `buildClipBodyInstance` 写 BOX 实例。
 * - 下游：`gl/sdfBoxProgram` 用同一组偏移绑定属性指针。
 * - 独立性：纯逻辑，不依赖 WebGL / DOM，可直接单测。
 */

import type { FlatInstance } from "../scene/instanceTypes";

/** 单实例 float 数（与既有 `CLIP_INSTANCE_FLOATS` 一致）。 */
export const CLIP_INSTANCE_FLOATS = 25;

/**
 * 字段偏移（float 下标）。
 *
 * 【评审检查项】必须与 `runtime/timelineClipGlRenderer` 的 `OFF_*` 逐项一致；
 * `instanceLayout.test.ts` 对关键项做了断言守卫。
 */
export const CLIP_INSTANCE_OFFSETS = {
    /** 4 floats：x, y, w, h。 */
    rect: 0,
    /** 1 float：圆角半径。 */
    radius: 4,
    /** 1 float：header 高度。 */
    headerH: 5,
    /** 4 floats：body 颜色（RGBA 0..1）。 */
    bodyRgba: 6,
    /** 4 floats：header 颜色。 */
    headerRgba: 10,
    /** 4 floats：描边颜色。 */
    borderRgba: 14,
    /** 1 float：描边宽度。 */
    borderWidth: 18,
    /** 1 float：前导重叠区宽度。 */
    overlapPx: 19,
    /** 1 float：分隔缝宽度（0 = 无）。 */
    seam: 20,
    /** 3 floats：分隔缝 RGB。 */
    seamRgb: 21,
    /** 1 float：实例模式。 */
    mode: 24,
} as const;

/** 实例模式：圆角盒（clip 块面）。 */
export const INSTANCE_MODE_BOX = 0;

/** 实例模式：平面矩形（网格线 / 分界线）。 */
export const INSTANCE_MODE_FLAT = 1;

/**
 * 把一个平面矩形实例写入 GL 实例缓冲（FLAT 模式）。
 *
 * 流程：写入矩形四元组 → 圆角 / header 高度归零 → 四个颜色槽只填 border 槽
 * （FLAT 模式的片元着色器只读该槽）→ 其余槽位清零 → 写入模式。
 *
 * 特殊说明：**其余槽位必须显式清零**——缓冲跨帧复用，残留的上一帧数据会被
 * 着色器读到（例如残留的 header 高度会让矩形出现一条分隔线）。
 *
 * @param out 目标缓冲（容量需 >= `(index + 1) × CLIP_INSTANCE_FLOATS`）。
 * @param index 实例序号。
 * @param instance 平面矩形实例（内容坐标 + RGBA）。
 */
export function writeFlatInstance(out: Float32Array, index: number, instance: FlatInstance): void {
    const base = index * CLIP_INSTANCE_FLOATS;
    const o = CLIP_INSTANCE_OFFSETS;
    out[base + o.rect] = instance.x;
    out[base + o.rect + 1] = instance.y;
    out[base + o.rect + 2] = instance.w;
    out[base + o.rect + 3] = instance.h;
    out[base + o.radius] = 0;
    out[base + o.headerH] = 0;
    for (let i = 0; i < 4; i += 1) {
        out[base + o.bodyRgba + i] = 0;
        out[base + o.headerRgba + i] = 0;
    }
    out[base + o.borderRgba] = instance.rgba[0];
    out[base + o.borderRgba + 1] = instance.rgba[1];
    out[base + o.borderRgba + 2] = instance.rgba[2];
    out[base + o.borderRgba + 3] = instance.rgba[3];
    out[base + o.borderWidth] = 0;
    out[base + o.overlapPx] = 0;
    out[base + o.seam] = 0;
    out[base + o.seamRgb] = 0;
    out[base + o.seamRgb + 1] = 0;
    out[base + o.seamRgb + 2] = 0;
    out[base + o.mode] = INSTANCE_MODE_FLAT;
}
