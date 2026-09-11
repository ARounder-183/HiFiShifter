/**
 * 时间轴渲染内核 · 实例数据类型
 *
 * 【主要内容】
 * 定义 CPU 侧构建、GPU 侧消费的「平面矩形实例」：内容坐标下的位置 / 尺寸 + RGBA 颜色。
 * 网格线、轨道分界线等「纯色矩形」元素共用这一结构；clip 块面使用更宽的专用实例布局
 * （见 `scene/clipInstances` 与 sdf-box program 的 BOX 模式）。
 *
 * 【作用】
 * 把「实例数据长什么样」从构建逻辑与 GL 上传逻辑中独立出来，两侧共用同一份类型。
 * GL 实例缓冲是裸 `Float32Array`，字段顺序 / 含义一旦漂移不会报错、只会画错，
 * 因此类型定义必须唯一。
 *
 * 【与其他模块的关系】
 * - 上游：`scene/gridInstances` 等场景构建模块产出实例数组；
 * - 下游：GL 渲染器把实例写入 `Float32Array` 并上传（FLAT 模式）。
 * - 独立性：纯类型定义，无运行时依赖。
 */

/** RGBA 颜色（各分量 0..1）。 */
export type Rgba = readonly [number, number, number, number];

/** 平面矩形实例（内容坐标，CSS px）。 */
export interface FlatInstance {
    /** 左缘 x（内容坐标）。 */
    readonly x: number;
    /** 上缘 y（内容坐标）。 */
    readonly y: number;
    /** 宽度（CSS px，可为分数：线宽按物理像素折算）。 */
    readonly w: number;
    /** 高度（CSS px）。 */
    readonly h: number;
    /** 填充色。 */
    readonly rgba: Rgba;
}
