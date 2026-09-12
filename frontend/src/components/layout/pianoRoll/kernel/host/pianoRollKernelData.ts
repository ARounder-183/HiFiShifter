/**
 * 参数编辑器内核 · 宿主数据镜像
 *
 * 【主要内容】
 * 声明宿主每帧需要从 React 侧读取的数据形状：工程时长、当前参数的**值域视口**
 * （`min` / `max` / `span`），以及阶段 2 GL 场景层所需的网格几何输入。
 *
 * 【作用】
 * 宿主是长生命周期的命令式运行时对象，若直接持有 React state 或 Redux store，
 * 数据更新会迫使宿主重建（丢失滚动位置与手势状态）。数据镜像把「回调用什么」
 * 与「React 何时重渲染」解耦：面板每次 render 更新镜像字段，宿主在帧提交时现读。
 *
 * 【为什么值域只有 min/max/span 而没有 center】
 * `center` 的真值是内核的像素 `scrollTop`（唯一事实源）。若镜像里再放一份
 * `center`，就出现两个真值源，滚动条拖拽与面板写入会互相覆盖。这里只提供换算
 * 参数（值域边界与跨度），`center` 一律经 `verticalValueScroll` 双向换算得出。
 * 注意 `grid.view` 里的 `center` 是**另一个东西**：它描述值域视口中心，由面板
 * 在自己的 ref 里持有（与竖向滚动条位置同源），不是内核的像素真值。
 *
 * 【与其他模块的关系】
 * - 上游：`PianoRollPanel` 构造镜像对象并每次 render 更新其字段。
 * - 下游：`pianoRollKernelHost` 在帧提交、值域换算与 GL 几何构建时读取。
 * - 独立性：纯类型 + 无逻辑，不依赖 DOM / React。
 */

/** 值域边界与跨度（与 `verticalValueScroll` 的入参同形）。 */
export interface PianoRollValueDomain {
    /** 该参数值域下界。 */
    readonly min: number;
    /** 该参数值域上界。 */
    readonly max: number;
    /** 视口当前可见的值跨度。 */
    readonly span: number;
}

/**
 * GL 场景层的网格几何输入（阶段 2）。
 *
 * 【为什么把颜色与投影也放进镜像】GL 层要自己算出网格线的**数值 RGBA** 与内容
 * 坐标 y；这两者都由面板的主题与投影函数决定。让宿主现读镜像，就不必在宿主里
 * 复制一份配色表或投影公式——复制必然分叉，而分叉的表现是"两种渲染模式的颜色 /
 * 位置微妙地不一致"，很难归因。
 */
export interface PianoRollGridSpec {
    /**
     * 网格种类：决定用哪套几何。
     *
     * 特殊说明：`"pitch"` 走半音线（每个整数半音一条、pc === 0 换色），其余三种
     * 走值域步进线（步进与强线间隔各不相同）。与 `render.ts` 的四个分支一一对应。
     */
    readonly kind: "pitch" | "cents" | "degrees" | "formantCents";
    /** 当前值域视口（中心与跨度）。 */
    readonly view: { readonly center: number; readonly span: number };
    /** 绝对值域下界（pitch 为 MIDI 下界，其余为参数值下界）。 */
    readonly absMin: number;
    /** 绝对值域上界。 */
    readonly absMax: number;
    /**
     * 值 → 视口 y 的投影。
     *
     * 特殊说明：**必须**与 Canvas2D 路径用的是同一个函数。另写一份投影会让两种
     * 渲染模式的网格错位——而这类错位只差亚像素，肉眼几乎无法判定哪个才是对的。
     */
    readonly valueToY: (value: number, heightPx: number) => number;
    /**
     * 强线颜色（pitch 下即 C 线颜色）。
     *
     * 特殊说明：用**数值 RGBA** 而不是 CSS 字符串：GL 需要 0..1 浮点，而在渲染
     * 热路径上反复解析 CSS 颜色是纯浪费。解析由面板在低频时机完成。
     */
    readonly strongRgba: readonly [number, number, number, number];
    /** 弱线颜色（pitch 下即其余半音的颜色）。 */
    readonly weakRgba: readonly [number, number, number, number];
}

/** 宿主每帧读取的数据镜像。 */
export interface PianoRollKernelData {
    /**
     * 工程总时长（秒），用于算内容宽度与横向滚动上限。
     *
     * 特殊说明：应与面板 `contentWidth` 同源（面板已含 `Math.max(1, …)` 保护），
     * 两边不一致会让滚动条 thumb 比例与实际可滚范围分叉。
     */
    readonly projectSec: number;
    /**
     * 当前选中参数的值域（供竖向「值域 ↔ 像素」换算）。
     *
     * 特殊说明：参数切换 / 值域变化时面板只需更新本字段；竖向滚动位置由内核持有，
     * 不会因为本字段变化而复位（与旧实现一致：切参数后滚动条位置按新值域重算，
     * 由面板调用 `host.setValueCenter` 显式提交）。
     */
    readonly valueDomain: PianoRollValueDomain;
    /**
     * GL 场景层的网格输入；缺省 / null 时 GL 层不画网格。
     *
     * 特殊说明：为 null 是**合法状态**（GL 层未启用，或该参数没有网格），宿主必须
     * 当作"没有网格"处理而不是报错。
     */
    readonly grid?: PianoRollGridSpec | null;
}
