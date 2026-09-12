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

    // ── 键盘轴颜色（仅 pitch 参数有意义；其余参数无键盘）────────────────
    //
    // 特殊说明：全部可选。非音高参数不提供这些字段，GL 层据此判定"没有键盘"并
    // 清空几何——否则切到别的参数后键盘会残留在画布上。
    /** 白键底色。 */
    readonly whiteKeyRgba?: readonly [number, number, number, number];
    /** 黑键底色。 */
    readonly blackKeyRgba?: readonly [number, number, number, number];
    /** 黑键右缘渐变的深色端。 */
    readonly blackKeyGradientRgba?: readonly [number, number, number, number];
    /** C 键分隔线（强）。 */
    readonly cSeparatorRgba?: readonly [number, number, number, number];
    /** 其余键分隔线（弱）。 */
    readonly keySeparatorRgba?: readonly [number, number, number, number];
    /** 键盘轴右缘分隔线。 */
    readonly axisBorderRgba?: readonly [number, number, number, number];

    // ── 文字（阶段 2 Task 5）─────────────────────────────────────────
    /**
     * 参数名（用于刻度标签的显示换算与刻度种类判定）。
     *
     * 特殊说明：度数参数的内部值是 degree-step 单位，标签必须经显示换算；缺省时
     * 按内部值直接格式化。GL 侧据此选择刻度种类，**必须**与 Canvas2D 路径同一个值。
     */
    readonly paramName?: string;
    /** 字体族（与 Canvas2D 路径共用同一个，避免两种模式字形不同）。 */
    readonly fontFamily?: string;
    /** 数值轴刻度标签颜色（`colors.tensionLabel`）。 */
    readonly tensionLabelRgba?: readonly [number, number, number, number];
    /** C 音名标签颜色（加粗）。 */
    readonly cLabelRgba?: readonly [number, number, number, number];
    /** 白键音名标签颜色。 */
    readonly whiteKeyLabelRgba?: readonly [number, number, number, number];
    /** 黑键音名标签颜色。 */
    readonly blackKeyLabelRgba?: readonly [number, number, number, number];
    /** 刻度线颜色（`colors.tensionLine`）。 */
    readonly tensionLineRgba?: readonly [number, number, number, number];
}

/**
 * 播放头叠加层的几何输入（阶段 2 Task 6，阶段 3 收窄为只管播放头）。
 *
 * 【为什么播放头要单独一层】播放帧只动播放头，而曲线不变。独立叠加层让
 * "曲线画布保持缓存、只清一块空画布"成为可能——这是阶段 2 消除播放重绘的关键。
 *
 * 【为什么选区块不在这里】选区块属于**曲线之下**的图层（Canvas2D 路径先画选区、
 * 再画曲线），而叠加层在曲线**之上**。把选区放进来会让它盖住曲线，与迁移前的
 * 观感相反。因此选区仍由主画布绘制，见 `render.ts` 的 `skipPlayhead` 说明。
 */
export interface PianoRollOverlaySpec {
    /**
     * 播放头位置（秒）；null 表示不画。
     *
     * 特殊说明：用**插值的视觉值**而不是 Redux 提交值（与面板 `drawRef` 的注释
     * 一致）——提交值滞后会让播放头与标尺错位。
     */
    readonly playheadSec?: number | null;
    /** 播放头颜色；缺省用主题的 `playheadLine`。 */
    readonly playheadRgba?: readonly [number, number, number, number];
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
    /**
     * 动态叠加层输入（选区块 + 播放头）；缺省 / null 时叠加层为空。
     *
     * 特殊说明：这一份**每帧变化**（播放头位置），因此宿主每帧读取；与 `grid`
     * 那种"低频变化"的字段分开，避免把播放头混进需要内容签名的几何里。
     */
    readonly overlay?: PianoRollOverlaySpec | null;
    /**
     * 曲线图层（阶段 3）；缺省 / null 表示没有曲线。
     *
     * 特殊说明：与 `grid`（低频、按签名缓存）不同，本字段**每帧读取并重建几何**，
     * 因为滚动/缩放会改变每个点的视口位置。见 `PianoRollCurveLayer` 的说明。
     */
    readonly curves?: PianoRollCurveLayerList | null;
}

/**
 * 生产者侧的数据镜像（字段可写）。
 *
 * 【为什么需要单独一个类型】`PianoRollKernelData` 的字段全是 `readonly`——那是给
 * **消费方**（宿主）的契约：宿主只读，不得改面板的数据。而面板是**生产方**，要逐
 * 字段更新同一份对象（每帧新建对象会让宿主每帧拿到新引用，破坏"引用稳定"的判定）。
 * 用 `-readonly` 映射类型从同一份声明派生，避免两个类型各写一遍而漂移。
 */
export type MutablePianoRollKernelData = {
    -readonly [K in keyof PianoRollKernelData]: PianoRollKernelData[K];
};

/**
 * 一条待绘制的曲线（阶段 3：曲线图层上 GL）。
 *
 * 【为什么用统一的描述符而不是 7 个专用字段】曲线有 7 类（检测 / 参考线 / 副参数 /
 * 原始 / 编辑 / 选区高亮 / 剪贴板预览），但它们的绘制语义完全相同：**一串采样值 +
 * 时间基准 + 描边样式 + 可选裁剪**。用统一描述符让面板把"取哪条数据、用什么颜色"
 * 留在 React 侧（那里才知道业务语义），宿主只负责"投影 + 建几何 + 画"。
 *
 * 【为什么带上投影所需的原始参数而不只传点】投影要用 `axis`（每帧可能变），而
 * `axis` 由宿主持有。传原始参数让宿主在**绘制时**才投影，避免面板跟着滚动重算。
 */
export interface PianoRollCurveLayer {
    /**
     * 采样值（pitch 为 MIDI，其余为参数内部值）。
     *
     * 特殊说明：调用方应传**已经可见性裁剪**的序列。宿主仍会按 `axis` 再裁一次
     * （见 `projectCurvePoints`），但提前裁剪能显著降低每帧的遍历量——最小缩放下
     * 视口可覆盖 466 秒，未裁剪的长曲线会有上万点。
     */
    readonly values: readonly number[];
    /** 参数名（决定是否施加 pitch 的 +0.5 半音偏移）。 */
    readonly param: string;
    /** 首个采样值对应的帧号。 */
    readonly startFrame: number;
    /** 采样步长（帧）。 */
    readonly stride: number;
    /** 每帧时长（毫秒）。 */
    readonly framePeriodMs: number;
    /** 线宽（CSS px）。可为分数（既有实现用 1.8 / 2 / 2.6 / 3.2 / 3.6）。 */
    readonly lineWidthPx: number;
    /** 颜色（预乘前的直通 RGBA，0..1）。 */
    readonly rgba: readonly [number, number, number, number];
    /**
     * 虚线图案 `[dash, gap]`（CSS px）；缺省表示实线。
     *
     * 特殊说明：必须由面板用与 Canvas2D 路径**同一个** `getFixedDashPattern` 取值
     * （它按 dpr 量化），否则两种模式的虚线疏密会不同。
     */
    readonly dash?: readonly [number, number] | null;
    /**
     * 投影模式，三者时间基准各不相同，混用会让曲线整体平移或长出尖刺：
     * - `"curve"`：走 `drawCurveTimed` 的语义（`startFrame + i × stride`）；
     * - `"clipboard"`：剪贴板预览（从选区起点按**原始帧距**排布）；
     * - `"detected"`：检测曲线（从 `curveStartSec` 按原始帧距排布，
     *   **并跳过 `midi <= 0` 的无声帧**）。
     */
    readonly projection: "curve" | "clipboard" | "detected";
    /** `projection === "clipboard"` 时的选区起点（秒）。 */
    readonly clipStartSec?: number;
    /** `projection === "clipboard"` 时的选区终点（秒）。 */
    readonly clipEndSec?: number;
    /**
     * `projection === "detected"` 时曲线第 0 帧对应的 timeline 绝对时间（秒）。
     *
     * 【为什么检测曲线不复用 `startFrame`】检测曲线来自后端推送的
     * `clipPitchCurves`，它自带**绝对起始秒**（`curveStartSec`），没有帧号概念。
     * 塞进 `startFrame` 需要先乘除帧周期，徒增一次换算与一处口径。
     */
    readonly curveStartSec?: number;
    /** 裁剪区（视口坐标 CSS px）；缺省不裁剪。两个需要裁剪的曲线图层用选区矩形。 */
    readonly clipRect?: { readonly x: number; readonly y: number; readonly w: number; readonly h: number } | null;
    /**
     * 值 → 视口 y 的投影，**必须绑定到本图层自己的参数**。
     *
     * 【为什么每条曲线各带一份】副参数（`secondaryParamViews`）的值域与主参数
     * 完全不同（例如音分 vs 度数 vs 共振峰）。若统一用 `editParam` 的投影，副参数
     * 曲线会被画到错误的高度——而且错得很"合理"（仍在视口内），很难归因。
     * Canvas2D 路径同样是按各自的 `param` 调用 `valueToY(param, …)`。
     */
    readonly valueToY: (value: number) => number;
}

/**
 * 曲线图层集合（阶段 3）。
 *
 * 特殊说明：本字段**每帧可能变化**（滚动/缩放会改变可见段），因此宿主每帧读取并
 * 重建几何——这与 `grid`（低频变化、按签名缓存）不同。软件栅格化下这个代价换来的
 * 收益是 5–16 倍（见 Phase 3 计划的 R7）。
 */
export type PianoRollCurveLayerList = readonly PianoRollCurveLayer[];
