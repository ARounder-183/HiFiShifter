/**
 * 表单行原语 —— 设置类界面里「标签 + 控件」的唯一排布来源。
 *
 * 【为什么需要它】审查发现设置行的标签宽度有 **12 个不同的魔法值**
 * （40 / 72 / 80 / 96 / 100 / 108 / 110 / 112 / 118 / 120 / 132 / 220），
 * 而且同一个概念被声明成 **3 个各自独立的常量**：
 * `DockLayoutSettingsDialog.LABEL_STYLE = 118`、
 * `NotebookDialogs.SETTING_LABEL_STYLE = 132`、
 * `TimelineDisplaySettingsDialog` 里三处内联 118。
 *
 * 更糟的是这些常量靠"沿用同一套排布约定"维持，代码注释自己都承认：
 * "一致性靠沿用同一套约定，而不是各自调参"。约定不是机制——第 13 个
 * 作者写出 119 时不会有任何东西阻止他。
 *
 * 现在标签宽度只有三档，且**由表单容器统一下发**：同一张表单里的所有行
 * 天然对齐，各行的作者根本不需要（也无法）表达宽度。
 */
import { createContext, useContext, useId, type ReactNode } from "react";
import { Checkbox, Switch } from "@radix-ui/themes";

import { cx } from "./cx";

/** 标签宽度档位。三档对应历史上最常用的 80 / 112 / 132。 */
export type AppFieldLabelWidth = "sm" | "md" | "lg" | "auto";

const LABEL_WIDTH_PX: Record<Exclude<AppFieldLabelWidth, "auto">, string> = {
    sm: "80px",
    md: "112px",
    lg: "132px",
};

const LabelWidthContext = createContext<AppFieldLabelWidth>("md");

/**
 * 布尔行的形态。由 `AppForm` 声明一次，`AppSwitchRow` 读取。
 *
 * 【为什么是表单的属性，不是每一行的属性】`aligned` 让布尔行的**控件**与
 * `AppField` 的控件列左缘对齐；`leading` 让控件贴左、标签紧随。
 *
 * 上一轮把 `AppSwitchRow` 统一成 `leading`，理由是"吸附/网格设置里 17 行是
 * 控件在左"。但那个对话框已整体重写，而另外两个用它的大对话框
 * （`DockLayoutSettingsDialog` 6 行、`NotebookDialogs` 9 行）原本**是**
 * 标签列形态 —— 改完之后这 15 行的开关不再与同一张表单里字段的标签列对齐。
 * 形态应当由"这张表单整体长什么样"决定，因此声明在表单上。
 */
const BooleanRowLayoutContext = createContext<"aligned" | "leading">("leading");

export interface AppFormProps {
    /** 本表单内所有 `AppField` 的标签宽度。默认 `md`（112px）。 */
    labelWidth?: AppFieldLabelWidth;
    /**
     * 布尔行的形态。
     *
     * - `aligned`（默认）—— 控件与 `AppField` 的控件列对齐。混排字段与开关的
     *   设置表单用这个（DockLayoutSettingsDialog / NotebookDialogs）。
     * - `leading` —— 控件贴左、标签紧随。纯布尔列表用这个（吸附/网格设置）。
     */
    booleanRow?: "aligned" | "leading";
    children: ReactNode;
    className?: string;
}

/**
 * 表单容器。只做一件事：把标签宽度下发给所有子行。
 *
 * @example
 * <AppForm labelWidth="lg">
 *   <AppField label={t("bitrate")}><Select .../></AppField>
 *   <AppField label={t("sample_rate")}><Select .../></AppField>
 * </AppForm>
 */
export function AppForm({
    labelWidth = "md",
    booleanRow = "leading",
    children,
    className,
}: AppFormProps) {
    return (
        <LabelWidthContext.Provider value={labelWidth}>
            <BooleanRowLayoutContext.Provider value={booleanRow}>
                <div className={cx("flex flex-col gap-3", className)}>{children}</div>
            </BooleanRowLayoutContext.Provider>
        </LabelWidthContext.Provider>
    );
}

export interface AppFieldProps {
    label: ReactNode;
    children: ReactNode;
    /** 覆盖表单级标签宽度。只在确有需要时使用。 */
    labelWidth?: AppFieldLabelWidth;
    /** 控件下方的补充说明。 */
    hint?: ReactNode;
    /** 控件下方的错误信息。与 `hint` 同时给出时错误优先。 */
    error?: ReactNode;
    /** 关联的控件 id（渲染为 `<label for>`），点击标签可聚焦控件。 */
    htmlFor?: string;
    className?: string;
}

/**
 * 设置行。
 *
 * 标签在左（定宽），控件填满余下空间；提示/错误挂在控件正下方，
 * 因此错误信息不会把整行的标签列挤歪。
 */
export function AppField({
    label,
    children,
    labelWidth,
    hint,
    error,
    htmlFor,
    className,
}: AppFieldProps) {
    const inherited = useContext(LabelWidthContext);
    const width = labelWidth ?? inherited;

    return (
        <div className={cx("flex items-start gap-2", className)}>
            {/*
             * 标签用 `hs-type-label`（12px、正文色），不是 `text-qt-sm` + muted。
             * 原实现 11px + muted 让标签比它自己的提示（12px）和取值（12px）都小、
             * 还更淡 —— 主次完全反了。现在：label(12) > caption(11) 且同为正文色系。
             */}
            <label
                className="hs-type-label shrink-0 pt-0.5"
                htmlFor={htmlFor}
                style={width === "auto" ? undefined : { minWidth: LABEL_WIDTH_PX[width] }}
            >
                {label}
            </label>
            <div className="app-field__control flex min-w-0 flex-1 flex-col gap-1">
                {children}
                {error ? (
                    // 错误用 caption 的尺寸但换成危险色（保留语义区分）
                    <span className="hs-type-caption" style={{ color: "var(--qt-danger-text)" }}>
                        {error}
                    </span>
                ) : hint ? (
                    <span className="hs-type-caption">{hint}</span>
                ) : null}
            </div>
        </div>
    );
}

export interface AppSwitchRowProps {
    /**
     * 可见标签。仅在控件**没有相邻可见标签**时才省略（如表格 / 矩阵里
     * 与整行内容并列的复选框），此时改用 `ariaLabel` 提供无障碍名称。
     */
    label?: ReactNode;
    /**
     * 控件的无障碍名称（`aria-label`）。`label` 省略时必填 ——
     * 无可见标签的控件不能没有名称。
     */
    ariaLabel?: string;
    checked: boolean;
    onCheckedChange: (checked: boolean) => void;
    disabled?: boolean;
    hint?: ReactNode;
    /**
     * 用哪种控件。
     *
     * 两者语义不同，不要随意替换：`switch` 表示"立即生效的开关"，
     * `checkbox` 表示"提交时一并生效的选项"。这里提供同一个外壳是因为
     * **排布与排版必须一致** —— 此前两者是两套手写行，标签一个 11px 一个 14px。
     */
    control?: "switch" | "checkbox";
    className?: string;
}

/**
 * 布尔行 —— 全应用**唯一**的布尔设置行形态：控件在左、标签紧随其后。
 *
 * 【为什么统一成"控件在左"而不是"标签列 + 控件"】
 * 布尔行是列表性质的（一屏十几行），标签全部左对齐在同一列才扫得动。
 * 若沿用 `AppField` 的定宽标签列，复选框与自己的标签之间会隔出 112px 空白，
 * 简单布尔列表读起来很别扭。原先 `吸附/网格设置` 里 17 行是"控件在左"、
 * 8 行是"标签列 + 控件"，两种混排正是"字号混杂"的现场。
 *
 * 【为什么不再有 `layout` 开关】上一版给了 `inline` / `between` 两种布局，
 * 于是"该用哪种"又变成作者的决定 —— 与本轮的核心教训相悖。现在只有一种。
 */
export function AppSwitchRow({
    label,
    ariaLabel,
    checked,
    onCheckedChange,
    disabled,
    hint,
    control = "switch",
    className,
}: AppSwitchRowProps) {
    const Control = control === "checkbox" ? Checkbox : Switch;
    const layout = useContext(BooleanRowLayoutContext);
    const inheritedLabelWidth = useContext(LabelWidthContext);

    /*
     * 标签必须是**真正的 `<label>`**，并与控件用 `id`/`htmlFor` 关联。
     *
     * 【回归修正】上一版标签是裸 `<span>`，而它替换掉的旧写法是
     * `<label><input/><span>文案</span></label>` —— 点标签文本即可切换。改完之后
     * **只有那 16px 的控件可点**，11 处设置行的点击目标凭空缩小（浏览器实测：
     * 点标签文本 `labelClickToggled: false`、`controlId: null`）。
     *
     * 用 `useId` 而不是手写 id：同一表单里可以有几十行，手写必然撞号。
     */
    const controlId = useId();
    const labelClass = cx(
        // 旧标签带 `cursor-pointer select-none`：整行可点，且拖选不会选中文案
        "cursor-pointer select-none",
        disabled && "cursor-default",
    );
    const controlNode = (
        <Control
            id={controlId}
            checked={checked}
            onCheckedChange={(value) => onCheckedChange(Boolean(value))}
            disabled={disabled}
            // 有可见标签时不再给 aria-label（避免可访问名重复）；仅 aria-only 行才用
            aria-label={label == null ? ariaLabel : undefined}
        />
    );

    if (layout === "leading") {
        return (
            <div className={cx("flex items-start gap-2", className)}>
                {/* 与 12px 标签的首行基线对齐（控件高 16–20px，标签行高 18px） */}
                <span className="shrink-0" style={{ marginTop: 1 }}>
                    {controlNode}
                </span>
                {label != null || hint ? (
                    <div className="flex min-w-0 flex-1 flex-col">
                        {label != null ? (
                            <label htmlFor={controlId} className={cx("hs-type-label", labelClass)}>
                                {label}
                            </label>
                        ) : null}
                        {hint ? <span className="hs-type-caption">{hint}</span> : null}
                    </div>
                ) : null}
            </div>
        );
    }

    // aligned：标签占 `AppField` 那一列，控件落在同一列起点 —— 与同表单的字段行对齐。
    return (
        <div className={cx("flex items-start gap-2", className)}>
            {label != null ? (
                <label
                    htmlFor={controlId}
                    className={cx("hs-type-label shrink-0 pt-0.5", labelClass)}
                    style={
                        inheritedLabelWidth === "auto"
                            ? undefined
                            : { minWidth: LABEL_WIDTH_PX[inheritedLabelWidth] }
                    }
                >
                    {label}
                </label>
            ) : null}
            <div className="flex min-w-0 flex-1 flex-col gap-1">
                {controlNode}
                {hint ? <span className="hs-type-caption">{hint}</span> : null}
            </div>
        </div>
    );
}

export interface AppFormSectionProps {
    /** 分区标题。用 `section` 角色（13px/600）——**不缩字号**，靠字重分层。 */
    title: ReactNode;
    /** 分区说明（可选），挂在标题下方。 */
    description?: ReactNode;
    /** 标题右侧的附加控件（如分区级开关）。 */
    action?: ReactNode;
    children: ReactNode;
    className?: string;
}

/**
 * 表单分区 —— 把长表单切成可扫读的段落。
 *
 * 【为什么需要它】`吸附/网格设置` 的内容有 1175px（视口 492px，2.4 屏），
 * 而分区标题与正文同号、只用 muted 表示，滚动时没有任何"路标"。
 * 本组件同时负责三件事：标题角色、分区留白、以及**分区之间的分隔**
 * —— 分隔由留白承担，调用方不再需要手写 Radix `Separator`
 * （那是全仓第四种分隔线做法，`吸附/网格设置` 里还留着 5 个）。
 */
export function AppFormSection({
    title,
    description,
    action,
    children,
    className,
}: AppFormSectionProps) {
    return (
        <section
            className={cx(
                // 分区之间要明显比行距松：分组靠留白承担（本组件取代了 Radix
                // Separator），若节间距与行距相同，分组就看不出来。
                "flex flex-col gap-3 [&:not(:first-child)]:mt-3",
                className,
            )}
        >
            <div className="flex items-center justify-between gap-2">
                <div className="flex min-w-0 flex-col">
                    <h3 className="hs-type-section m-0">{title}</h3>
                    {description ? <p className="hs-type-caption m-0">{description}</p> : null}
                </div>
                {action ? <div className="shrink-0">{action}</div> : null}
            </div>
            <div className="flex flex-col gap-3">{children}</div>
        </section>
    );
}
