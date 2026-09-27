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
import { createContext, useContext, type ReactNode } from "react";
import { Switch, Text } from "@radix-ui/themes";

import { cx } from "./cx";

/** 标签宽度档位。三档对应历史上最常用的 80 / 112 / 132。 */
export type AppFieldLabelWidth = "sm" | "md" | "lg" | "auto";

const LABEL_WIDTH_PX: Record<Exclude<AppFieldLabelWidth, "auto">, string> = {
    sm: "80px",
    md: "112px",
    lg: "132px",
};

const LabelWidthContext = createContext<AppFieldLabelWidth>("md");

export interface AppFormProps {
    /** 本表单内所有 `AppField` 的标签宽度。默认 `md`（112px）。 */
    labelWidth?: AppFieldLabelWidth;
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
export function AppForm({ labelWidth = "md", children, className }: AppFormProps) {
    return (
        <LabelWidthContext.Provider value={labelWidth}>
            <div className={cx("flex flex-col gap-3", className)}>{children}</div>
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
            <label
                className="shrink-0 pt-0.5 text-qt-sm leading-5 text-qt-text-muted"
                htmlFor={htmlFor}
                style={width === "auto" ? undefined : { minWidth: LABEL_WIDTH_PX[width] }}
            >
                {label}
            </label>
            <div className="flex min-w-0 flex-1 flex-col gap-1">
                {children}
                {error ? (
                    // `AppErrorText` 的等价内联形态：避免为一个简单场景引入额外依赖
                    <Text size="1" color="red">
                        {error}
                    </Text>
                ) : hint ? (
                    <Text size="1" color="gray">
                        {hint}
                    </Text>
                ) : null}
            </div>
        </div>
    );
}

export interface AppSwitchRowProps {
    label: ReactNode;
    checked: boolean;
    onCheckedChange: (checked: boolean) => void;
    disabled?: boolean;
    hint?: ReactNode;
    className?: string;
}

/**
 * 开关行：标签在左、`Switch` 在右，两端对齐。
 *
 * 这类行在 `DockLayoutSettingsDialog`、`NotebookDialogs`、`ClipFormantToolWindow`
 * 各有一份实现，三份的间距与对齐都不完全一致。
 */
export function AppSwitchRow({
    label,
    checked,
    onCheckedChange,
    disabled,
    hint,
    className,
}: AppSwitchRowProps) {
    return (
        <div className={cx("flex items-center justify-between gap-3", className)}>
            <div className="flex min-w-0 flex-col">
                <span className="text-qt-sm leading-5 text-qt-text">{label}</span>
                {hint ? (
                    <Text size="1" color="gray">
                        {hint}
                    </Text>
                ) : null}
            </div>
            <Switch checked={checked} onCheckedChange={onCheckedChange} disabled={disabled} />
        </div>
    );
}
