/**
 * 按钮原语 —— 全应用按钮外观的**唯一**来源。
 *
 * 【为什么需要它】审查时统计：全仓 133 处 `<Button>`、32 处 `<IconButton>`、
 * 97 处裸 `<button>`。这些调用点大多数其实只在重复同一件事——
 * "灰色的次要按钮"（`size="1" variant="soft" color="gray"`，83 次）、
 * "强调色的确认按钮"（裸 `<Button>`，靠 Radix 默认的 solid+accent）、
 * "开关按钮"（`variant={on ? "solid" : "ghost"}`，约 20 处，散落成三元式）。
 *
 * 重复本身不是问题，**漂移才是**：同一个语义在不同文件里写出不同的
 * variant 组合，于是出现了 `variant="surface"` 只出现 1 次、
 * `size="2"` 只出现 4 次、`AboutDialog` 把次要动作放到 Cancel 左边这类
 * 一次性例外。把语义收进 `intent` 后，调用方不再需要知道 Radix 的
 * variant 词汇表，也就无从写出新的例外。
 *
 * 【为什么不用一个"控件基类"】度量（令牌）、交互协议（组合壳）、视觉状态
 * （本层）三者变更频率相差一个数量级。混进一个基类后，改任一层都会迫使
 * 另外两层跟着动。本层只负责第三件事。
 *
 * 【对扩展 API 的意义】本文件同时是第三方控件的视觉契约：只要用
 * `AppButton`，就自动跟随主题与尺寸体系；若坚持自造按钮，则消费
 * `--qt-*` 令牌同样能继承主题（见 `src/index.css` 的度量令牌块）。
 */
import { Button, IconButton } from "@radix-ui/themes";
import { cloneElement, isValidElement, type ComponentPropsWithoutRef, type ReactNode } from "react";

import { cx } from "./cx";

/** 按钮语义。调用方只选语义，不选 Radix 的 variant/color 组合。 */
export type AppButtonIntent =
    /** 主操作（确认/保存/导出）。一个对话框里最多一个。 */
    | "primary"
    /** 次要操作（取消/关闭/普通动作）。默认值，也是绝大多数场景。 */
    | "default"
    /** 无底色的弱操作，用于不抢视线的行内动作。 */
    | "subtle"
    /** 破坏性操作（删除/丢弃/覆盖）。 */
    | "danger";

/**
 * 尺寸档。
 *
 * 【为什么默认是 `md`（32px）】上一版默认 `sm`（24px），于是全部 42 个对话框的
 * 页脚按钮都从重构前的 32px 被压到 24px —— 用户实测反馈下方的按钮过小。
 * 默认值必须是大多数场景该用的那个，而对话框页脚就是大多数场景。
 *
 * `sm` 留给**行内**动作（列表行里的按钮、紧凑工具条），由调用方显式选择。
 */
export type AppButtonSize = "sm" | "md";

const INTENT_PROPS: Record<AppButtonIntent, Pick<ComponentPropsWithoutRef<typeof Button>, "variant" | "color">> =
    {
        // 裸 Button 即 Radix 默认的 solid + accent —— 与现有确认按钮逐字节一致。
        primary: { variant: "solid" },
        // 与现有 83 处 `variant="soft" color="gray"` 一致。
        default: { variant: "soft", color: "gray" },
        subtle: { variant: "ghost", color: "gray" },
        danger: { variant: "soft", color: "red" },
    };

/**
 * 强调程度。
 *
 * 【为什么与 `intent` 分成两个维度】`intent` 回答"这个按钮做什么"（确认 / 取消 /
 * 删除），`emphasis` 回答"它有多强"。两者独立：破坏性操作既可能是次要的
 * （列表行里的"删除"），也可能是**整个对话框的主操作**（"清空速度图"——
 * 那是用户打开该对话框唯一要做的事）。
 *
 * 上一轮把两者压成了一个 `danger` 语义并固定为浅色，于是所有破坏性确认按钮
 * 都从实心变成了浅色，"清空速度图"这类主操作失去了应有的分量。
 *
 * 省略时按语义取默认：`primary` → 实心，其余 → 浅色。
 */
export type AppButtonEmphasis = "solid" | "soft";

export interface AppButtonProps extends Omit<ComponentPropsWithoutRef<typeof Button>, "variant" | "color" | "size"> {
    intent?: AppButtonIntent;
    size?: AppButtonSize;
    emphasis?: AppButtonEmphasis;
}

/**
 * 应用按钮。
 *
 * @example
 * <AppButton intent="primary" onClick={save}>保存</AppButton>
 * <AppButton onClick={cancel}>取消</AppButton>
 * <AppButton intent="danger" onClick={discard}>丢弃</AppButton>
 */
export function AppButton({
    intent = "default",
    size = "md",
    emphasis,
    className,
    ...rest
}: AppButtonProps) {
    const base = INTENT_PROPS[intent];
    return (
        <Button
            {...base}
            variant={emphasis === undefined ? base.variant : emphasis}
            size={size === "md" ? "2" : "1"}
            className={cx("app-button", className)}
            {...rest}
        />
    );
}

/** 图标尺寸档位，取值来自 `--qt-icon-*` 令牌。 */
export type AppIconSize = "sm" | "md" | "lg";

const ICON_PX: Record<AppIconSize, string> = {
    sm: "var(--qt-icon-sm)",
    md: "var(--qt-icon-md)",
    lg: "var(--qt-icon-lg)",
};

/**
 * 把统一尺寸注入图标节点。
 *
 * 审查发现全仓有 9 种显式图标尺寸（4/6/7/8/9/10/12/14/15）。逐处改调用方
 * 治标不治本——新写的按钮还是会自带尺寸。让**按钮**统一注入，调用方就
 * 无处也不需表达尺寸，这才是机制上的收敛。
 *
 * 非合法元素（字符串、null）原样返回，允许调用方传任意节点。
 */
function withIconSize(icon: ReactNode, size: AppIconSize): ReactNode {
    if (!isValidElement<{ width?: number | string; height?: number | string }>(icon)) return icon;
    const px = ICON_PX[size];
    // 调用方显式传了尺寸则以调用方为准：有些图标（如带角标的复合图）需要特例。
    if (icon.props.width !== undefined || icon.props.height !== undefined) return icon;
    return cloneElement(icon, { width: px, height: px });
}

export interface AppIconButtonProps
    extends Omit<ComponentPropsWithoutRef<typeof IconButton>, "variant" | "color" | "size" | "children"> {
    /** 图标节点。尺寸由本组件统一注入，调用方不要自己写 width/height。 */
    icon: ReactNode;
    /**
     * 悬停提示。**必填**：它同时作为 `aria-label`，是无文字按钮唯一的
     * 可访问名称。原实现里有相当一部分图标按钮漏了 aria-label。
     */
    tooltip: string;
    /** 激活态（可切换按钮的"开"）。激活时用实心，并与未激活的 ghost 同尺寸。 */
    active?: boolean;
    intent?: "default" | "danger";
    size?: AppIconSize;
    /**
     * 强调色。
     *
     * `neutral`（默认）—— 中性灰，用于次要动作与不表达状态的图标按钮。
     * `accent` —— 主题强调色，用于**表达"已启用"的开关**。
     *
     * 【为什么必须由调用方声明】"激活时该不该用强调色"是**语义**（它是否在
     * 表达一种"开着"的状态），上下文推断不出来 —— 与 `AppButton` 的 `intent`
     * 同类，属于**不该由原语写死**的那一维。
     *
     * 【回归修正】上一版把 `color` 写死成 `gray`，而它替换掉的旧写法是
     * `<IconButton variant={on ? "solid" : "ghost"}>`（**不带 color**）——
     * Radix 会回落到主题强调色。于是 18 个开关从强调色变成了灰色
     * （实测 `rgb(91,91,214)` → `rgb(111,109,120)`），而原语没有表达强调色的途径。
     */
    emphasis?: "neutral" | "accent";
}

/**
 * 图标按钮。
 *
 * 用 `size="1"`（24px）而非 Radix 默认的 `2`：主工具栏与对话框按钮都对齐
 * 这个高度。`index.css` 已把 ghost 变体的盒子归一化，因此 `active` 切换
 * solid↔ghost 时不会产生 1px 抖动。
 */
export function AppIconButton({
    icon,
    tooltip,
    active = false,
    intent = "default",
    emphasis = "neutral",
    size = "md",
    className,
    ...rest
}: AppIconButtonProps) {
    /*
     * 强调色 = **不传 `color`**，让 Radix 回落到主题强调色（`--accent-9`）。
     * 这与它替换掉的旧写法逐字节一致，因此那 18 个开关恢复原色。
     */
    const color = intent === "danger" ? "red" : emphasis === "accent" ? undefined : "gray";
    return (
        <IconButton
            variant={active ? "solid" : "ghost"}
            color={color}
            size="1"
            data-tooltip={tooltip}
            aria-label={tooltip}
            aria-pressed={active || undefined}
            className={cx("app-icon-button", className)}
            {...rest}
        >
            {withIconSize(icon, size)}
        </IconButton>
    );
}
