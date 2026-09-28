/**
 * 排版角色原语。
 *
 * 【它解决的问题】全应用此前用 Radix 的 `Text size="1"|"2"` 直接表达字号，
 * 于是"标题/节标题/正文/标签/提示"这五种角色被映射到两三个字号上，
 * 层级散架：对话框标题一度比正文标签还小，节标题比它统领的行还小。
 *
 * 本模块把角色固定下来，调用方只说"这是标题"，不说"这是 20px"。
 * 角色定义（字号 + 字重 + 行高 + 颜色）见 `src/index.css` 的「排版角色层」。
 *
 * 【为什么不用 Radix 的 Text】Radix 的 `size` 是**字号**词汇表（1–9），
 * 不是**角色**词汇表；沿用它就等于把"该选哪个字号"这个决定重新交回给作者，
 * 而本轮的全部教训正是"这个决定不该由作者做"。
 */
import type { ElementType, ReactNode } from "react";

import { cx } from "./cx";

/**
 * 排版角色。
 *
 * - `display` 对话框标题
 * - `section` 表单分区标题
 * - `body` 正文、复选框行标签、控件值
 * - `label` 字段标签
 * - `caption` 说明、提示、单位、状态
 * - `mono` 数值读数（等宽数字）
 */
export type AppTextRole = "display" | "section" | "body" | "label" | "caption" | "mono";

export interface AppTextProps {
    role?: AppTextRole;
    /** 渲染成什么元素。默认 `span`；段落用 `p`，节标题用 `h3`。 */
    as?: ElementType;
    children: ReactNode;
    className?: string;
    /** 关联的表单控件 id（`as="label"` 时使用）。 */
    htmlFor?: string;
    title?: string;
}

/**
 * 按角色渲染文本。
 *
 * @example
 * <AppText role="section">网格</AppText>
 * <AppText role="caption">px</AppText>
 * <AppText as="p" role="body">{description}</AppText>
 */
export function AppText({
    role = "body",
    as: Tag = "span",
    children,
    className,
    htmlFor,
    title,
}: AppTextProps) {
    return (
        <Tag className={cx(`hs-type-${role}`, className)} htmlFor={htmlFor} title={title}>
            {children}
        </Tag>
    );
}
