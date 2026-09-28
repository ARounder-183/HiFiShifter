/**
 * 控件密度 —— 由**容器**声明，控件读取。
 *
 * 【为什么需要它】上一轮把尺寸决定权从调用方收走，交给了原语的固定默认值 ——
 * 于是 `AppSelect` 没有 `size`，12 个原本 `size="1"`（24px）的下拉全部涨到 Radix
 * 默认 `size="2"`（32px），其中一些就坐在 22px 高的行里（文件浏览器、快速搜索、
 * 轨道头），视觉上直接溢出行外；`AppNumberField` 则把 13 个原本满宽的字段压成
 * 72px 的小盒子。
 *
 * "把选择交给调用方"与"把选择交给原语的硬编码默认"是同一个错误的两面。正确的
 * 形态是**由上下文决定**：控件不需要问自己该多大，它只需要知道"我在哪"。
 *
 * | 提供者 | 密度 | 效果 |
 * |---|---|---|
 * | `AppDialog` 的正文 | `form` | 控件 32px；数字框默认铺满控件列 |
 * | `AppToolbar` / `PanelToolbar` | `compact` | 控件 24px；数字框默认 72px |
 *
 * 【为什么默认是 `form`】大多数取值控件在对话框表单里；紧凑表面（工具条）是少数
 * 且都有明确的容器可以声明。默认取多数派，少数派显式声明。
 */
import { createContext, useContext } from "react";

export type AppDensity = "form" | "compact";

const DensityContext = createContext<AppDensity>("form");

/** 供容器组件下发密度。 */
export const AppDensityProvider = DensityContext.Provider;

/**
 * 读取当前密度。
 *
 * @param override 显式覆盖 —— 给**不是工具条**的紧凑表面用
 *   （例如快速搜索的排序行、轨道头里的算法下拉）。
 */
export function useDensity(override?: AppDensity): AppDensity {
    const inherited = useContext(DensityContext);
    return override ?? inherited;
}

/** 密度 → Radix `size` 属性（'1' = 24px，'2' = 32px）。 */
export function radixSizeFor(density: AppDensity): "1" | "2" {
    return density === "compact" ? "1" : "2";
}
