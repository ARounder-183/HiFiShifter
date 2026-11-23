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
 * | 触摸输入（任意容器） | `touch` | 控件 44px；数字框铺满（见下） |
 *
 * 【为什么默认是 `form`】大多数取值控件在对话框表单里；紧凑表面（工具条）是少数
 * 且都有明确的容器可以声明。默认取多数派，少数派显式声明。
 *
 * 【为什么 `touch` 是"抬高一切"而不是"第四种密度"】44px 在触摸上是**下限**，不是
 * 一种风格：一个 24px 的按钮在手指下点不中，而"这个面板比较紧凑所以可以用 24px"
 * 在触摸上不成立 —— 空间不够应该改布局（折叠、换行、抽屉），不是把目标缩小。
 * 因此触摸形态下 `compact` 与 `form` **都**解析成 `touch`。
 */
import { createContext, useContext } from "react";

import { surfaceProfile } from "../services/surfaceProfileStore";

export type AppDensity = "form" | "compact" | "touch";

const DensityContext = createContext<AppDensity>("form");

/** 供容器组件下发密度。 */
export const AppDensityProvider = DensityContext.Provider;

/**
 * 当前输入通道是不是触摸。
 *
 * 【为什么读剖面而不是读 `<html data-input>`】两者同源（都由 `mount.tsx` 写入），
 * 但剖面在**模块加载期**就已就位，而 DOM 属性要等挂载后才写。密度在首帧就要用，
 * 读剖面才拿得到。jsdom 测试里剖面是桌面后备值，因此既有测试的行为不变。
 */
function isTouchInput(): boolean {
    return surfaceProfile().input.primary === "touch";
}

/**
 * 读取当前密度。
 *
 * @param override 显式覆盖 —— 给**不是工具条**的紧凑表面用
 *   （例如快速搜索的排序行、轨道头里的算法下拉）。
 */
export function useDensity(override?: AppDensity): AppDensity {
    const inherited = useContext(DensityContext);
    const declared = override ?? inherited;
    // 触摸形态把 `compact` / `form` 一起抬到 `touch`（见文件头的说明）。
    if (isTouchInput()) return "touch";
    return declared;
}

/** 密度 → Radix `size` 属性（'1' = 24px，'2' = 32px，'3' = 40px）。 */
export function radixSizeFor(density: AppDensity): "1" | "2" | "3" {
    if (density === "touch") return "3";
    return density === "compact" ? "1" : "2";
}
