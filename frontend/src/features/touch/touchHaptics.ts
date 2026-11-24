// 触觉反馈的唯一出口。
//
// 【为什么单独成文件】`navigator.vibrate` 在桌面浏览器里不存在、在 iOS Safari
// 里存在但静默失败、在 Android WebView 里需要 VIBRATE 权限。三处差异如果散在
// 调用点，就会变成"某个机型上点了没反应"这种无法复现的缺陷。
//
// 约定：
// 1. 能力缺失 / 调用抛错一律**静默**（触觉是增强，不是功能）；
// 2. 只有**状态改变**才震（长按弹菜单、分割成功、吸附命中），不做逐帧震动；
// 3. 用户关闭「触觉反馈」时零调用（连能力探测都不做）。

/** 触觉事件的种类。 */
export type HapticKind = "long-press" | "split" | "snap" | "boundary";

/** 各事件的震动模式（毫秒）。 */
const PATTERNS: Record<HapticKind, number | number[]> = {
    // 长按弹菜单：一次短促的"确认"，与 iOS 的 peek 同量级。
    "long-press": 12,
    // 分割成功：两下，第二下更轻 —— 与"切开了"的语义匹配。
    split: [10, 30, 6],
    // 吸附命中：极短，几乎只有"嗒"一下的感觉。
    snap: 6,
    // 边界到头：一次略长的回弹提示。
    boundary: 18,
};

let enabled = true;

/** 开关（由设置项驱动；缺省开启）。 */
export function setHapticsEnabled(next: boolean): void {
    enabled = next;
}

/** 当前是否启用（测试用）。 */
export function hapticsEnabled(): boolean {
    return enabled;
}

/**
 * 触发一次触觉反馈。
 *
 * 任何异常都被吞掉：`navigator.vibrate` 在部分 WebView 里会抛
 * `NotAllowedError`（没有用户手势上下文），而它绝不是"该让编辑失败"的理由。
 */
export function haptic(kind: HapticKind): void {
    if (!enabled) return;
    try {
        // `navigator.vibrate` 的类型在不同 TS lib 版本里不一致（有的声明成
        // `Iterable<number>`）。这里只关心「有没有这个能力」，参数形状按规范走，
        // 因此显式收窄成规范签名再调用。
        const vibrate = (
            navigator as unknown as {
                vibrate?: (pattern: number | number[]) => boolean;
            }
        ).vibrate;
        if (typeof vibrate !== "function") return;
        vibrate.call(navigator, PATTERNS[kind]);
    } catch {
        // 见上：触觉是增强，失败即静默。
    }
}

/** 仅测试用。 */
export function resetHapticsForTests(): void {
    enabled = true;
}
