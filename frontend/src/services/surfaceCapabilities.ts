/**
 * 形态能力的**唯一**读取入口。
 *
 * 【为什么需要它】移动端"没有右键、没有滚轮、没有悬停、没有修饰键"这四条，如果不
 * 收成一张表，就必然会在每个交互面各写一次判断 —— 而新增一个交互面时一定会漏掉
 * 一次。漏掉的症状是"这个按钮在手机上点了没反应"，且没有任何报错。
 *
 * 形状与 `hostCapabilities.ts` 逐条对应（`CapabilityId` / `{allowed, reason}` /
 * `capabilityDeniedText`）：能力是"能不能做"，理由是"为什么不能"。缺了理由，界面
 * 只能退化成一句通用文案 —— 插件侧已经踩过一次。
 *
 * 【为什么 id 是字符串而不是枚举】它们跨越 IPC 边界，后端 `platform/mod.rs` 里
 * 是同一批字符串常量。联合类型在这里提供编译期保护，运行期就是那几个字面量。
 */

import type { MessageKey } from "../i18n/messages";
import { translateOutsideReact } from "../i18n/I18nProvider";
import { surfaceProfile, type SurfaceDecision } from "./surfaceProfileStore";

/** 能力 id。与后端 `platform/mod.rs` 的 `CAP_*` 常量逐字一致。 */
export type SurfaceCapabilityId =
    | "input.hover"
    | "input.wheel"
    | "input.secondaryButton"
    | "input.modifierKeys"
    | "input.pressure"
    | "input.multiTouch"
    | "window.multi"
    | "window.resizable"
    | "fs.pathAccess"
    | "export.share"
    | "export.revealInFolder"
    | "audio.background"
    | "audio.focusEvents"
    | "dialog.nativeFilePicker";

/**
 * 后端可能给出的理由键。
 *
 * 【为什么是一份白名单而不是 `string`】理由键要喂给 `translateOutsideReact`，而它
 * 对未知键的处理是"原样返回键名"—— 于是界面上会出现 `surface_reason_no_path_access`
 * 这种字符串。白名单把它降级成 `null`（没有理由），界面回落到通用文案，比露出内部
 * 标识符好得多。
 */
const REASON_KEYS = [
    "surface_reason_touch_input",
    "surface_reason_no_path_access",
    "surface_reason_not_on_this_surface",
] as const satisfies readonly MessageKey[];

type SurfaceReasonKey = (typeof REASON_KEYS)[number];

const REASON_KEY_SET: ReadonlySet<string> = new Set(REASON_KEYS);

function asReasonKey(raw: string | null): SurfaceReasonKey | null {
    return raw !== null && REASON_KEY_SET.has(raw) ? (raw as SurfaceReasonKey) : null;
}

/** 一条能力的判定结果。 */
export interface SurfaceCapabilityDecision {
    allowed: boolean;
    reason: SurfaceReasonKey | null;
}

/**
 * 查询一条能力。
 *
 * 【剖面里没有这一条时按 `allowed: false` 处理】这是**保守**方向：把一个"不知道能不能"
 * 的入口摆出来，用户点下去才发现不可用；不摆出来只是少一个入口。与
 * `hostCapabilities` 里"读不出宿主版本就不猜"同一条取舍。
 */
export function surfaceCapability(id: SurfaceCapabilityId): SurfaceCapabilityDecision {
    const entry: SurfaceDecision | undefined = surfaceProfile().capabilities[id];
    if (!entry) return { allowed: false, reason: null };
    return { allowed: entry.allowed, reason: asReasonKey(entry.reason) };
}

/** 只要布尔时的短入口（等价于 `surfaceCapability(id).allowed`）。 */
export function surfaceAllows(id: SurfaceCapabilityId): boolean {
    return surfaceCapability(id).allowed;
}

/**
 * 不可用时的成句文案；可用或没有理由时返回空串。
 *
 * 【为什么走 `translateOutsideReact`】调用点大多不在组件里（菜单项的 tooltip、
 * 命令式异常、手势仲裁），与 `capabilityReasonText` 同一取舍：无浏览器环境时回落
 * 英文词典。
 */
export function surfaceDeniedText(id: SurfaceCapabilityId): string {
    const decision = surfaceCapability(id);
    if (decision.allowed || decision.reason === null) return "";
    return translateOutsideReact(decision.reason);
}

/**
 * 一批能力**同时**成立时才为真。
 *
 * 【为什么需要这个糖】真实判断几乎都是组合：'能显示右键菜单' = 有右键 **或** 有长按；
 * '能多窗口' = 能多窗口 **且** 形态是桌面。逐个 `&&` 写会让人在某一处漏掉一条。
 */
export function surfaceAllowsAll(...ids: SurfaceCapabilityId[]): boolean {
    return ids.every((id) => surfaceAllows(id));
}

/**
 * 触摸形态下"次级手势"是否存在（长按合成 contextmenu）。
 *
 * 【为什么单独一个谓词】它是 P3 长按层的总开关，被时间轴、钢琴卷帘、文件浏览器、
 * 停靠标签四处共用。写成组合式判断会让"到底什么情况下长按生效"散在四处。
 */
export function hasSecondaryGesture(): boolean {
    return !surfaceAllows("input.secondaryButton") && surfaceProfile().input.multi_touch;
}
