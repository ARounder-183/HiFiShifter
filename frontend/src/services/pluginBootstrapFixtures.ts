// 测试用 `hostCapabilities` 夹具。
//
// 【为什么需要它】写能力（片段写 / 分割 / 剪贴板 / 导入 / 建轨 / 撤销栈）只有
// `hostCapabilities` 一个来源，而它是一个**字段齐全**的结构。测试里逐个手写九个字段
// 既啰嗦又容易漏，于是集中成两个基线工厂：`araOnlyCaps`（只读宿主）与
// `writableHostCaps`（有完整原生写能力的宿主）。
import type { PluginBootstrap } from "./pluginHost";

export type HostCaps = NonNullable<PluginBootstrap["hostCapabilities"]>;

/** ARA-only 基线：能看 / 能听 / 能渲染，写能力全否。 */
export function araOnlyCaps(overrides: Partial<HostCaps> = {}): HostCaps {
    return {
        transport: true,
        clipGeometry: true,
        clipWrite: false,
        split: false,
        undoStack: false,
        mediaImport: false,
        trackCreate: false,
        itemClipboard: false,
        fadeAxes: false,
        ...overrides,
    };
}

/** 有完整原生写能力的宿主基线（写片段 + 撤销栈同时成立才允许拖拽）。 */
export function writableHostCaps(overrides: Partial<HostCaps> = {}): HostCaps {
    return araOnlyCaps({ clipWrite: true, undoStack: true, ...overrides });
}
