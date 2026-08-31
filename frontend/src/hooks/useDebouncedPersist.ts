/**
 * 防抖持久化 hook：把高频变化的值延迟写入 `localStorage`。
 *
 * 【主要内容】值变化后延迟 `delayMs` 落盘；期间若值再次变化则重置计时，
 * 只在变更停止后写一次。组件卸载时若仍有未落盘的值，会立即补写。
 *
 * 【作用】`localStorage.setItem` 是**同步**且落盘的 I/O。把它放在依赖
 * `pxPerSec` / `rowHeight` 的 effect 里，会让滚轮缩放的**每一帧**都触发一次
 * 同步磁盘写入，与同帧的 React 渲染、画布重绘挤在一起，表现为明显卡顿。
 * 防抖后，一次连续缩放手势只写一次。
 *
 * 【与其他模块的关系】
 * - 当前使用者：`components/layout/timeline/TimelineScrollArea`（持久化
 *   时间线的 `pxPerSec` 与 `rowHeight`）。
 * - 读取侧不受影响：`useTimelineState` 在初始化时读这两个键，防抖只推迟
 *   写入时机，不改变键名与取值语义。
 */

import { useEffect, useRef } from "react";

/**
 * 默认防抖延迟（毫秒）。
 *
 * 取值依据：滚轮/触控板连续缩放的事件间隔通常在 16~50 ms；300 ms 足以把
 * 一整段手势合并成一次写入，同时对用户而言"松手即已保存"（且卸载时会强制
 * 补写，见 hook 实现）。
 */
export const PERSIST_DEBOUNCE_MS = 300;

/** 可持久化的值类型；非字符串统一走 `String()` 转换。 */
export type PersistableValue = string | number | boolean;

/**
 * 执行一次落盘。
 *
 * `localStorage` 在隐私模式或配额耗尽时会抛异常；持久化失败不应影响交互，
 * 因此这里静默忽略。
 */
function writePersist(key: string, value: PersistableValue): void {
    try {
        localStorage.setItem(key, String(value));
    } catch {
        // 忽略：持久化失败不应打断交互。
    }
}

/**
 * 把 `value` 防抖写入 `localStorage` 的 `key`。
 *
 * 流程：
 * 1. 值变化 → 记录待写值，起一个 `delayMs` 的定时器；
 * 2. 期间值再变 → effect 清理函数取消旧定时器，用新值重新计时；
 * 3. 定时器触发 → 落盘，并把"待写值"置空；
 * 4. 卸载 → 若"待写值"仍非空（说明最后一次变更还没落盘），立即补写一次。
 *
 * 特殊说明：
 * - "待写值"置空发生在落盘**之前**，因此卸载补写能靠它准确判断有无未落盘
 *   的值，不会重复写。
 * - 值通过 effect 闭包捕获而非 ref：防抖本就是"取消旧定时器 + 重建新的"，
 *   依赖值重建定时器是应有之义；同时遵守了 React 的规则（不在渲染期读写
 *   ref）。
 *
 * @param key localStorage 的键。
 * @param value 要持久化的值。
 * @param delayMs 防抖延迟（毫秒），默认 `PERSIST_DEBOUNCE_MS`。
 */
export function useDebouncedPersist(
    key: string,
    value: PersistableValue,
    delayMs: number = PERSIST_DEBOUNCE_MS,
): void {
    /** 尚未落盘的值；`null` 表示没有待写内容。 */
    const pendingRef = useRef<PersistableValue | null>(null);

    useEffect(() => {
        pendingRef.current = value;
        const timer = setTimeout(() => {
            pendingRef.current = null;
            writePersist(key, value);
        }, delayMs);

        return () => clearTimeout(timer);
    }, [key, value, delayMs]);

    // 卸载补写：避免"缩放后马上关窗口"丢失最后一次设置。
    useEffect(
        () => () => {
            const pending = pendingRef.current;
            if (pending !== null) {
                pendingRef.current = null;
                writePersist(key, pending);
            }
        },
        [key],
    );
}
