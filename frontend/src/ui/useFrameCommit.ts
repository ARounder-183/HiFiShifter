/**
 * 滚轮调值的帧合并提交。
 *
 * 【为什么能力层必须内建它】`AppSelect` / `AppNumberField` / `AppSlider` 把"滚轮调值"
 * 变成了所有控件的默认能力 —— 但没有问过"这个控件的变更代价是多少"。
 *
 * 代价可以很大：`吸附/网格设置` 的「网格间距」下拉的 `onValueChange` 会 dispatch 一个
 * **同步 Tauri 命令**（`set_project_timeline_settings`），而该命令会全量重建节拍器响点表
 * （最多 200 万项 + 稳定排序），带 Tempo Map 时还会整份克隆时间轴并压一条撤销记录。
 * 同步命令跑在 **UI 线程**上（Tauri 2.10：只有 `async fn` 才进线程池），于是
 * **一次滚轮手势 = 每格一次 UI 线程阻塞**，累计到约 5s 消息泵饥饿，Windows 就判定
 * "未响应"。实测可复现。
 *
 * 旧实现里这个下拉**根本没有 `onWheel`** —— 是能力层把它接上的，也就把这条路径从
 * 不可达变成了默认可达。所以节流不能靠调用方自觉，必须由能力层承担。
 *
 * 【做法】每帧最多提交一次，且手势内**累积步进**：连续 N 格只产生 1 次提交，值一次走
 * 到位。累积值放在 ref 里而**不进 state** —— 显示值始终来自外部真值，因此后端拒绝时
 * 不会出现"界面显示 A、工程实际是 B"的分裂。
 *
 * 与 `ActionBar` 的 BPM 字段同源（`utils/commitOncePerFrame.ts` + 手势累加器），
 * 只是收进能力层后所有控件都自动获得。
 */
import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { createFrameCommitter, type FrameCommitter } from "../utils/commitOncePerFrame";

/**
 * 创建一个每帧最多提交一次的提交器。
 *
 * @param commit 真正的提交回调；其身份变化会被自动跟进（不必写进依赖数组）。
 */
export function useFrameCommitter<T>(commit: (value: T) => void): FrameCommitter<T> {
    /*
     * 最新回调经 ref 转发：调用方每次渲染都会重建闭包（多为内联箭头函数），
     * 直接依赖它会让提交器反复重建、丢掉挂起值。写入发生在 effect 里（不是渲染期），
     * 与 `useNonPassiveWheel` 的处理一致。
     */
    const commitRef = useRef(commit);
    useEffect(() => {
        commitRef.current = commit;
    });

    /*
     * 提交器在 `useLayoutEffect` 里创建 —— **不在渲染期**。
     * React Compiler 的引用规则会拒绝"渲染期把读 ref 的闭包传给函数"
     * （规则无法得知闭包何时执行），因此创建动作必须移出渲染期；
     * `useLayoutEffect` 在首次绘制前完成，用户不可能在此之前触发滚轮。
     */
    const committerRef = useRef<FrameCommitter<T> | null>(null);
    useLayoutEffect(() => {
        const committer = createFrameCommitter<T>((value) => commitRef.current(value));
        committerRef.current = committer;
        // 卸载时丢弃挂起值：此刻提交会打到已卸载的父组件上。
        // （挂起值最多存在一帧，因此不会丢掉用户的最后一格。）
        return () => {
            committer.cancel();
            committerRef.current = null;
        };
    }, []);

    /** 稳定代理：方法在事件/帧回调里执行，因此读 ref 合法。 */
    const [proxy] = useState<FrameCommitter<T>>(() => ({
        schedule: (value: T) => committerRef.current?.schedule(value),
        flush: () => committerRef.current?.flush(),
        cancel: () => committerRef.current?.cancel(),
        isPending: () => committerRef.current?.isPending() ?? false,
    }));

    return proxy;
}

/**
 * 手势内累积步进。
 *
 * 【为什么需要】滚轮每格都会调一次处理函数，而处理函数是从**外部值**算"下一格"的。
 * 若外部值要等一次 IPC 往返才更新，那么同一帧里的 N 格会各自算出**同一个**下一格 ——
 * 值只走一格，用户觉得"滚了没反应"，于是滚得更用力，把阻塞放大。
 *
 * 累积器把"上次请求到的值"记在 ref 里，后续格子从它继续走；一旦外部值追上来
 * （或换成别的值），累积器自动失效 —— 无需定时器，也不会与外部真值长期分叉。
 */
export interface WheelStepAccumulator<T> {
    /** 从外部值 `external` 出发继续走一格后的值（由调用方给出 step 函数）。 */
    advance: (external: T, step: (base: T) => T) => T;
}

export function useWheelStepAccumulator<T>(): WheelStepAccumulator<T> {
    const stateRef = useRef<{ base: T; value: T } | null>(null);
    return {
        advance(external, step) {
            const current = stateRef.current;
            // 外部值变了（追上来了，或被别处改了）⇒ 交还控制权，从外部值重新起步。
            const base = current && Object.is(current.base, external) ? current.value : external;
            const next = step(base);
            stateRef.current = { base: external, value: next };
            return next;
        },
    };
}
