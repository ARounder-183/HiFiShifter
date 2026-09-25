/**
 * 试听切换 —— "点一下播放 / 再点一下停止"的**唯一**实现。
 *
 * 【为什么需要收成一个 hook】此前 `FileBrowserPanel` 只有"播放"分支（再点一次
 * 从头重放，还会与在播的旧音源叠加），而 `QuickSearchPopup` 自己手写了三处
 * `stop()` 后再 `play()`。两处语义不同、且都不是用户要的"切换"。同一语义只应
 * 存在一处。
 *
 * 状态仍以 Redux 的 `fileBrowser.previewingFile` 为准（UI 高亮用它），本 hook
 * 负责让它与引擎真实状态一致 —— 包括失败时回滚。
 */
import { useCallback, useEffect, useRef } from "react";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { audioPreview } from "./audioPreview";
import { setPreviewingFile } from "./fileBrowserSlice";

export interface PreviewToggle {
    /** 当前正在试听的文件路径（null = 没有在播）。 */
    previewingFile: string | null;
    /**
     * 点击一个音频文件：
     * - 它正在播 ⇒ 停止；
     * - 否则 ⇒ 停掉当前、播放它。
     */
    toggle(path: string): void;
    /**
     * 显式播放（不做切换）。
     *
     * 用于"选中即试听"的键盘导航：用户移动选择时应当听到新的那一条，而不是因为
     * 目标恰好是刚才那条而被判成"再点一次停止"。
     */
    play(path: string): void;
    /** 显式停止（切换目录、面板关闭等场景）。 */
    stop(): void;
}

export function usePreviewToggle(): PreviewToggle {
    const dispatch = useAppDispatch();
    const previewingFile = useAppSelector((state) => state.fileBrowser.previewingFile);
    /**
     * 播放会话号：每次 play / stop 递增，在飞的异步回滚据此判断自己是否仍是
     * "最新一轮"。
     *
     * `audioPreview.play()` 返回 `false` 有两种含义：取数/解码失败，**或**被
     * 新的 `play()` / `stop()` 抢占（audioPreview.ts 的会话号失效路径）。若不
     * 区分，"先点慢加载的 A 再点 B"时，A 迟到的 `false` 会把属于 B 的高亮
     * 清掉 —— 用户看着高亮消失，声音却在放。
     */
    const playSequenceRef = useRef(0);

    const stop = useCallback(() => {
        // 先作废在飞 play 的回滚资格：stop 自己已清掉高亮，迟到的失败不得再写状态。
        playSequenceRef.current += 1;
        audioPreview.stop();
        dispatch(setPreviewingFile(null));
    }, [dispatch]);

    const play = useCallback(
        (path: string) => {
            const sequence = ++playSequenceRef.current;
            // 先乐观置位，让高亮立刻响应；失败时回滚（见下）。
            dispatch(setPreviewingFile(path));
            void audioPreview
                .play(path, () => {
                    // 自然播放结束：清掉高亮。（引擎保证被抢占的旧音源不会
                    // 触发本回调，见 audioPreview.ts 的 onended 登记。）
                    dispatch(setPreviewingFile(null));
                })
                .then((started) => {
                    // 取数 / 解码失败：退出播放，不提示用户，仅回滚状态 ——
                    // 但只有失败者仍是最新一轮播放时才回滚；被抢占的旧会话
                    // 不许碰新一轮的高亮。
                    if (!started && playSequenceRef.current === sequence) {
                        dispatch(setPreviewingFile(null));
                    }
                });
        },
        [dispatch],
    );

    const toggle = useCallback(
        (path: string) => {
            // 正在播同一个文件 ⇒ 停止（"再点一次停"）。
            if (previewingFile === path) {
                stop();
                return;
            }
            play(path);
        },
        [play, previewingFile, stop],
    );

    /**
     * 卸载即停止。
     *
     * 面板关闭后音频继续播放是明显的错误行为；同时 Redux 里的 `previewingFile`
     * 若不清理，重新打开面板时那一行会一直显示"正在播放"的高亮。把这条不变量放在
     * hook 里，任何使用方都不必各自记得收尾。
     */
    useEffect(() => {
        return () => {
            audioPreview.stop();
            dispatch(setPreviewingFile(null));
        };
    }, [dispatch]);

    return { previewingFile, toggle, play, stop };
}
