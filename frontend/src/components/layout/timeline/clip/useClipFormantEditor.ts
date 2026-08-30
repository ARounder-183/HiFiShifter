import React from "react";
import type { ClipFormantMorph } from "../../../../features/session/sessionTypes";

const DEFAULT_FORMANT_MORPH: ClipFormantMorph = {
    enabled: false,
    targetF1Hz: 800,
    targetF2Hz: 1400,
    strength: 0.5,
};

export function debounceMs(): number {
    return 180;
}

export function useClipFormantEditor(params: {
    clipId: string;
    value: ClipFormantMorph | undefined;
    onCommit: (clipId: string, value: ClipFormantMorph, checkpoint: boolean) => void;
}) {
    const { clipId, value, onCommit } = params;
    const [draft, setDraft] = React.useState<ClipFormantMorph>(value ?? DEFAULT_FORMANT_MORPH);
    const timerRef = React.useRef<number | null>(null);
    const draftRef = React.useRef<ClipFormantMorph>(value ?? DEFAULT_FORMANT_MORPH);
    const clipIdRef = React.useRef(clipId);
    /** 本次窗口会话是否真的改过（防抖预览已发或本地草稿已动）。 */
    const dirtyRef = React.useRef(false);

    const commit = React.useCallback(
        (targetClipId: string, next: ClipFormantMorph, checkpoint: boolean) => {
            onCommit(targetClipId, next, checkpoint);
        },
        [onCommit],
    );

    const flush = React.useCallback(() => {
        if (timerRef.current !== null) {
            window.clearTimeout(timerRef.current);
            timerRef.current = null;
        }
        // 无操作守卫（B7）：打开窗口未做任何修改就关闭 / 切换 clip 时不落盘
        // —— 否则每次"打开即关"都会产生一个撤销后毫无变化的死撤销步。
        // 用会话脏标记而非"值是否相等"判定：防抖预览的响应回灌可能已让受控
        // value 追上 draft，但最终 checkpoint:true 提交仍然必要（撤销权柄）。
        if (!dirtyRef.current) return;
        commit(clipIdRef.current, draftRef.current, true);
    }, [commit]);

    React.useEffect(() => {
        if (clipIdRef.current !== clipId) {
            flush();
            clipIdRef.current = clipId;
            dirtyRef.current = false; // 新 clip = 新会话
        }

        const nextDraft = value ?? DEFAULT_FORMANT_MORPH;
        draftRef.current = nextDraft;
        setDraft(nextDraft);
    }, [clipId, value, flush]);

    React.useEffect(() => {
        draftRef.current = draft;
    }, [draft]);

    React.useEffect(
        () => () => {
            if (timerRef.current !== null) {
                window.clearTimeout(timerRef.current);
            }
        },
        [],
    );

    const updateDraft = React.useCallback(
        (patch: Partial<ClipFormantMorph>) => {
            setDraft((prev) => {
                const next = { ...prev, ...patch };
                draftRef.current = next;
                dirtyRef.current = true; // 会话已修改：关闭时必须以 checkpoint:true 落盘
                if (timerRef.current !== null) {
                    window.clearTimeout(timerRef.current);
                }
                timerRef.current = window.setTimeout(() => {
                    commit(clipIdRef.current, draftRef.current, false);
                    timerRef.current = null;
                }, debounceMs());
                return next;
            });
        },
        [commit],
    );

    return { draft, updateDraft, flush };
}
