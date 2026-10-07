/* eslint-disable react-refresh/only-export-components -- 文件同时导出组件与 Hook/常量（刷新边界按文件粒度接受） */
import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
    type ReactNode,
} from "react";
import { coreApi } from "../services/api";
import { HISTORY_JUMP_EVENT } from "../features/session/historyJump";

// ─── 状态类型 ────────────────────────────────────────────────────────────────

export interface PitchAnalysisState {
    /** 是否正在分析 */
    pending: boolean;
    /** 整体进度 0~1，null 表示未知 */
    progress: number | null;
    /** 当前正在分析的 clip 名称 */
    currentClip: string | null;
    /** 已完成的 clip 数量 */
    completedClips: number | null;
    /** 总 clip 数量 */
    totalClips: number | null;
}

const DEFAULT_STATE: PitchAnalysisState = {
    pending: false,
    progress: null,
    currentClip: null,
    completedClips: null,
    totalClips: null,
};

// ─── Context ─────────────────────────────────────────────────────────────────

interface PitchAnalysisContextValue {
    state: PitchAnalysisState;
    setState: (patch: Partial<PitchAnalysisState>) => void;
    reset: () => void;
}

const PitchAnalysisContext = createContext<PitchAnalysisContextValue | null>(null);

// ─── Provider ────────────────────────────────────────────────────────────────

export function PitchAnalysisProvider({ children }: { children: ReactNode }) {
    const [state, setStateRaw] = useState<PitchAnalysisState>(DEFAULT_STATE);

    // 使用 ref 避免 setState 闭包过期问题
    const stateRef = useRef(state);
    stateRef.current = state;

    const setState = useCallback((patch: Partial<PitchAnalysisState>) => {
        setStateRaw((prev) => ({ ...prev, ...patch }));
    }, []);

    const reset = useCallback(() => {
        setStateRaw(DEFAULT_STATE);
    }, []);

    // ── 全局 Tauri 事件监听（不依赖 PianoRoll 面板是否打开）────────────────
    useEffect(() => {
        let disposed = false;
        let unlistenStarted: (() => void) | null = null;
        let unlistenProgress: (() => void) | null = null;
        let unlistenUpdated: (() => void) | null = null;

        /*
         * 历史跳转（撤销 / 重做 / 跳步 / 打开或新建工程）之后，在途的旧批次进度
         * 必须被忽略，直到下一次 `pitch_orig_analysis_started`。
         *
         * 【为什么需要这个标记】撤销会把时间线整体换成另一份快照 —— 被导入的 clip
         * 已经不在时间线上了，但后端在途的分析线程只认工程代次、不认"clip 还在不在"，
         * 于是它仍会陆续发来进度事件。若不忽略，状态栏会在用户撤销之后重新亮起
         * "正在分析音高"并长时间不退（分析对象已不存在）；若只复位不忽略，则下一次
         * 进度事件又会把它点亮。真正的下一批分析必定先发 `started`，因此以它为准。
         */
        let ignoreStaleProgress = false;

        const onHistoryJump = () => {
            ignoreStaleProgress = true;
            setStateRaw(DEFAULT_STATE);
        };
        window.addEventListener(HISTORY_JUMP_EVENT, onHistoryJump);

        async function setup() {
            // ① 先做一次初始查询，防止分析在 Provider 挂载前就已开始
            try {
                const progress = await coreApi.getPitchAnalysisProgress();
                if (!disposed && progress && progress.totalClips && progress.totalClips > 0) {
                    const p = Number(progress.progress);
                    const completed = Number(progress.completedClips ?? 0);
                    const finished =
                        p >= 1 || (progress.totalClips > 0 && completed >= progress.totalClips);
                    // 复位即可，绝不能提前 return：后面的监听器注册
                    // 仍必须执行，否则本会话内再发起的新分析将没有进度事件可听。
                    if (finished) {
                        setStateRaw(DEFAULT_STATE);
                    } else {
                        setStateRaw({
                            pending: true,
                            progress: Number.isFinite(p) ? Math.max(0, Math.min(1, p)) : 0,
                            currentClip: progress.currentClipName ?? null,
                            completedClips: progress.completedClips ?? null,
                            totalClips: progress.totalClips ?? null,
                        });
                    }
                }
            } catch {
                // 非 Tauri 环境忽略
            }

            // ② 注册事件监听
            try {
                const mod = window.__HFS_PLUGIN_BOOTSTRAP__
                    ? await import("../services/hostEvents")
                    : await import("@tauri-apps/api/event");

                // 后端所有事件 payload 均为 camelCase（serde rename_all = "camelCase"）
                type StartedPayload = { rootTrackId?: string; key?: string };
                type ProgressPayload = {
                    rootTrackId?: string;
                    progress?: number;
                    currentClipName?: string | null;
                    completedClips?: number;
                    totalClips?: number;
                };
                type UpdatedPayload = { rootTrackId?: string };

                unlistenStarted = await mod.listen<StartedPayload>(
                    "pitch_orig_analysis_started",
                    (event) => {
                        if (disposed) return;
                        void event.payload;
                        // 新批次开始：此前被忽略的旧批次进度不再有影响。
                        ignoreStaleProgress = false;
                        setStateRaw({
                            pending: true,
                            progress: 0,
                            currentClip: null,
                            completedClips: null,
                            totalClips: null,
                        });
                    },
                );
                if (disposed) {
                    unlistenStarted();
                    unlistenStarted = null;
                    return;
                }

                unlistenProgress = await mod.listen<ProgressPayload>(
                    "pitch_orig_analysis_progress",
                    (event) => {
                        if (disposed) return;
                        // 历史跳转之后的进度属于已消失的时间线：忽略，直到下一批
                        // `started`（见上面 ignoreStaleProgress 的说明）。
                        if (ignoreStaleProgress) return;
                        const payload = event.payload ?? {};
                        const p = Number(payload?.progress);
                        if (!Number.isFinite(p)) return;
                        const pp = Math.max(0, Math.min(1, p));
                        const completed = Number(payload?.completedClips ?? 0);
                        const total = Number(payload?.totalClips ?? 0);
                        const finished = pp >= 1 || (total > 0 && completed >= total);
                        // The final progress event races with pitch_orig_updated
                        // (different backend threads), so treat completion as terminal.
                        if (finished) {
                            setStateRaw(DEFAULT_STATE);
                            return;
                        }
                        setStateRaw({
                            pending: true,
                            progress: pp,
                            // 注意：后端字段名为 camelCase
                            currentClip: payload.currentClipName ?? null,
                            completedClips: payload.completedClips ?? null,
                            totalClips: payload.totalClips ?? null,
                        });
                    },
                );
                if (disposed) {
                    unlistenProgress();
                    unlistenProgress = null;
                    return;
                }

                unlistenUpdated = await mod.listen<UpdatedPayload>(
                    "pitch_orig_updated",
                    (event) => {
                        if (disposed) return;
                        void event;
                        setStateRaw(DEFAULT_STATE);
                    },
                );
                if (disposed) {
                    unlistenUpdated();
                    unlistenUpdated = null;
                    return;
                }
            } catch {
                // Safe no-op：浏览器 / pywebview 构建中没有 Tauri API。
            }
        }

        void setup();

        return () => {
            disposed = true;
            window.removeEventListener(HISTORY_JUMP_EVENT, onHistoryJump);
            unlistenStarted?.();
            unlistenProgress?.();
            unlistenUpdated?.();
        };
    }, []);

    const value = useMemo(() => ({ state, setState, reset }), [state, setState, reset]);

    return <PitchAnalysisContext.Provider value={value}>{children}</PitchAnalysisContext.Provider>;
}

// ─── Hooks ───────────────────────────────────────────────────────────────────

/** 读取 pitch 分析进度状态（用于 UI 展示） */
export function usePitchAnalysis(): PitchAnalysisState {
    const ctx = useContext(PitchAnalysisContext);
    if (!ctx) {
        throw new Error("usePitchAnalysis must be used within PitchAnalysisProvider");
    }
    return ctx.state;
}

/** 写入 pitch 分析进度状态（供外部手动更新，通常由 Provider 内部事件监听自动维护） */
export function usePitchAnalysisDispatch() {
    const ctx = useContext(PitchAnalysisContext);
    if (!ctx) {
        throw new Error("usePitchAnalysisDispatch must be used within PitchAnalysisProvider");
    }
    return { setState: ctx.setState, reset: ctx.reset };
}
