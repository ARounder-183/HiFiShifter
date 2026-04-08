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

        async function setup() {
            // ① 先做一次初始查询，防止分析在 Provider 挂载前就已开始
            try {
                const progress = await coreApi.getPitchAnalysisProgress();
                if (!disposed && progress && progress.totalClips && progress.totalClips > 0) {
                    const p = Number(progress.progress);
                    setStateRaw({
                        pending: true,
                        progress: Number.isFinite(p) ? Math.max(0, Math.min(1, p)) : 0,
                        currentClip: progress.currentClipName ?? null,
                        completedClips: progress.completedClips ?? null,
                        totalClips: progress.totalClips ?? null,
                    });
                }
            } catch {
                // 非 Tauri 环境忽略
            }

            // ② 注册事件监听
            try {
                const mod = await import("@tauri-apps/api/event");

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
                        setStateRaw({
                            pending: true,
                            progress: 0,
                            currentClip: null,
                            completedClips: null,
                            totalClips: null,
                        });
                    },
                );

                unlistenProgress = await mod.listen<ProgressPayload>(
                    "pitch_orig_analysis_progress",
                    (event) => {
                        if (disposed) return;
                        const payload = event.payload ?? {};
                        const p = Number(payload?.progress);
                        if (!Number.isFinite(p)) return;
                        const pp = Math.max(0, Math.min(1, p));
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

                unlistenUpdated = await mod.listen<UpdatedPayload>(
                    "pitch_orig_updated",
                    (event) => {
                        if (disposed) return;
                        void event;
                        setStateRaw(DEFAULT_STATE);
                    },
                );
            } catch {
                // Safe no-op：浏览器 / pywebview 构建中没有 Tauri API。
            }
        }

        void setup();

        return () => {
            disposed = true;
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
