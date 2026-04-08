import { createContext, useContext, useState, useMemo, useCallback, type ReactNode } from "react";

export interface PianoRollStatus {
    /** usePianoRollData 的数据加载中 */
    dataLoading: boolean;
    /** asyncRefresh 正在异步刷新 pitch */
    asyncRefreshActive: boolean;
    /** asyncRefresh 进度 0-100 */
    asyncRefreshProgress: number;
    /** asyncRefresh 状态文本 */
    asyncRefreshStatus: string | null;
    /** asyncRefresh 错误 */
    asyncRefreshError: string | null;
}

interface PianoRollStatusContextValue {
    status: PianoRollStatus;
    update: (patch: Partial<PianoRollStatus>) => void;
}

const DEFAULT_STATUS: PianoRollStatus = {
    dataLoading: false,
    asyncRefreshActive: false,
    asyncRefreshProgress: 0,
    asyncRefreshStatus: null,
    asyncRefreshError: null,
};

const PianoRollStatusContext = createContext<PianoRollStatusContextValue | null>(null);

export function PianoRollStatusProvider({ children }: { children: ReactNode }) {
    const [status, setStatus] = useState<PianoRollStatus>(DEFAULT_STATUS);

    const update = useCallback((patch: Partial<PianoRollStatus>) => {
        setStatus((prev) => ({ ...prev, ...patch }));
    }, []);

    const value = useMemo(() => ({ status, update }), [status, update]);

    return (
        <PianoRollStatusContext.Provider value={value}>{children}</PianoRollStatusContext.Provider>
    );
}

/** 读取 PianoRoll 加载状态（用于 status bar） */
export function usePianoRollStatus(): PianoRollStatus {
    const ctx = useContext(PianoRollStatusContext);
    if (!ctx) {
        throw new Error("usePianoRollStatus must be used within PianoRollStatusProvider");
    }
    return ctx.status;
}

/** 更新 PianoRoll 加载状态（由 PianoRollPanel 调用） */
export function usePianoRollStatusUpdate(): (patch: Partial<PianoRollStatus>) => void {
    const ctx = useContext(PianoRollStatusContext);
    if (!ctx) {
        throw new Error("usePianoRollStatusUpdate must be used within PianoRollStatusProvider");
    }
    return ctx.update;
}
