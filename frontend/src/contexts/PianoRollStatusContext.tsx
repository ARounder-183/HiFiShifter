/* eslint-disable react-refresh/only-export-components -- 文件同时导出组件与 Hook/常量（刷新边界按文件粒度接受） */
import { createContext, useContext, useState, useMemo, useCallback, type ReactNode } from "react";

export interface PianoRollStatus {
    /** usePianoRollData 的数据加载中 */
    dataLoading: boolean;
}

interface PianoRollStatusContextValue {
    status: PianoRollStatus;
    update: (patch: Partial<PianoRollStatus>) => void;
}

const DEFAULT_STATUS: PianoRollStatus = {
    dataLoading: false,
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
