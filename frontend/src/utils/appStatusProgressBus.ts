/*
 * 状态栏高频进度片的外部 store（变长拉伸 / 波形分析 / 播放渲染进度）。
 *
 * 【为什么存在】这三路进度只服务状态栏的三个指示片，却曾以 React state 住在
 * `AppInner` 里 —— 于是每个 progress 事件（波形分析可达每秒多次）都会重渲染
 * 整棵应用树（时间轴、参数编辑器、全部面板）。搬到外部 store 后，事件只写
 * 这里；显示侧由 `AppStatusProgressChips` 自行订阅（见其文件头说明）。
 *
 * 【发布语义】所有 setter 做浅等值比较：内容不变不发布、不换快照引用
 * （`useSyncExternalStore` 要求快照身份稳定）。进度百分比本身高频变化，
 * 订阅方就是唯一显示它的那一小块组件，重渲染被限制在那一片上。
 */

export interface StretchingState {
    active: boolean;
    clipName: string | null;
}

export interface WaveformAnalysisState {
    active: boolean;
    sourcePath: string | null;
    progress: number | null;
}

interface StatusProgressState {
    stretching: StretchingState;
    waveformAnalysis: WaveformAnalysisState;
    renderingProgress: number | null;
}

let state: StatusProgressState = {
    stretching: { active: false, clipName: null },
    waveformAnalysis: { active: false, sourcePath: null, progress: null },
    renderingProgress: null,
};

const listeners = new Set<() => void>();

function emit(): void {
    for (const listener of listeners) listener();
}

export const appStatusProgressBus = {
    subscribe(listener: () => void): () => void {
        listeners.add(listener);
        return () => {
            listeners.delete(listener);
        };
    },

    getSnapshot(): StatusProgressState {
        return state;
    },

    setStretching(next: StretchingState): void {
        const prev = state.stretching;
        if (prev.active === next.active && prev.clipName === next.clipName) return;
        state = { ...state, stretching: next };
        emit();
    },

    setWaveformAnalysis(next: WaveformAnalysisState): void {
        const prev = state.waveformAnalysis;
        if (
            prev.active === next.active &&
            prev.sourcePath === next.sourcePath &&
            prev.progress === next.progress
        ) {
            return;
        }
        state = { ...state, waveformAnalysis: next };
        emit();
    },

    setRenderingProgress(progress: number | null): void {
        if (state.renderingProgress === progress) return;
        state = { ...state, renderingProgress: progress };
        emit();
    },
};
