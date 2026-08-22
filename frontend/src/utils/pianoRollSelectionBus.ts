/**
 * PianoRoll 选区信息总线（轻量订阅，供 MenuBar 等外部组件读取）。
 *
 * 目的：让“基准音阶 = 工程音阶”的对话框能判断当前参数编辑器选区范围内
 * 是否受到 Tempo Map 音阶变化的影响，从而在选项文本中提示用户。
 * 数据量极小且更新频率低（选区/参数窗口变化时推送）。
 */

export interface PianoRollSelectionInfo {
    /** 选区起始帧（绝对帧）。 */
    startFrame: number;
    /** 选区帧数。 */
    frameCount: number;
    /** 帧周期（毫秒）。 */
    framePeriodMs: number;
}

let current: PianoRollSelectionInfo | null = null;
const listeners = new Set<() => void>();

export function publishPianoRollSelection(info: PianoRollSelectionInfo | null): void {
    if (info) {
        if (
            current &&
            current.startFrame === info.startFrame &&
            current.frameCount === info.frameCount &&
            current.framePeriodMs === info.framePeriodMs
        ) {
            return;
        }
    } else if (current === null) {
        return;
    }
    current = info;
    for (const listener of listeners) {
        try {
            listener();
        } catch {
            // 忽略订阅者异常
        }
    }
}

export function getPianoRollSelection(): PianoRollSelectionInfo | null {
    return current;
}

export function subscribePianoRollSelection(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}
