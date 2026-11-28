// 编辑链路埋点的**唯一出口**：把环形缓冲里的低频事件刷进文件日志。
//
// 【为什么需要一次 flush 而不是逐条写】拖动期间 reducer 每帧都跑；直接在 reducer
// 里调 `frontendLog` 会把去重表、策略矩阵、IPC 序列化全部拖进热路径。
// `emitEditTelemetry` 只做一次数组 push，这里每 2 秒取一次。
//
// 【为什么 2 秒】埋点是为了排查「回包被谁吃了 / 手势什么时候开始」这类**秒级**
// 问题，2 秒的粒度足够；更密只是徒增日志体积。
import { useEffect } from "react";

import { drainEditTelemetry, formatEditTelemetryEvent } from "../features/session/editTelemetry";
import { frontendLog } from "../services/frontendLog";

const FLUSH_INTERVAL_MS = 2000;

export function useEditTelemetryFlush(): void {
    useEffect(() => {
        const timer = window.setInterval(() => {
            const { events, dropped } = drainEditTelemetry();
            if (events.length === 0 && dropped === 0) return;
            for (const event of events) {
                frontendLog(event.level, `[edit] ${formatEditTelemetryEvent(event)}`);
            }
            if (dropped > 0) {
                // 缓冲区溢出说明这段时间的事件密度超出了设计预期（例如一次手势
                // 刷出上千条）。这本身是有用的信号：它解释了下一条日志为什么
                // 看不到前因。
                frontendLog("warn", `[edit] telemetry dropped ${dropped} event(s) (buffer full)`);
            }
        }, FLUSH_INTERVAL_MS);
        return () => window.clearInterval(timer);
    }, []);
}
