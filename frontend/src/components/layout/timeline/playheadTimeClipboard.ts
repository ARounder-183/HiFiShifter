/**
 * 复制播放光标时间（主/副两段）到剪贴板。
 *
 * 【为什么从 store 现读，而不是由调用方传值】这是一次性动作：值必须在**按下那一刻**
 * 取。让调用方订阅它，等于让菜单所在的整棵子树跟着 30Hz 的播放轮询重渲染 ——
 * 而这份数据只在用户点"复制"时才被看一眼。
 *
 * 【为什么是这一处】标尺右键菜单与读数右键菜单都有这一项，两处各读一次 store
 * 就有机会读到不同的字段（一个用视觉插值、一个用已提交值）。同一个菜单项在两处
 * 复制出不同的时间，是那种没人会报、但一定有人踩的偏差。
 */
import { store } from "../../../app/store";
import { copyTextToClipboard } from "../../../utils/clipboard";
import { formatCursorTime } from "./timeFormat";

export function copyPlayheadTimeToClipboard(): void {
    const session = store.getState().session;
    const text = formatCursorTime(
        session.primaryTimeUnit,
        session.secondaryTimeUnit,
        session.playheadSec,
        {
            bpm: session.bpm,
            beatsPerBar: Math.max(1, Math.round(session.beats || 4)),
            grid: session.grid,
            tempoMap: session.tempoMap,
        },
    ).combined;
    void copyTextToClipboard(text);
}
