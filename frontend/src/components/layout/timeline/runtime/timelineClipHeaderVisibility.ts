export type TimelineClipHeaderContentKind = "audio" | "pitch_reference" | "midi_take";

export function resolveTimelineClipHeaderVisibility(
    widthPx: number,
    contentKind: TimelineClipHeaderContentKind | boolean = "audio",
): {
    showAny: boolean;
    showChain: boolean;
    showMute: boolean;
    showFormant: boolean;
    showGainKnob: boolean;
    showPlaybackRate: boolean;
    showGainLabel: boolean;
    showName: boolean;
} {
    const width = Math.max(0, widthPx);
    // boolean 是旧调用形态的兼容别名：true = 旧音高参考语义。
    const kind: TimelineClipHeaderContentKind =
        contentKind === true ? "pitch_reference" : contentKind === false ? "audio" : contentKind;

    if (kind === "pitch_reference") {
        return {
            showAny: width >= 52,
            showChain: width >= 52,
            showMute: width >= 52,
            showFormant: false,
            showGainKnob: false,
            showGainLabel: width >= 96,
            showPlaybackRate: width >= 116,
            showName: width >= 152,
        };
    }

    // MIDI Take 是**普通内容片段**：增益/静音/链/速率全部沿用音频阈值，
    // 只隐藏共振峰（它只对音频共振峰分析有意义）。音高参考仍走上面的
    // 精简布局：它的增益不参与音高覆盖，显示旋钮会制造“调了没反应”的假象。
    return {
        showAny: width >= 32,
        showChain: width >= 36,
        showMute: width >= 52,
        showFormant: kind === "audio" && width >= 68,
        showGainKnob: width >= 32,
        showGainLabel: width >= 96,
        showPlaybackRate: width >= 116,
        showName: width >= 152,
    };
}
