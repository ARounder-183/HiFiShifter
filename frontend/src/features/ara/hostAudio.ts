/**
 * 宿主音频读数（插件模式）的解析与判定。
 *
 * 【为什么单独一个模块】这份读数有两个消费点 —— 时间轴上方的一次性提示条
 * （`HostAudioNotice`）与片段本身的"等待宿主音频"样式。两处必须对**同一个分类名**
 * 得出同一个结论，否则会出现"提示条说挂错了轨道、片段却看起来正常"这种自相矛盾。
 * 判定逻辑集中在这里，两边都从这里取。
 *
 * 【为什么必须校验分类名】后端将来可能加新的分类。旧前端若把不认识的字符串当成
 * 已知状态渲染，就会给出一条**错误的指引**（用户照着改工程，问题更糟）。所以只认
 * 白名单里的分类名，其余一律当作"本次没带该字段"，沿用上一次已知值。
 */
import type { HostAudioPayload, HostAudioState } from "../../types/api";

/** 已知分类名白名单；与后端 `render::extension::HostAudioState::as_str` 一一对应。 */
export const HOST_AUDIO_STATES: ReadonlySet<string> = new Set<HostAudioState>([
    "ready",
    "awaiting_regions",
    "folder_parent_without_regions",
]);

/**
 * 解析后端载荷。返回 `null` 表示"没有可用读数"（字段缺失、分类未知、类型不对），
 * 调用方应沿用上一次已知值而不是清空或猜测。
 */
export function parseHostAudio(raw: unknown): HostAudioPayload | null {
    if (typeof raw !== "object" || raw === null) return null;
    const candidate = raw as { state?: unknown; waiting_clips?: unknown };
    if (typeof candidate.state !== "string" || !HOST_AUDIO_STATES.has(candidate.state)) {
        return null;
    }
    const count = Number(candidate.waiting_clips);
    return {
        state: candidate.state as HostAudioState,
        waiting_clips: Number.isFinite(count) && count > 0 ? Math.floor(count) : 0,
    };
}

/**
 * 是否需要"插件挂在轨道组父轨上"这条提示。
 *
 * 【为什么只有这一种情形提示】`awaiting_regions` 是加载过程中的正常中间态（状态栏
 * 已有的 `等待宿主音频` 读数覆盖了它），为它弹一条横条只会在每次打开工程时打扰用户。
 * 而 folder 父轨是**用法问题**：症状（轨道和片段都看得见、能拖、但没内容）极具
 * 误导性，不给提示用户根本无从判断。
 */
export function needsFolderTrackNotice(status: HostAudioPayload | null): boolean {
    return status?.state === "folder_parent_without_regions";
}

/**
 * 该片段是否"还在等宿主音频"。
 *
 * 【判据沿用既有的 `source_path` 缺失】占位片段没有 `source_path`（见后端
 * `present_host_inventory` 与 `retain_display_waveforms`：未经 ARA 授权不得读 PCM）。
 * 不新增状态字段，避免同一件事维护两处真相。
 *
 * MIDI / 音高参考片段同样没有 `source_path`，但它有音符内容可画，不该被标成"等待音频"。
 */
export function isAwaitingHostAudio(clip: {
    sourcePath?: string | null;
    midiNoteCount?: number | null;
}): boolean {
    return !clip.sourcePath && clip.midiNoteCount == null;
}
