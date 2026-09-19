import type { MessageKey } from "../i18n/messages";
/**
 * Take 声道模式的共享工具（与后端 audio/channel_mode.rs 语义一一对应）。
 *
 * 数值对齐 REAPER `CHANMODE`：
 * 0=正常（按源） / 1=交换左右 / 2=混合为单声道 / 3=仅左 / 4=仅右。
 */

/** Take 声道模式（原始 i32，0..=4） */
export type TakeChannelModeRaw = 0 | 1 | 2 | 3 | 4;

export const TAKE_CHANNEL_MODE_NORMAL = 0;
export const TAKE_CHANNEL_MODE_SWAP = 1;
export const TAKE_CHANNEL_MODE_MONO_MIX = 2;
export const TAKE_CHANNEL_MODE_MONO_LEFT = 3;
export const TAKE_CHANNEL_MODE_MONO_RIGHT = 4;

/** 规范化任意原始值到 0..=4（未知值回落 0=正常）。 */
export function normalizeChannelMode(value: number | null | undefined): TakeChannelModeRaw {
    const v = Number(value);
    if (Number.isInteger(v) && v >= 0 && v <= 4) return v as TakeChannelModeRaw;
    return 0;
}

/**
 * 有效声道数：模式折叠为单声道时为 1；否则取源声道数（>2 视为 2，未知按 1）。
 *
 * 与后端 `channel_mode::effective_channels` 一致 —— 波形带数、徽章与
 * 导出通道推断共用此函数。
 */
export function effectiveChannels(
    sourceChannels: number | null | undefined,
    channelMode: number | null | undefined,
): 1 | 2 {
    const mode = normalizeChannelMode(channelMode);
    if (mode === 2 || mode === 3 || mode === 4) return 1;
    const src = Number(sourceChannels);
    if (!Number.isFinite(src) || src <= 1) return 1;
    return 2;
}

/** 模式循环顺序（0→1→2→3→4→0），供 UI 循环按钮使用。 */
const MODE_CYCLE_ORDER: TakeChannelModeRaw[] = [0, 1, 2, 3, 4];

/** 循环到下一个声道模式。 */
export function nextChannelMode(value: number | null | undefined): TakeChannelModeRaw {
    const current = normalizeChannelMode(value);
    const index = MODE_CYCLE_ORDER.indexOf(current);
    return MODE_CYCLE_ORDER[(index + 1) % MODE_CYCLE_ORDER.length];
}

/** 模式的 i18n key（clip_channel_mode_*）。 */
export function channelModeI18nKey(value: number | null | undefined): MessageKey {
    switch (normalizeChannelMode(value)) {
        case 1:
            return "clip_channel_mode_swap";
        case 2:
            return "clip_channel_mode_mono_mix";
        case 3:
            return "clip_channel_mode_mono_left";
        case 4:
            return "clip_channel_mode_mono_right";
        default:
            return "clip_channel_mode_normal";
    }
}

/** 模式的紧凑徽章缩写（语言无关符号，菜单行尾按钮与 ClipHeader 徽章共用）。 */
export function channelModeShortLabel(value: number | null | undefined): string {
    switch (normalizeChannelMode(value)) {
        case 1:
            return "⇄";
        case 2:
            return "MIX";
        case 3:
            return "L";
        case 4:
            return "R";
        default:
            return "L·R";
    }
}
