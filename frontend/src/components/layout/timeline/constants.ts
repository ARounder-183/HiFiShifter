export const DEFAULT_PX_PER_BEAT = 75;
export const MIN_PX_PER_BEAT = 8;
export const MAX_PX_PER_BEAT = 2000;

// 以秒为单位的缩放常量（pxPerSec = pxPerBeat / secPerBeat = pxPerBeat * bpm / 60）
// DEFAULT_PX_PER_SEC = 240 对应 120 BPM 时 pxPerBeat = 240 * (60/120) = 120
export const DEFAULT_PX_PER_SEC = 150;
export const MIN_PX_PER_SEC = 4;
export const MAX_PX_PER_SEC = 8000;

export const DEFAULT_ROW_HEIGHT = 96;
export const MIN_ROW_HEIGHT = 80;
export const MAX_ROW_HEIGHT = 192;
export const TRACK_ADD_ROW_HEIGHT = 32;

export const CLIP_HEADER_HEIGHT = 18;
export const CLIP_BODY_PADDING_Y = 6;
