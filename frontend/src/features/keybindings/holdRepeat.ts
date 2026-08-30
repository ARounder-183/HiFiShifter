/**
 * holdRepeat.ts — 长按重复执行管理器
 *
 * 提供与「粘贴」（clip.paste）一致的长按节奏：
 * 按下快捷键立即执行一次 → 停顿 `initialDelayMs` → 之后每
 * `repeatIntervalMs` 重复执行一次，直到松开按键 / 窗口失焦 /
 * 按下其它按键（= 意图变化，终止长按）。
 *
 * 供「添加轨道」「克隆选中轨道」「参数线上移/下移」等动作共用，
 * 也替换 TimelinePanel 中粘贴内嵌的长按逻辑 —— 所有长按重复动作
 * 共用一个管理器、一套节奏与一套终止规则。
 *
 * 与 OS 自动重复的关系：长按进行中，同键的 OS 自动重复事件会被
 * `consumeHoldRepeatKeyDown` 吞掉（节奏完全由计时器控制，避免
 * 双触发）；未进行长按时，OS 自动重复由各调用方自行决定
 * （如方向键类导航仍在 REPEATABLE_ACTIONS 中走系统重复）。
 */

import type { Keybinding } from "./types";

export interface HoldRepeatOptions {
    /** 按下后到开始重复的停顿（毫秒）。默认 400。 */
    initialDelayMs?: number;
    /** 重复执行的间隔（毫秒）。默认 50。 */
    repeatIntervalMs?: number;
}

interface ActiveHold {
    key: string;
    fire: () => void;
    initialTimer: number | null;
    repeatTimer: number | null;
}

let active: ActiveHold | null = null;
let listenersAttached = false;

function isModifierKey(key: string): boolean {
    const k = key.toLowerCase();
    return k === "control" || k === "shift" || k === "alt" || k === "meta";
}

function stop(): void {
    if (!active) return;
    if (active.initialTimer != null) clearTimeout(active.initialTimer);
    if (active.repeatTimer != null) clearInterval(active.repeatTimer);
    active = null;
}

function onKeyUp(e: KeyboardEvent): void {
    const a = active;
    if (!a) return;
    // 松开主键或任意修饰键都结束长按（与粘贴实现一致）。
    if (e.key.toLowerCase() === a.key || isModifierKey(e.key)) {
        stop();
    }
}

function onBlur(): void {
    stop();
}

function attachGlobalListeners(): void {
    // 单元测试运行在无 DOM 环境时跳过监听挂载；计时/吞咽逻辑仍可测。
    if (listenersAttached || typeof window === "undefined") return;
    listenersAttached = true;
    window.addEventListener("keyup", onKeyUp, true);
    window.addEventListener("blur", onBlur);
}

/** 当前是否有长按重复进行中。 */
export function isHoldRepeatActive(): boolean {
    return active != null;
}

/**
 * 开始一次长按重复。调用方应先自行执行首次动作，再调用本函数
 * （与粘贴流程一致：先 onPaste()，再 beginHoldRepeat(kb, onPaste)）。
 *
 * @param binding 触发该动作的键位（用于识别同键自动重复与松开）。
 * @param fire    重复执行的回调；调用方需确保它读取最新状态。
 */
export function beginHoldRepeat(
    binding: Keybinding,
    fire: () => void,
    options?: HoldRepeatOptions,
): void {
    stop();
    attachGlobalListeners();
    const initialDelayMs = Math.max(0, options?.initialDelayMs ?? 400);
    const repeatIntervalMs = Math.max(1, options?.repeatIntervalMs ?? 50);
    const hold: ActiveHold = {
        key: binding.key.toLowerCase(),
        fire,
        initialTimer: null,
        repeatTimer: null,
    };
    active = hold;
    hold.initialTimer = setTimeout(() => {
        hold.initialTimer = null;
        hold.repeatTimer = setInterval(fire, repeatIntervalMs);
    }, initialDelayMs);
}

/** 手动终止长按重复（keyup / blur 已由内部监听处理）。 */
export function stopHoldRepeat(): void {
    stop();
}

/**
 * 在各 keydown 处理器的最前面调用（优先级最高；先于可编辑目标判断）。
 *
 * - 长按进行中：
 *   - 同键 OS 自动重复 → preventDefault 并吞掉（节奏由计时器控制）；
 *   - 其它按键的自动重复 → 同样吞掉（避免与计时器双触发）；
 *   - 任何非重复按键 → 视为意图变化，终止长按（粘贴语义）。
 * - 无长按进行中：返回 false，调用方照常处理。
 *
 * @returns true 表示该 keydown 已被管理器消费，调用方应直接 return。
 */
export function consumeHoldRepeatKeyDown(e: KeyboardEvent): boolean {
    if (!active) return false;
    if (e.repeat) {
        if (e.key.toLowerCase() === active.key) {
            e.preventDefault();
        }
        return true;
    }
    stop();
    return false;
}
