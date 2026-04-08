import type { KeybindingOverrides } from "./types";

const STORAGE_KEY = "hifishifter.keybindings";

/**
 * 从 localStorage 加载用户自定义的快捷键覆盖项
 */
export function loadKeybindingOverrides(): KeybindingOverrides {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
            return {};
        }
        return parsed as KeybindingOverrides;
    } catch {
        return {};
    }
}

/**
 * 将用户自定义的快捷键覆盖项保存到 localStorage
 */
export function saveKeybindingOverrides(overrides: KeybindingOverrides): void {
    try {
        // 只保存非空的覆盖项
        const cleaned = Object.fromEntries(Object.entries(overrides).filter(([, v]) => v != null));
        if (Object.keys(cleaned).length === 0) {
            localStorage.removeItem(STORAGE_KEY);
        } else {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(cleaned));
        }
    } catch {
        // localStorage 不可用时静默失败
    }
}
