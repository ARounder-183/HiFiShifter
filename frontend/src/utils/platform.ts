/**
 * Platform detection helpers shared by keyboard and mouse-modifier logic.
 *
 * On macOS the Command key is the primary modifier (shortcuts, multi-select),
 * while Control is reserved for secondary/context-menu behavior.  On Windows
 * and Linux Control is the primary modifier.
 */

export const IS_MAC =
    typeof navigator !== "undefined" &&
    /Mac|iPhone|iPad|iPod/i.test(navigator.platform || navigator.userAgent);

export const IS_LINUX =
    typeof navigator !== "undefined" && /Linux/i.test(navigator.platform || navigator.userAgent);

/**
 * Windows only.
 *
 * Used to gate features that depend on Windows-only third-party software —
 * e.g. VocalShifter writes its clipboard exchange files into `%TEMP%`, so the
 * "paste VocalShifter clipboard" menu entry is hidden elsewhere rather than
 * offered and then failing.
 *
 * Same detection style as `IS_MAC` / `IS_LINUX`: `navigator.platform` with the
 * user-agent as fallback (`platform` is deprecated but is what the rest of the
 * codebase uses, and it is never empty in the WebView2 runtime).
 */
export const IS_WINDOWS =
    typeof navigator !== "undefined" && /Win/i.test(navigator.platform || navigator.userAgent);

export type ModifierEventLike = {
    ctrlKey: boolean;
    metaKey?: boolean;
};

/**
 * Returns true when the platform's primary modifier is currently held.
 * macOS → Command (metaKey), Windows/Linux → Control (ctrlKey).
 */
export function isPrimaryModifierDown(event: ModifierEventLike): boolean {
    return IS_MAC ? Boolean(event.metaKey) : Boolean(event.ctrlKey);
}
