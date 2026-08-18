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
