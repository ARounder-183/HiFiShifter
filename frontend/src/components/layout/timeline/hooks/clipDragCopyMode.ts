import { IS_MAC } from "../../../../utils/platform.ts";

export function resolveClipDragCopyMode(args: {
    existingCopyMode: boolean;
    ctrlKey: boolean;
    modifierActive: boolean;
}): boolean {
    // The configured copy-drag binding is authoritative. Keep the legacy
    // hard-coded Ctrl fallback only on Windows/Linux; on macOS the ctrl
    // field maps to Command, so the default binding still covers ⌘+drag.
    return args.existingCopyMode || args.modifierActive || (!IS_MAC && args.ctrlKey);
}
