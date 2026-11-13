/**
 * 把「另存为」的失败码变成状态栏提示。
 *
 * 【为什么是一个 hook】这条提示必须挂在**唯一**有状态栏的位置（`App`），而失败
 * 来源有五个入口 —— 逐个 `catch` 会把同一段映射抄五份，键盘通道尤其容易漏。
 * thunk 只发布错误码（见 `saveProjectAsNotice`），措辞在这里统一决定。
 */
import { useEffect } from "react";

import { useI18n } from "../../i18n/I18nProvider";
import type { AppStatusTone } from "../../ui/StatusChip";
import { subscribeSaveProjectAsFailure } from "./saveProjectAsNotice";

type ShowNotice = (text: string, holdMs?: number, tone?: AppStatusTone, tooltip?: string) => void;

export function useSaveProjectAsNotice(showNotice: ShowNotice): void {
    const { tf } = useI18n();
    useEffect(
        () =>
            subscribeSaveProjectAsFailure((code) => {
                const mismatch = code === "format_extension_mismatch";
                showNotice(
                    mismatch ? tf("save_as_format_mismatch") : tf("status_save_as_failed"),
                    6000,
                    "warning",
                    // 具体错误码放进 tooltip：状态片只放状态，诊断信息不该挤在那一行。
                    mismatch ? "" : code,
                );
            }),
        [showNotice, tf],
    );
}
