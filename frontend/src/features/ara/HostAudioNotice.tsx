/**
 * 宿主音频提示条（插件模式）：把"插件挂在轨道组父轨上"这件事说清楚。
 *
 * 【为什么需要一条横条，而不是状态栏再多个片】症状极具误导性 —— 轨道和片段都看得见、
 * 也能拖动去改 REAPER，唯独片段里没有内容。用户无从判断是自己用法不对还是插件坏了；
 * 状态栏的 `等待宿主音频` 又恰好是加载过程中的正常读数，无法区分这两种处境。这里给出
 * **原因 + 可执行的下一步**，并在问题消失后自动消失。
 *
 * 【为什么只在 folder 父轨这一种情形出现】`awaiting_regions` 是正常的中间态，每次
 * 打开工程都会短暂出现；为它弹横条等于制造噪音（状态栏的读数已经覆盖了它）。
 *
 * 【措辞红线】不说"ARA 规范不支持跨轨" —— ARA 2.0 规范**允许**一个实例服务多个
 * region sequence；真正的原因是 **REAPER 按轨道管理 ARA 插件**。错的解释会把用户
 * （以及日后真正可行的改进方向）带偏。
 */
import { Flex } from "@radix-ui/themes";

import { useI18n } from "../../i18n/I18nProvider";
import type { HostAudioPayload } from "../../types/api";
import { needsFolderTrackNotice, needsUnidentifiedHostNotice } from "./hostAudio";

export function HostAudioNotice({ status }: { status: HostAudioPayload | null }) {
    const { t } = useI18n();
    // 宿主不受支持与 folder 父轨是**两种不同的成因**，文案必须分开：前者是"这个宿主
    // 还没适配"，后者是"插件挂错了轨道"。共用一句话会把用户引向错误的下一步
    //（改轨道布局 vs 换宿主）。
    if (needsUnidentifiedHostNotice(status)) {
        return (
            <Flex
                role="status"
                direction="column"
                gap="1"
                className="bg-qt-window border-b border-qt-border px-qt-4 py-qt-2 select-none"
            >
                <span className="hs-type-label font-bold">
                    {t("ara_host_audio_unidentified_title")}
                </span>
                <span className="hs-type-caption" style={{ color: "var(--qt-text-muted)" }}>
                    {t("ara_host_audio_unidentified_body")}
                </span>
            </Flex>
        );
    }
    if (!needsFolderTrackNotice(status)) return null;
    return (
        <Flex
            role="status"
            direction="column"
            gap="1"
            className="bg-qt-window border-b border-qt-border px-qt-4 py-qt-2 select-none"
        >
            <span className="hs-type-label font-bold">{t("ara_host_audio_folder_title")}</span>
            <span className="hs-type-caption" style={{ color: "var(--qt-text-muted)" }}>
                {t("ara_host_audio_folder_body")}
            </span>
        </Flex>
    );
}
