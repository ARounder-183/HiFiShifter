/**
 * 宿主音频提示条（插件模式）：把"为什么片段里没有内容"这件事说清楚。
 *
 * 【为什么需要一条横条，而不是状态栏再多个片】这些症状极具误导性 —— 轨道和片段都
 * 看得见、也能拖动去改宿主，唯独片段里没有内容。用户无从判断是自己用法不对还是插件
 * 坏了；状态栏的 `等待宿主音频` 又恰好是加载过程中的正常读数，无法区分这几种处境。
 * 这里给出**原因 + 可执行的下一步**，并在问题消失后自动消失。
 *
 * 【为什么每种成因各一条】它们的**下一步完全不同**：
 *   - folder 父轨 ⇒ 改轨道布局；
 *   - 分配了区域但对不上 ⇒ 检查该轨的片段与区域是否一一对应；
 *   - 有片段被跳过 ⇒ 看是哪一类片段不被支持。
 * 共用一句话会把用户引向一件无关的事。
 *
 * 【为什么没有"宿主不受支持 / 只提供 ARA"那两条】整族删除了。ARA-only 宿主
 *（Studio One / Cubase / Nuendo / Logic …）本来就正常工作：时间线有内容、音频能出、
 * 参数能调，只读的只有片段几何写回 —— 而那由能力位门控，不需要横幅。"既不提供原生
 * 扩展、也不提供 ARA"的宿主根本拿不到文档，插件报的是 `ARA_NOT_BOUND` 那条终态错误，
 * 不会表现为"空白时间线 + 一条猜错方向的横幅"。
 *
 * 【措辞红线】不说"ARA 规范不支持跨轨" —— ARA 2.0 规范**允许**一个实例服务多个
 * region sequence；真正的原因是 **REAPER 按轨道管理 ARA 插件**。错的解释会把用户
 * （以及日后真正可行的改进方向）带偏。
 */
import { Flex } from "@radix-ui/themes";
import type { ReactNode } from "react";

import { useI18n } from "../../i18n/I18nProvider";
import type { HostAudioPayload } from "../../types/api";
import {
    needsFolderTrackNotice,
    needsSkippedClipsNotice,
    needsUnmatchedRegionsNotice,
} from "./hostAudio";

/**
 * 提示条的**唯一**版式：标题 + 说明两行。
 *
 * 【为什么抽出来】四种成因此前各写一遍同一段 `Flex` —— 改一次版式要改四处，
 * 而漏掉一处只会表现为"某一种提示长得不太一样"，很难在 review 里发现。
 */
function Notice({ title, body }: { title: string; body: ReactNode }) {
    return (
        <Flex
            role="status"
            direction="column"
            gap="1"
            className="bg-qt-window border-b border-qt-border px-qt-4 py-qt-2 select-none"
        >
            <span className="hs-type-label font-bold">{title}</span>
            <span className="hs-type-caption" style={{ color: "var(--qt-text-muted)" }}>
                {body}
            </span>
        </Flex>
    );
}

export function HostAudioNotice({ status }: { status: HostAudioPayload | null }) {
    const { t, plural } = useI18n();
    // 【为什么排在 folder 之前】"宿主分配了区域但一条都对不上"比"还没拿到区域"更具体：
    // 它排除了"宿主尚未分配"这一成因，用户不必再等。
    if (needsUnmatchedRegionsNotice(status)) {
        return (
            <Notice
                title={t("ara_host_audio_unmatched_title")}
                body={t("ara_host_audio_unmatched_body")}
            />
        );
    }
    const skipped = needsSkippedClipsNotice(status);
    if (skipped > 0) {
        // 被跳过的片段在插件里**根本不存在**：不提示的话用户只看到"内容不见了"。
        return (
            <Notice
                title={t("ara_host_audio_skipped_title")}
                body={plural("ara_host_audio_skipped_body", skipped)}
            />
        );
    }
    if (!needsFolderTrackNotice(status)) return null;
    return (
        <Notice title={t("ara_host_audio_folder_title")} body={t("ara_host_audio_folder_body")} />
    );
}
