/**
 * 时间轴渲染内核 · 不可用提示界面
 *
 * 【主要内容】
 * WebGL2 不可用时替代时间轴轨道区显示的界面：标题、原因、按命中概率排序的排查
 * 清单，以及可一键复制的诊断信息。
 *
 * 【作用：为什么必须是"可自助排障"的界面】
 * 内核是唯一渲染路径 —— 没有 Canvas2D 回退，这个界面就是失败场景下用户能看到的
 * **全部**。因此它不能是一行红字：必须让用户（或提供支持的人）能判断原因并采到
 * 诊断信息。实测确认 WebGL2 在纯 CPU 环境（含无 GPU 机器）**是可用的**（ANGLE +
 * SwiftShader 软件光栅），只有显式禁止软件光栅一类配置才会失败，所以清单按实际
 * 命中概率排序。
 *
 * 【不做的事】
 * - 不提供"重试"按钮：失败原因（无 GL）在会话内不会自愈，重试只会反复刷日志。
 * - 不静默降级、不显示空白：宁可明确报错，也不让用户面对一个没有内容的界面。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 在 `isKernelAvailable()` 为 false 时渲染本组件。
 * - 下游：`collectGlDiagnostics()` 提供诊断数据；剪贴板写入沿用本工程既有的
 *   `navigator.clipboard` + `execCommand` 兜底模式（见 `TimelinePanel` 的
 *   复制播放头时间）。
 * - 独立性：只依赖 i18n 与诊断模块，不读 Redux。
 */
import React from "react";
import { Button, Flex, Text } from "@radix-ui/themes";

import { useI18n } from "../../../../i18n/I18nProvider";
import { collectGlDiagnostics } from "../../renderKernel/gl/glDiagnostics";

interface Props {
    /** 失败原因（来自内核视图的回报，用于日志与展示）。 */
    readonly reason: string;
}

export const KernelUnavailableNotice: React.FC<Props> = ({ reason }) => {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [copied, setCopied] = React.useState(false);

    /**
     * 复制诊断信息到剪贴板。
     *
     * 流程：收集诊断 → 优先 `navigator.clipboard.writeText` → 失败则退回
     * `textarea` + `execCommand("copy")`（与面板既有的复制实现同一模式）→ 成功后
     * 短暂显示"已复制"。
     *
     * 特殊说明：两级兜底是必要的——Tauri / 非安全上下文下 `navigator.clipboard`
     * 可能不可用，而诊断信息正是排障时最需要交出去的东西。复制失败不阻断任何流程
     * （界面本身仍展示原因）。
     */
    const handleCopy = React.useCallback(async () => {
        const d = collectGlDiagnostics();
        const text = [
            `reason: ${reason}`,
            `webgl2: ${d.webgl2}`,
            `webgl1: ${d.webgl1}`,
            `devicePixelRatio: ${d.devicePixelRatio}`,
            `userAgent: ${d.userAgent}`,
        ].join("\n");
        try {
            await navigator.clipboard.writeText(text);
        } catch {
            try {
                const textarea = document.createElement("textarea");
                textarea.value = text;
                textarea.style.position = "fixed";
                textarea.style.opacity = "0";
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand("copy");
                textarea.remove();
            } catch {
                // 忽略复制失败：界面仍展示原因与排查清单。
            }
        }
        setCopied(true);
        window.setTimeout(() => setCopied(false), 2000);
    }, [reason]);

    return (
        <Flex
            direction="column"
            gap="3"
            align="center"
            justify="center"
            className="absolute inset-0 px-6 text-center"
        >
            <Text size="4" weight="bold">
                {tAny("kernel_unavailable_title")}
            </Text>
            <Text size="2" color="gray">
                {tAny("kernel_unavailable_reason")}
            </Text>
            <Text size="1" color="gray" className="max-w-[620px] text-left whitespace-pre-line">
                {tAny("kernel_unavailable_hints")}
            </Text>
            <Button size="1" variant="soft" onClick={() => void handleCopy()}>
                {copied
                    ? tAny("kernel_unavailable_copied")
                    : tAny("kernel_unavailable_diagnostics")}
            </Button>
        </Flex>
    );
};
