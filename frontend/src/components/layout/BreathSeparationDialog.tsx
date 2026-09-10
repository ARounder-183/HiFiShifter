/**
 * 开启气声分离前的确认对话框（P2-5）。
 *
 * 为什么需要它：气声分离（HNSEP）对**每个音频块整段**做一次神经网络推理，首次
 * 开启时表现为"长时间没有任何反应"（README 也承认"首次需要较长时间处理"）。
 * 这里在用户真正开启**之前**把工作量讲清楚，并留一条退路 —— 取消即保持关闭。
 *
 * 为什么只报告事实、不给"预计耗时 X 秒"：整段推理的耗时取决于片段时长与当前
 * 执行提供者（GPU/CPU），在没有实测吞吐的情况下给出秒数就是编造。因此这里只
 * 报告可确证的两件事：需要处理几个块、共约多少分钟音频。
 *
 * `available === false` 时额外警告：气声模型不可用会让这些音频块**渲染失败并
 * 保持静音**（该失败同时会经 `render_warning` 上报，见 P0-5），所以更要在开启
 * 前说清楚，而不是等用户听到无声再来排查。
 */

import { Button, Dialog, Flex, Text } from "@radix-ui/themes";
import type { BreathSeparationWorkload } from "../../services/api/params";
import { useI18n } from "../../i18n/I18nProvider";

interface BreathSeparationDialogProps {
    open: boolean;
    /** 工作量统计；null 表示尚未取得（理论上打开时已有值）。 */
    workload: BreathSeparationWorkload | null;
    onOpenChange: (open: boolean) => void;
    /** 用户确认开启。 */
    onConfirm: () => void;
}

/**
 * 极简模板替换：`t()` 不支持插值（见 I18nProvider 的 `t: (key) => string`），
 * 所以带数字的文案在组件侧填值。用 `split/join` 而非正则，避免 `{}` 需要转义。
 */
function fill(template: string, vars: Record<string, string>): string {
    return Object.entries(vars).reduce(
        (acc, [key, value]) => acc.split(`{${key}}`).join(value),
        template,
    );
}

export function BreathSeparationDialog({
    open,
    workload,
    onOpenChange,
    onConfirm,
}: BreathSeparationDialogProps) {
    const { t } = useI18n();

    const unavailable = workload?.available === false;
    const clipCount = workload?.clipCount ?? 0;
    // 总时长按分钟呈现：秒级精度对"要等多久"的判断没有意义。
    const minutes = Math.max(1, Math.round((workload?.totalDurationSec ?? 0) / 60));

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 480 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{t("breath_enable_confirm_title")}</Dialog.Title>

                <Flex direction="column" gap="3" mt="3">
                    <Text size="2" style={{ lineHeight: 1.7 }}>
                        {fill(t("breath_enable_confirm_body"), {
                            count: String(clipCount),
                            minutes: String(minutes),
                        })}
                    </Text>
                    <Text size="2" style={{ lineHeight: 1.7 }}>
                        {t("breath_enable_confirm_cached_note")}
                    </Text>

                    {unavailable ? (
                        <Text size="2" color="amber" style={{ lineHeight: 1.7 }}>
                            {t("breath_enable_confirm_unavailable")}
                        </Text>
                    ) : null}
                </Flex>

                <Flex gap="3" mt="4" justify="end">
                    <Dialog.Close>
                        <Button size="2" variant="surface" color="gray">
                            {t("cancel")}
                        </Button>
                    </Dialog.Close>
                    <Button size="2" onClick={onConfirm}>
                        {unavailable
                            ? t("breath_enable_confirm_ok_anyway")
                            : t("breath_enable_confirm_ok")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
