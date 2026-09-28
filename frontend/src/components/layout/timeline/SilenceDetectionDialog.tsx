import { useCallback, useEffect, useRef, useState } from "react";
import { Flex, Text, Checkbox } from "@radix-ui/themes";
import { useI18n } from "../../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import {
    setSilenceDetectOptions,
    setSilencePreview,
    persistUiSettings,
    analyzeSilenceRemote,
    removeSilenceRemote,
} from "../../../features/session/sessionSlice";
import {
    SILENCE_DETECT_DEFAULTS,
    type SilenceDetectSettings,
} from "../../../features/session/sessionTypes";
import type { ClipSilenceReport } from "../../../types/api";
import { AppSelect, AppSlider, AppSliderReadout } from "../../../ui";
import { AppDialog } from "../../../ui/Dialog";
import { AppField, AppForm } from "../../../ui/Field";
import { useShortcutSuppression } from "../../../ui/shortcutScope";

/** 预览分析的防抖时间（ms）。 */
const PREVIEW_DEBOUNCE_MS = 300;

/**
 * 静音检测设置对话框（Clip 右键菜单 → "静音检测…"）。
 *
 * - 打开/参数变化时防抖调用后端干跑分析，检测到的静音区以红色覆盖层
 *   实时标在时间线 Clip 上（`session.silencePreviewSegments`）；
 * - 应用时后端单命令完成"切分 → 删除 → 闭合 → 切边淡化"（单次撤销）；
 * - 应用成功后把当前参数记忆为下次默认（persistUiSettings）。
 */
export const SilenceDetectionDialog: React.FC<{
    open: boolean;
    onOpenChange: (open: boolean) => void;
    clipIds: string[];
}> = ({ open, onOpenChange, clipIds }) => {
    const { tf } = useI18n();
    const dispatch = useAppDispatch();
    const storedOptions = useAppSelector((s) => s.session.silenceDetectOptions);
    const clips = useAppSelector((s) => s.session.clips);
    const preview = useAppSelector((s) => s.session.silencePreviewSegments);

    const [options, setOptions] = useState<SilenceDetectSettings>(storedOptions);
    const [reports, setReports] = useState<ClipSilenceReport[]>([]);
    const [analyzing, setAnalyzing] = useState(false);
    const [applying, setApplying] = useState(false);
    const debounceRef = useRef<number | null>(null);

    // 打开时用已记忆的参数初始化，并清掉上一次的预览报告。
    useEffect(() => {
        if (open) {
            setOptions(storedOptions);
            setReports([]);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅在打开瞬间用 store 种子化
    }, [open]);

    // 模态打开期间抑制全局快捷键：捕获阶段的 window 监听先于对话框内部处理，
    // 箭头/空格/字母键否则会穿透到背后（暗改轨道选择 / 误触播放）。
    // 走统一作用域，取代此前的 `data-silence-dialog-open` body 属性。
    useShortcutSuppression(open);

    // 关闭/卸载时清除覆盖层。
    const closeAndCleanup = useCallback(() => {
        if (debounceRef.current != null) {
            window.clearTimeout(debounceRef.current);
            debounceRef.current = null;
        }
        dispatch(setSilencePreview(null));
        onOpenChange(false);
    }, [dispatch, onOpenChange]);

    // 预览：打开 + 参数变化 → 防抖干跑分析（fulfilled reducer 写入预览覆盖层）。
    useEffect(() => {
        if (!open) return;
        if (clipIds.length === 0) {
            dispatch(setSilencePreview(null));
            return;
        }
        if (debounceRef.current != null) window.clearTimeout(debounceRef.current);
        debounceRef.current = window.setTimeout(() => {
            debounceRef.current = null;
            setAnalyzing(true);
            void dispatch(analyzeSilenceRemote({ clipIds, options: { ...options } }))
                .unwrap()
                .then((result) => setReports(result.reports))
                .catch(() => setReports([]))
                .finally(() => setAnalyzing(false));
        }, PREVIEW_DEBOUNCE_MS);
        return () => {
            if (debounceRef.current != null) {
                window.clearTimeout(debounceRef.current);
                debounceRef.current = null;
            }
        };
    }, [open, clipIds, options, dispatch]);

    const handleApply = async () => {
        if (applying) return;
        setApplying(true);
        try {
            await dispatch(removeSilenceRemote({ clipIds, options: { ...options } })).unwrap();
            // 记忆本次参数为下次默认。
            dispatch(setSilenceDetectOptions(options));
            void dispatch(persistUiSettings());
            dispatch(setSilencePreview(null));
            onOpenChange(false);
        } catch {
            // 失败保持对话框打开（fulfilled/rejected 状态由 status 区域呈现）。
        } finally {
            setApplying(false);
        }
    };

    const reportFor = (id: string) => reports.find((r) => r.clipId === id);
    const okReports = reports.filter((r) => r.ok);
    const totalRegions = okReports.reduce((acc, r) => acc + r.regions.length, 0);
    const totalSilent = okReports.reduce((acc, r) => acc + r.totalSilentSec, 0);
    const previewFor = (id: string) => preview?.[id];

    const update = (patch: Partial<SilenceDetectSettings>) =>
        setOptions((prev) => ({ ...prev, ...patch }));

    // 双击参数行 = 该参数重置回默认值（提示文案挂在行 title 上）。
    const resetHint = tf("silence_double_click_reset");

    return (
        <AppDialog
            open={open}
            onOpenChange={(next) => {
                if (!next) closeAndCleanup();
            }}
            title={tf("ctx_silence_detection")}
            size="md"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: closeAndCleanup },
                {
                    id: "apply",
                    label: applying ? tf("silence_applying") : tf("silence_apply"),
                    intent: "primary",
                    disabled: applying || clipIds.length === 0,
                    onClick: handleApply,
                },
            ]}
        >
            <AppForm>
                <div
                    data-tooltip={resetHint}
                    onDoubleClick={() => update({ method: SILENCE_DETECT_DEFAULTS.method })}
                >
                    <AppField label={tf("silence_method")}>
                        <AppSelect
                            // 旧写法是 size="1"（24px）：对话框里也要紧凑
                            density="compact"
                            value={options.method}
                            onValueChange={(v) => update({ method: v as "rms" | "peak" })}
                            options={[
                                { value: "rms", label: tf("silence_method_rms") },
                                { value: "peak", label: tf("silence_method_peak") },
                            ]}
                        />
                    </AppField>
                </div>

                <div
                    data-tooltip={resetHint}
                    onDoubleClick={() =>
                        update({ thresholdDb: SILENCE_DETECT_DEFAULTS.thresholdDb })
                    }
                >
                    <AppField label={tf("silence_threshold")}>
                        <Flex align="center" gap="2">
                            <AppSlider
                                value={options.thresholdDb}
                                unit="levelDb"
                                min={-96}
                                max={-6}
                                disabled={options.adaptive}
                                ariaLabel={tf("silence_threshold")}
                                onChange={(next) => update({ thresholdDb: next })}
                            />
                            <AppSliderReadout>
                                {options.adaptive
                                    ? tf("silence_adaptive_short")
                                    : `${Math.round(options.thresholdDb)} dB`}
                            </AppSliderReadout>
                        </Flex>
                    </AppField>
                </div>

                <label
                    className="flex items-center gap-2 text-[12px]"
                    data-tooltip={resetHint}
                    onDoubleClick={() => update({ adaptive: SILENCE_DETECT_DEFAULTS.adaptive })}
                >
                    <Checkbox
                        checked={options.adaptive}
                        onCheckedChange={(v) => update({ adaptive: v === true })}
                    />
                    {tf("silence_adaptive")}
                </label>

                {(
                    [
                        ["minSilenceMs", "silence_min_silence", 10, 2000],
                        ["minSoundMs", "silence_min_sound", 0, 500],
                        ["paddingMs", "silence_padding", 0, 200],
                        ["cutFadeMs", "silence_cut_fade", 0, 100],
                    ] as const
                ).map(([key, labelKey, min, max]) => (
                    <div
                        key={key}
                        data-tooltip={resetHint}
                        onDoubleClick={() =>
                            update({
                                [key]: SILENCE_DETECT_DEFAULTS[key],
                            } as Partial<SilenceDetectSettings>)
                        }
                    >
                        <AppField label={tf(labelKey)}>
                            <Flex align="center" gap="2">
                                <AppSlider
                                    value={options[key]}
                                    unit="milliseconds"
                                    min={min}
                                    max={max}
                                    ariaLabel={tf(labelKey)}
                                    onChange={(next) => update({ [key]: next })}
                                />
                                <AppSliderReadout>{Math.round(options[key])} ms</AppSliderReadout>
                            </Flex>
                        </AppField>
                    </div>
                ))}

                <div
                    data-tooltip={resetHint}
                    onDoubleClick={() => update({ action: SILENCE_DETECT_DEFAULTS.action })}
                >
                    <AppField label={tf("silence_action")}>
                        <AppSelect
                            // 旧写法是 size="1"（24px）：对话框里也要紧凑
                            density="compact"
                            value={options.action}
                            onValueChange={(v) =>
                                update({ action: v as SilenceDetectSettings["action"] })
                            }
                            options={[
                                { value: "close", label: tf("silence_action_close") },
                                { value: "keep", label: tf("silence_action_keep") },
                                { value: "split", label: tf("silence_action_split") },
                            ]}
                        />
                    </AppField>
                </div>

                <label
                    className="flex items-center gap-2 text-[12px]"
                    data-tooltip={resetHint}
                    onDoubleClick={() =>
                        update({ deleteSilentClips: SILENCE_DETECT_DEFAULTS.deleteSilentClips })
                    }
                >
                    <Checkbox
                        checked={options.deleteSilentClips}
                        onCheckedChange={(v) => update({ deleteSilentClips: v === true })}
                    />
                    {tf("silence_delete_silent_clips")}
                </label>
                <label
                    className="flex items-center gap-2 text-[12px]"
                    data-tooltip={resetHint}
                    onDoubleClick={() =>
                        update({ syncAllTakes: SILENCE_DETECT_DEFAULTS.syncAllTakes })
                    }
                >
                    <Checkbox
                        checked={options.syncAllTakes}
                        onCheckedChange={(v) => update({ syncAllTakes: v === true })}
                    />
                    {tf("silence_sync_all_takes")}
                </label>

                {/* 预览摘要（与时间线上的红色覆盖层联动） */}
                <Flex direction="column" gap="1" className="rounded border border-qt-border p-2">
                    <Text size="1" className="text-qt-text-muted">
                        {analyzing
                            ? tf("silence_analyzing")
                            : totalRegions > 0
                              ? tf("silence_preview_summary")
                                    .replace("{n}", String(totalRegions))
                                    .replace("{dur}", totalSilent.toFixed(2))
                              : tf("silence_no_silence")}
                    </Text>
                    {clipIds.slice(0, 6).map((id) => {
                        const clip = clips.find((c) => c.id === id);
                        const report = reportFor(id);
                        const regions = previewFor(id)?.length ?? 0;
                        const name = clip?.name ?? id;
                        return (
                            <Text key={id} size="1" className="text-qt-text-muted">
                                {`· ${name}: `}
                                {report && !report.ok
                                    ? tf("silence_skipped") +
                                      (report.message ? ` (${report.message})` : "")
                                    : regions > 0
                                      ? tf("silence_preview_clip")
                                            .replace("{n}", String(regions))
                                            .replace(
                                                "{dur}",
                                                (report?.totalSilentSec ?? 0).toFixed(2),
                                            )
                                      : tf("silence_no_silence")}
                            </Text>
                        );
                    })}
                    {clipIds.length > 6 ? (
                        <Text size="1" className="text-qt-text-muted">
                            {tf("silence_more_clips").replace("{n}", String(clipIds.length - 6))}
                        </Text>
                    ) : null}
                </Flex>
            </AppForm>
        </AppDialog>
    );
};
