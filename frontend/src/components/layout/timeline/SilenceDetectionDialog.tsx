import { useCallback, useEffect, useRef, useState } from "react";
import { Dialog, Flex, Text, Button, Select, Checkbox } from "@radix-ui/themes";
import { useI18n } from "../../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import { isModifierActive, selectKeybinding } from "../../../features/keybindings/keybindingsSlice";
import { applySelectWheelChange } from "../../../utils/selectWheel";
import { useWheelScrollGuard } from "../../../utils/useWheelScrollGuard";
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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const dispatch = useAppDispatch();
    // 滑块滚轮的“精细调整”修饰键（与其他编辑对话框一致：modifier.paramFineAdjust）。
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    const storedOptions = useAppSelector((s) => s.session.silenceDetectOptions);
    const clips = useAppSelector((s) => s.session.clips);
    const preview = useAppSelector((s) => s.session.silencePreviewSegments);

    const [options, setOptions] = useState<SilenceDetectSettings>(storedOptions);
    const [reports, setReports] = useState<ClipSilenceReport[]>([]);
    const [analyzing, setAnalyzing] = useState(false);
    const [applying, setApplying] = useState(false);
    const debounceRef = useRef<number | null>(null);
    // 滚轮守卫：滑块滚轮步进时阻止对话框内容滚动（React onWheel 的
    // preventDefault 是 passive no-op，见 useWheelScrollGuard）。
    const wheelGuard = useWheelScrollGuard<HTMLDivElement>('input[type="range"]');

    // 打开时用已记忆的参数初始化，并清掉上一次的预览报告。
    useEffect(() => {
        if (open) {
            setOptions(storedOptions);
            setReports([]);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅在打开瞬间用 store 种子化
    }, [open]);

    // 打开时在 body 上标记：模态对话框打开期间阻塞全局快捷键（捕获阶段
    // 的 window 监听先于对话框内部处理，箭头/空格/字母键否则会穿透到
    // 对话框背后暗改轨道选择 / 误触播放）。
    useEffect(() => {
        if (open) {
            document.body.setAttribute("data-silence-dialog-open", "true");
        } else {
            document.body.removeAttribute("data-silence-dialog-open");
        }
        return () => {
            document.body.removeAttribute("data-silence-dialog-open");
        };
    }, [open]);

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

    /**
     * 滑块滚轮步进：向上 / 向下各走一步；按住“精细调整”修饰键时步长为 1，
     * 否则用该控件的粗步长。与 TransposeCentsDialog 等编辑对话框同款手势。
     * 阻止默认滚动由 Dialog.Content 上的原生非被动守卫完成（React onWheel
     * 的 preventDefault 是 passive no-op，见 useWheelScrollGuard）。
     */
    const wheelDelta = (e: React.WheelEvent<HTMLInputElement>, coarse: number): number => {
        const fine = isModifierActive(paramFineAdjustKb, e.nativeEvent);
        return (e.deltaY < 0 ? 1 : -1) * (fine ? 1 : coarse);
    };

    const clampRange = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));

    // 双击参数行 = 该参数重置回默认值（提示文案挂在行 title 上）。
    const resetHint = tAny("silence_double_click_reset");

    return (
        <Dialog.Root
            open={open}
            onOpenChange={(next) => {
                if (!next) closeAndCleanup();
            }}
        >
            <Dialog.Content
                ref={wheelGuard}
                style={{ maxWidth: 460 }}
                onKeyDown={(e) => e.stopPropagation()}
            >
                <Dialog.Title>{tAny("ctx_silence_detection")}</Dialog.Title>

                <Flex direction="column" gap="3" mt="3">
                    <Flex
                        align="center"
                        gap="2"
                        title={resetHint}
                        onDoubleClick={() => update({ method: SILENCE_DETECT_DEFAULTS.method })}
                    >
                        <Text size="2" style={{ minWidth: 96 }}>
                            {tAny("silence_method")}
                        </Text>
                        <Select.Root
                            size="1"
                            value={options.method}
                            onValueChange={(v) => update({ method: v as "rms" | "peak" })}
                        >
                            <Select.Trigger
                                style={{ flex: 1 }}
                                onWheel={(e) =>
                                    applySelectWheelChange({
                                        event: e,
                                        currentValue: options.method,
                                        options: ["rms", "peak"] as const,
                                        onChange: (v) => update({ method: v }),
                                    })
                                }
                            />
                            <Select.Content>
                                <Select.Item value="rms">{tAny("silence_method_rms")}</Select.Item>
                                <Select.Item value="peak">
                                    {tAny("silence_method_peak")}
                                </Select.Item>
                            </Select.Content>
                        </Select.Root>
                    </Flex>

                    <Flex
                        align="center"
                        gap="2"
                        title={resetHint}
                        onDoubleClick={() =>
                            update({ thresholdDb: SILENCE_DETECT_DEFAULTS.thresholdDb })
                        }
                    >
                        <Text size="2" style={{ minWidth: 96 }}>
                            {tAny("silence_threshold")}
                        </Text>
                        <input
                            type="range"
                            min={-96}
                            max={-6}
                            step={1}
                            value={Math.round(options.thresholdDb)}
                            disabled={options.adaptive}
                            onChange={(e) => update({ thresholdDb: Number(e.target.value) })}
                            onWheel={(e) => {
                                if (options.adaptive) return;
                                const delta = wheelDelta(e, 3);
                                update({
                                    thresholdDb: clampRange(
                                        Math.round(options.thresholdDb) + delta,
                                        -96,
                                        -6,
                                    ),
                                });
                            }}
                            style={{ flex: 1 }}
                        />
                        <Text size="1" style={{ minWidth: 56, textAlign: "right" }}>
                            {options.adaptive
                                ? tAny("silence_adaptive_short")
                                : `${Math.round(options.thresholdDb)} dB`}
                        </Text>
                    </Flex>

                    <label
                        className="flex items-center gap-2 text-[12px]"
                        title={resetHint}
                        onDoubleClick={() => update({ adaptive: SILENCE_DETECT_DEFAULTS.adaptive })}
                    >
                        <Checkbox
                            checked={options.adaptive}
                            onCheckedChange={(v) => update({ adaptive: v === true })}
                        />
                        {tAny("silence_adaptive")}
                    </label>

                    {(
                        [
                            ["minSilenceMs", "silence_min_silence", 10, 2000],
                            ["minSoundMs", "silence_min_sound", 0, 500],
                            ["paddingMs", "silence_padding", 0, 200],
                            ["cutFadeMs", "silence_cut_fade", 0, 100],
                        ] as const
                    ).map(([key, labelKey, min, max]) => (
                        <Flex
                            key={key}
                            align="center"
                            gap="2"
                            title={resetHint}
                            onDoubleClick={() =>
                                update({
                                    [key]: SILENCE_DETECT_DEFAULTS[key],
                                } as Partial<SilenceDetectSettings>)
                            }
                        >
                            <Text size="2" style={{ minWidth: 96 }}>
                                {tAny(labelKey)}
                            </Text>
                            <input
                                type="range"
                                min={min}
                                max={max}
                                step={1}
                                value={Math.round(options[key])}
                                onChange={(e) => update({ [key]: Number(e.target.value) })}
                                onWheel={(e) => {
                                    const coarse = key === "cutFadeMs" ? 5 : 10;
                                    const delta = wheelDelta(e, coarse);
                                    update({
                                        [key]: clampRange(
                                            Math.round(options[key]) + delta,
                                            min,
                                            max,
                                        ),
                                    });
                                }}
                                style={{ flex: 1 }}
                            />
                            <Text size="1" style={{ minWidth: 52, textAlign: "right" }}>
                                {Math.round(options[key])} ms
                            </Text>
                        </Flex>
                    ))}

                    <Flex
                        align="center"
                        gap="2"
                        title={resetHint}
                        onDoubleClick={() => update({ action: SILENCE_DETECT_DEFAULTS.action })}
                    >
                        <Text size="2" style={{ minWidth: 96 }}>
                            {tAny("silence_action")}
                        </Text>
                        <Select.Root
                            size="1"
                            value={options.action}
                            onValueChange={(v) =>
                                update({ action: v as SilenceDetectSettings["action"] })
                            }
                        >
                            <Select.Trigger
                                style={{ flex: 1 }}
                                onWheel={(e) =>
                                    applySelectWheelChange({
                                        event: e,
                                        currentValue: options.action,
                                        options: ["close", "keep", "split"] as const,
                                        onChange: (v) => update({ action: v }),
                                    })
                                }
                            />
                            <Select.Content>
                                <Select.Item value="close">
                                    {tAny("silence_action_close")}
                                </Select.Item>
                                <Select.Item value="keep">
                                    {tAny("silence_action_keep")}
                                </Select.Item>
                                <Select.Item value="split">
                                    {tAny("silence_action_split")}
                                </Select.Item>
                            </Select.Content>
                        </Select.Root>
                    </Flex>

                    <label
                        className="flex items-center gap-2 text-[12px]"
                        title={resetHint}
                        onDoubleClick={() =>
                            update({ deleteSilentClips: SILENCE_DETECT_DEFAULTS.deleteSilentClips })
                        }
                    >
                        <Checkbox
                            checked={options.deleteSilentClips}
                            onCheckedChange={(v) => update({ deleteSilentClips: v === true })}
                        />
                        {tAny("silence_delete_silent_clips")}
                    </label>
                    <label
                        className="flex items-center gap-2 text-[12px]"
                        title={resetHint}
                        onDoubleClick={() =>
                            update({ syncAllTakes: SILENCE_DETECT_DEFAULTS.syncAllTakes })
                        }
                    >
                        <Checkbox
                            checked={options.syncAllTakes}
                            onCheckedChange={(v) => update({ syncAllTakes: v === true })}
                        />
                        {tAny("silence_sync_all_takes")}
                    </label>

                    {/* 预览摘要（与时间线上的红色覆盖层联动） */}
                    <Flex
                        direction="column"
                        gap="1"
                        className="rounded border border-qt-border p-2"
                    >
                        <Text size="1" className="text-qt-text-muted">
                            {analyzing
                                ? tAny("silence_analyzing")
                                : totalRegions > 0
                                  ? tAny("silence_preview_summary")
                                        .replace("{n}", String(totalRegions))
                                        .replace("{dur}", totalSilent.toFixed(2))
                                  : tAny("silence_no_silence")}
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
                                        ? tAny("silence_skipped") +
                                          (report.message ? ` (${report.message})` : "")
                                        : regions > 0
                                          ? tAny("silence_preview_clip")
                                                .replace("{n}", String(regions))
                                                .replace(
                                                    "{dur}",
                                                    (report?.totalSilentSec ?? 0).toFixed(2),
                                                )
                                          : tAny("silence_no_silence")}
                                </Text>
                            );
                        })}
                        {clipIds.length > 6 ? (
                            <Text size="1" className="text-qt-text-muted">
                                {tAny("silence_more_clips").replace(
                                    "{n}",
                                    String(clipIds.length - 6),
                                )}
                            </Text>
                        ) : null}
                    </Flex>
                </Flex>

                <Flex gap="3" mt="4" justify="end">
                    <Button variant="soft" color="gray" onClick={closeAndCleanup}>
                        {tAny("cancel")}
                    </Button>
                    <Button
                        variant="solid"
                        onClick={() => void handleApply()}
                        disabled={applying || clipIds.length === 0}
                    >
                        {applying ? tAny("silence_applying") : tAny("silence_apply")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
};
