import { useState, useEffect, useMemo } from "react";
import { Flex, Text, TextField, Select } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import type { ScaleKey } from "../../utils/musicalScales";
import { useAppSelector } from "../../app/hooks";
import { isModifierActive, selectKeybinding } from "../../features/keybindings/keybindingsSlice";
import { applySelectWheelChange } from "../../utils/selectWheel";
import { useWheelScrollGuard } from "../../utils/useWheelScrollGuard";
import { buildScaleSelectGroups } from "../../utils/scaleSelection";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    defaultSmoothness?: number;
    onConfirm?: (cents: number, edgeSmoothnessPercent: number) => void;
}

export function TransposeCentsDialog({
    open,
    onOpenChange,
    defaultSmoothness = 0,
    onConfirm,
}: Props) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [cents, setCents] = useState("0");
    const [smoothness, setSmoothness] = useState(String(Math.round(defaultSmoothness)));
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );

    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect -- 对话框打开时用 props 初始化局部 state（既有模式；重构会改变打开时序）
        if (open) setSmoothness(String(Math.round(defaultSmoothness)));
    }, [open, defaultSmoothness]);

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_transpose_cents")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(
                            Number(cents) || 0,
                            Math.max(0, Math.min(100, Number(smoothness) || 0)),
                        );
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={tAny("dlg_cents")}>
                    <TextField.Root
                        size="2"
                        type="number"
                        value={cents}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setCents(e.target.value)
                        }
                    />
                </AppField>
                <AppField label={tAny("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <input
                            type="range"
                            min={0}
                            max={100}
                            step={1}
                            value={Math.round(Number(smoothness) || 0)}
                            onWheel={(e) => {
                                e.preventDefault();
                                const fine = isModifierActive(paramFineAdjustKb, e.nativeEvent);
                                const step = fine ? 1 : 5;
                                const dir = e.deltaY < 0 ? 1 : -1;
                                const current = Math.round(Number(smoothness) || 0);
                                const next = Math.max(0, Math.min(100, current + dir * step));
                                setSmoothness(String(next));
                            }}
                            onChange={(e) => setSmoothness(e.currentTarget.value)}
                            style={{ flex: 1 }}
                        />
                        <Text size="1" style={{ minWidth: 40, textAlign: "right" }}>
                            {Math.round(Number(smoothness) || 0)}%
                        </Text>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface TransposeDegreesProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    defaultScale?: ScaleKey;
    defaultUseProjectScale?: boolean;
    projectScaleLabel?: string;
    defaultSmoothness?: number;
    onConfirm?: (degrees: number, scaleValue: string, edgeSmoothnessPercent: number) => void;
}

export function TransposeDegreesDialog({
    open,
    onOpenChange,
    defaultScale = "C",
    defaultUseProjectScale = true,
    projectScaleLabel,
    defaultSmoothness = 0,
    onConfirm,
}: TransposeDegreesProps) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [degrees, setDegrees] = useState("3");
    const [scaleValue, setScaleValue] = useState<string>(
        defaultUseProjectScale ? "__project__" : defaultScale,
    );
    const [smoothness, setSmoothness] = useState(String(Math.round(defaultSmoothness)));
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    const customScalePresets = useAppSelector((state) => state.session.customScalePresets);
    const scaleSelectGroups = useMemo(
        () =>
            buildScaleSelectGroups(
                projectScaleLabel ?? tAny("project_scale_generic"),
                customScalePresets,
            ),
        [projectScaleLabel, customScalePresets, tAny],
    );

    useEffect(() => {
        if (open) {
            // eslint-disable-next-line react-hooks/set-state-in-effect -- 对话框打开时用 props 初始化局部 state（既有模式；重构会改变打开时序）
            setScaleValue(defaultUseProjectScale ? "__project__" : defaultScale);
            setSmoothness(String(Math.round(defaultSmoothness)));
        }
    }, [open, defaultScale, defaultSmoothness, defaultUseProjectScale]);

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_transpose_degrees")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(
                            Number(degrees) || 0,
                            scaleValue,
                            Math.max(0, Math.min(100, Number(smoothness) || 0)),
                        );
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={tAny("transpose_degrees_amount")}>
                    <TextField.Root
                        size="2"
                        type="number"
                        value={degrees}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setDegrees(e.target.value)
                        }
                    />
                </AppField>
                <AppField label={tAny("base_scale")}>
                    <Select.Root value={scaleValue} size="2" onValueChange={setScaleValue}>
                        <Select.Trigger
                            onWheel={(event) => {
                                applySelectWheelChange({
                                    event,
                                    currentValue: scaleValue,
                                    options: scaleSelectGroups.wheelOptions,
                                    onChange: setScaleValue,
                                });
                            }}
                        />
                        <Select.Content>
                            <Select.Item value={scaleSelectGroups.projectOption.value}>
                                {scaleSelectGroups.projectOption.label}
                            </Select.Item>
                            <Select.Separator />
                            {scaleSelectGroups.builtinOptions.map((option) => (
                                <Select.Item key={option.value} value={option.value}>
                                    {option.label}
                                </Select.Item>
                            ))}
                            {scaleSelectGroups.customOptions.length > 0 && <Select.Separator />}
                            {scaleSelectGroups.customOptions.map((option) => (
                                <Select.Item key={option.value} value={option.value}>
                                    {option.label}
                                </Select.Item>
                            ))}
                        </Select.Content>
                    </Select.Root>
                </AppField>
                <AppField label={tAny("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <input
                            type="range"
                            min={0}
                            max={100}
                            step={1}
                            value={Math.round(Number(smoothness) || 0)}
                            onWheel={(e) => {
                                e.preventDefault();
                                const fine = isModifierActive(paramFineAdjustKb, e.nativeEvent);
                                const step = fine ? 1 : 5;
                                const dir = e.deltaY < 0 ? 1 : -1;
                                const current = Math.round(Number(smoothness) || 0);
                                const next = Math.max(0, Math.min(100, current + dir * step));
                                setSmoothness(String(next));
                            }}
                            onChange={(e) => setSmoothness(e.currentTarget.value)}
                            style={{ flex: 1 }}
                        />
                        <Text size="1" style={{ minWidth: 40, textAlign: "right" }}>
                            {Math.round(Number(smoothness) || 0)}%
                        </Text>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface SetPitchProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    titleText?: string;
    valueLabelText?: string;
    defaultValue?: number;
    defaultSmoothness?: number;
    onConfirm?: (value: number, edgeSmoothnessPercent: number) => void;
}

export function SetPitchDialog({
    open,
    onOpenChange,
    titleText,
    valueLabelText,
    defaultValue = 60,
    defaultSmoothness = 0,
    onConfirm,
}: SetPitchProps) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [note, setNote] = useState(String(defaultValue));
    const [smoothness, setSmoothness] = useState(String(Math.round(defaultSmoothness)));
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );

    useEffect(() => {
        if (open) {
            // eslint-disable-next-line react-hooks/set-state-in-effect -- 对话框打开时用 props 初始化局部 state（既有模式；重构会改变打开时序）
            setSmoothness(String(Math.round(defaultSmoothness)));
            setNote(String(defaultValue));
        }
    }, [open, defaultSmoothness, defaultValue]);

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={titleText ?? tAny("menu_set_pitch")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        const parsedNote = Number(note);
                        const nextValue = Number.isFinite(parsedNote) ? parsedNote : defaultValue;
                        onConfirm?.(nextValue, Math.max(0, Math.min(100, Number(smoothness) || 0)));
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={valueLabelText ?? tAny("dlg_midi_note")}>
                    <TextField.Root
                        size="2"
                        type="number"
                        value={note}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setNote(e.target.value)
                        }
                    />
                </AppField>
                <AppField label={tAny("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <input
                            type="range"
                            min={0}
                            max={100}
                            step={1}
                            value={Math.round(Number(smoothness) || 0)}
                            onWheel={(e) => {
                                e.preventDefault();
                                const fine = isModifierActive(paramFineAdjustKb, e.nativeEvent);
                                const step = fine ? 1 : 5;
                                const dir = e.deltaY < 0 ? 1 : -1;
                                const current = Math.round(Number(smoothness) || 0);
                                const next = Math.max(0, Math.min(100, current + dir * step));
                                setSmoothness(String(next));
                            }}
                            onChange={(e) => setSmoothness(e.currentTarget.value)}
                            style={{ flex: 1 }}
                        />
                        <Text size="1" style={{ minWidth: 40, textAlign: "right" }}>
                            {Math.round(Number(smoothness) || 0)}%
                        </Text>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface AverageProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onConfirm?: (strength: number) => void;
}

export function AverageDialog({ open, onOpenChange, onConfirm }: AverageProps) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [strength, setStrength] = useState("100");
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_average")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(Math.max(0, Math.min(100, Math.round(Number(strength) || 0))));
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={tAny("dlg_average_strength")}>
                    <Flex align="center" gap="2">
                        <input
                            type="range"
                            min={0}
                            max={100}
                            step={1}
                            value={Math.round(Number(strength) || 0)}
                            onWheel={(e) => {
                                e.preventDefault();
                                const fine = isModifierActive(paramFineAdjustKb, e.nativeEvent);
                                const step = fine ? 1 : 5;
                                const dir = e.deltaY < 0 ? 1 : -1;
                                const current = Math.round(Number(strength) || 0);
                                const next = Math.max(0, Math.min(100, current + dir * step));
                                setStrength(String(next));
                            }}
                            onChange={(e) => {
                                setStrength(e.currentTarget.value);
                            }}
                            style={{ flex: 1 }}
                        />
                        <Text size="1" style={{ minWidth: 40, textAlign: "right" }}>
                            {Math.round(Number(strength) || 0)}%
                        </Text>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface SmoothProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    defaultSmoothness?: number;
    onConfirm?: (strength: number) => void;
}

export function SmoothDialog({
    open,
    onOpenChange,
    defaultSmoothness = 50,
    onConfirm,
}: SmoothProps) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [strength, setStrength] = useState(50);
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );

    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect -- 对话框打开时用 props 初始化局部 state（既有模式；重构会改变打开时序）
        if (open) setStrength(Math.max(0, Math.min(100, Math.round(defaultSmoothness))));
    }, [open, defaultSmoothness]);

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_smooth")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(Math.max(0, Math.min(100, Math.round(strength))));
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={tAny("dlg_smoothness")}>
                    <Flex align="center" gap="2">
                        <input
                            type="range"
                            min={0}
                            max={100}
                            step={1}
                            value={Math.round(strength)}
                            onWheel={(e) => {
                                e.preventDefault();
                                const fine = isModifierActive(paramFineAdjustKb, e.nativeEvent);
                                const step = fine ? 1 : 5;
                                const dir = e.deltaY < 0 ? 1 : -1;
                                const next = Math.max(
                                    0,
                                    Math.min(100, Math.round(strength) + dir * step),
                                );
                                setStrength(next);
                            }}
                            onChange={(e) => {
                                setStrength(Number(e.currentTarget.value) || 0);
                            }}
                            style={{ flex: 1 }}
                        />
                        <Text size="1" style={{ minWidth: 40, textAlign: "right" }}>
                            {Math.round(strength)}%
                        </Text>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface VibratoProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    editParam?: string;
    /** 当前参数的值域（用于自动钳制振幅默认值） */
    paramRange?: { min: number; max: number };
    onConfirm?: (
        amplitude: number,
        rate: number,
        attack: number,
        release: number,
        phase: number,
    ) => void;
}

export function VibratoDialog({
    open,
    onOpenChange,
    onConfirm,
    editParam,
    paramRange,
}: VibratoProps) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const isPitch = editParam === "pitch";
    // 气声音量（breath_gain，0..2）按原值域给小振幅；dyn 落到默认 30 ——
    // 它在 op 侧是**深度百分比**（±30% 乘性调制，静音保持静音）。
    // 【注意】不能按"值域 0..2"来识别 breath_gain：dyn 的值域也是 0..2。
    const isBreathGain = editParam === "breath_gain";
    const defaultAmplitude = isPitch ? "30" : isBreathGain ? "1" : "30";

    const [amplitude, setAmplitude] = useState(defaultAmplitude);
    const [rate, setRate] = useState("5.5");
    const [attack, setAttack] = useState("50");
    const [release, setRelease] = useState("50");
    const [phase, setPhase] = useState("0");

    // 对话框常驻挂载（useState 初始值只在首挂载生效）：每次打开必须按
    // 当前参数重置默认值，否则对 breath_gain 打开时仍显示 pitch 的 30。
    useEffect(() => {
        if (open) {
            setAmplitude(defaultAmplitude);
            setRate("5.5");
            setAttack("50");
            setRelease("50");
            setPhase("0");
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 打开时按最新参数重置
    }, [open, editParam, paramRange]);

    // 兜底 NaN/Infinity 而不吞掉合法的 0（`Number(x) || default` 会把 0
    // 替换成默认值——对幅度/速率/相位，0 都是合法输入）。
    const parseNumberOr = (raw: string, fallback: number): number => {
        const value = Number(raw);
        return Number.isFinite(value) ? value : fallback;
    };

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_add_vibrato")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(
                            parseNumberOr(amplitude, 30),
                            parseNumberOr(rate, 5.5),
                            parseNumberOr(attack, 50),
                            parseNumberOr(release, 50),
                            parseNumberOr(phase, 0),
                        );
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={isPitch ? tAny("dlg_amplitude_cents") : tAny("dlg_amplitude")}>
                    <TextField.Root
                        size="2"
                        type="number"
                        value={amplitude}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setAmplitude(e.target.value)
                        }
                    />
                </AppField>
                <AppField label={tAny("dlg_rate_hz")}>
                    <TextField.Root
                        size="2"
                        type="number"
                        value={rate}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setRate(e.target.value)
                        }
                    />
                </AppField>
                <AppField label={tAny("dlg_attack_ms")}>
                    <TextField.Root
                        size="2"
                        type="number"
                        value={attack}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setAttack(e.target.value)
                        }
                    />
                </AppField>
                <AppField label={tAny("dlg_release_ms")}>
                    <TextField.Root
                        size="2"
                        type="number"
                        value={release}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setRelease(e.target.value)
                        }
                    />
                </AppField>
                <AppField label={tAny("dlg_phase_deg")}>
                    <TextField.Root
                        size="2"
                        type="number"
                        value={phase}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setPhase(e.target.value)
                        }
                    />
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface QuantizeProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    valueMode?: boolean;
    defaultQuantizeUnit?: number;
    defaultTolerance?: number;
    defaultScale?: ScaleKey;
    defaultUseProjectScale?: boolean;
    projectScaleLabel?: string;
    defaultToleranceCents?: number;
    defaultSmoothness?: number;
    onConfirm?: (
        unit: "semitone" | "scale" | "value",
        scaleValue: string,
        toleranceCents: number,
        quantizeUnit: number | undefined,
        edgeSmoothnessPercent: number,
    ) => void;
}

export function QuantizeDialog({
    open,
    onOpenChange,
    valueMode = false,
    defaultQuantizeUnit = 1,
    defaultTolerance = 0,
    defaultScale = "C",
    defaultUseProjectScale = true,
    projectScaleLabel,
    defaultToleranceCents = 0,
    defaultSmoothness = 0,
    onConfirm,
}: QuantizeProps) {
    // 滚轮守卫：滑块滚轮步进时阻止祖先容器滚动（见 useWheelScrollGuard）。
    const quantizeWheelGuard = useWheelScrollGuard<HTMLDivElement>('input[type="range"]');
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const toleranceDefault = defaultTolerance ?? defaultToleranceCents;
    const [unit, setUnit] = useState<"semitone" | "scale">("semitone");
    const [scaleValue, setScaleValue] = useState<string>(
        defaultUseProjectScale ? "__project__" : defaultScale,
    );
    const customScalePresets = useAppSelector((state) => state.session.customScalePresets);
    const scaleSelectGroups = useMemo(
        () =>
            buildScaleSelectGroups(
                projectScaleLabel ?? tAny("project_scale_generic"),
                customScalePresets,
            ),
        [projectScaleLabel, customScalePresets, tAny],
    );
    const [toleranceCents, setToleranceCents] = useState<string>(String(toleranceDefault));
    const [quantizeUnit, setQuantizeUnit] = useState<string>(String(defaultQuantizeUnit));
    const [smoothness, setSmoothness] = useState(String(Math.round(defaultSmoothness)));
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );

    useEffect(() => {
        if (open) {
            // eslint-disable-next-line react-hooks/set-state-in-effect -- 对话框打开时用 props 初始化局部 state（既有模式；重构会改变打开时序）
            setScaleValue(defaultUseProjectScale ? "__project__" : defaultScale);
            setToleranceCents(String(toleranceDefault));
            setQuantizeUnit(String(defaultQuantizeUnit));
            setSmoothness(String(Math.round(defaultSmoothness)));
        }
    }, [
        open,
        defaultScale,
        toleranceDefault,
        defaultUseProjectScale,
        defaultQuantizeUnit,
        defaultSmoothness,
    ]);

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_quantize")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        const parsed = Math.abs(Math.round(Number(toleranceCents) || 0));
                        const parsedUnit = Math.abs(Number(quantizeUnit) || 0);
                        onConfirm?.(
                            valueMode ? "value" : unit,
                            scaleValue,
                            parsed,
                            valueMode ? parsedUnit : undefined,
                            Math.max(0, Math.min(100, Number(smoothness) || 0)),
                        );
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <div ref={quantizeWheelGuard}>
                <AppForm>
                    {!valueMode && (
                        <AppField label={tAny("quantize_unit")}>
                            <Select.Root
                                value={unit}
                                size="2"
                                onValueChange={(v) => setUnit(v as "semitone" | "scale")}
                            >
                                <Select.Trigger
                                    onWheel={(event) => {
                                        applySelectWheelChange({
                                            event,
                                            currentValue: unit,
                                            options: ["semitone", "scale"],
                                            onChange: (next) =>
                                                setUnit(next as "semitone" | "scale"),
                                        });
                                    }}
                                />
                                <Select.Content>
                                    <Select.Item value="semitone">
                                        {tAny("quantize_semitone")}
                                    </Select.Item>
                                    <Select.Item value="scale">
                                        {tAny("quantize_scale")}
                                    </Select.Item>
                                </Select.Content>
                            </Select.Root>
                        </AppField>
                    )}
                    {!valueMode && unit === "scale" && (
                        <AppField label={tAny("base_scale")}>
                            <Select.Root value={scaleValue} size="2" onValueChange={setScaleValue}>
                                <Select.Trigger
                                    onWheel={(event) => {
                                        applySelectWheelChange({
                                            event,
                                            currentValue: scaleValue,
                                            options: scaleSelectGroups.wheelOptions,
                                            onChange: setScaleValue,
                                        });
                                    }}
                                />
                                <Select.Content>
                                    <Select.Item value={scaleSelectGroups.projectOption.value}>
                                        {scaleSelectGroups.projectOption.label}
                                    </Select.Item>
                                    <Select.Separator />
                                    {scaleSelectGroups.builtinOptions.map((option) => (
                                        <Select.Item key={option.value} value={option.value}>
                                            {option.label}
                                        </Select.Item>
                                    ))}
                                    {scaleSelectGroups.customOptions.length > 0 && (
                                        <Select.Separator />
                                    )}
                                    {scaleSelectGroups.customOptions.map((option) => (
                                        <Select.Item key={option.value} value={option.value}>
                                            {option.label}
                                        </Select.Item>
                                    ))}
                                </Select.Content>
                            </Select.Root>
                        </AppField>
                    )}
                    {valueMode && (
                        <AppField label={tAny("quantize_unit")}>
                            <TextField.Root
                                size="2"
                                type="number"
                                value={quantizeUnit}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                    setQuantizeUnit(e.target.value)
                                }
                            />
                        </AppField>
                    )}
                    <AppField
                        label={
                            valueMode ? tAny("quantize_tolerance") : tAny("pitch_snap_tolerance")
                        }
                    >
                        <TextField.Root
                            size="2"
                            type="number"
                            value={toleranceCents}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setToleranceCents(e.target.value)
                            }
                        />
                    </AppField>
                    <AppField label={tAny("edge_smoothness")}>
                        <Flex align="center" gap="2">
                            <input
                                type="range"
                                min={0}
                                max={100}
                                step={1}
                                value={Math.round(Number(smoothness) || 0)}
                                onWheel={(e) => {
                                    // 阻止默认滚动由 Dialog.Content 上的原生非被动
                                    // 守卫完成（React onWheel 的 preventDefault 是
                                    // no-op，见 useWheelScrollGuard）。
                                    const fine = isModifierActive(paramFineAdjustKb, e.nativeEvent);
                                    const step = fine ? 1 : 5;
                                    const dir = e.deltaY < 0 ? 1 : -1;
                                    const current = Math.round(Number(smoothness) || 0);
                                    const next = Math.max(0, Math.min(100, current + dir * step));
                                    setSmoothness(String(next));
                                }}
                                onChange={(e) => setSmoothness(e.currentTarget.value)}
                                style={{ flex: 1 }}
                            />
                            <Text size="1" style={{ minWidth: 40, textAlign: "right" }}>
                                {Math.round(Number(smoothness) || 0)}%
                            </Text>
                        </Flex>
                    </AppField>
                </AppForm>
            </div>
        </AppDialog>
    );
}

interface MeanQuantizeProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    valueMode?: boolean;
    defaultQuantizeUnit?: number;
    defaultTolerance?: number;
    defaultScale?: ScaleKey;
    defaultUseProjectScale?: boolean;
    projectScaleLabel?: string;
    defaultToleranceCents?: number;
    defaultSmoothness?: number;
    onConfirm?: (
        unit: "semitone" | "scale" | "value",
        scaleValue: string,
        toleranceCents: number,
        quantizeUnit: number | undefined,
        edgeSmoothnessPercent: number,
    ) => void;
}

export function MeanQuantizeDialog({
    open,
    onOpenChange,
    valueMode = false,
    defaultQuantizeUnit = 1,
    defaultTolerance = 0,
    defaultScale = "C",
    defaultUseProjectScale = true,
    projectScaleLabel,
    defaultToleranceCents = 0,
    defaultSmoothness = 0,
    onConfirm,
}: MeanQuantizeProps) {
    // 滚轮守卫：滑块滚轮步进时阻止祖先容器滚动（见 useWheelScrollGuard）。
    const meanQuantizeWheelGuard = useWheelScrollGuard<HTMLDivElement>('input[type="range"]');
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const toleranceDefault = defaultTolerance ?? defaultToleranceCents;
    const [unit, setUnit] = useState<"semitone" | "scale">("semitone");
    const [scaleValue, setScaleValue] = useState<string>(
        defaultUseProjectScale ? "__project__" : defaultScale,
    );
    const customScalePresets = useAppSelector((state) => state.session.customScalePresets);
    const scaleSelectGroups = useMemo(
        () =>
            buildScaleSelectGroups(
                projectScaleLabel ?? tAny("project_scale_generic"),
                customScalePresets,
            ),
        [projectScaleLabel, customScalePresets, tAny],
    );
    const [toleranceCents, setToleranceCents] = useState<string>(String(toleranceDefault));
    const [quantizeUnit, setQuantizeUnit] = useState<string>(String(defaultQuantizeUnit));
    const [smoothness, setSmoothness] = useState(String(Math.round(defaultSmoothness)));
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );

    useEffect(() => {
        if (open) {
            // eslint-disable-next-line react-hooks/set-state-in-effect -- 对话框打开时用 props 初始化局部 state（既有模式；重构会改变打开时序）
            setScaleValue(defaultUseProjectScale ? "__project__" : defaultScale);
            setToleranceCents(String(toleranceDefault));
            setQuantizeUnit(String(defaultQuantizeUnit));
            setSmoothness(String(Math.round(defaultSmoothness)));
        }
    }, [
        open,
        defaultScale,
        toleranceDefault,
        defaultUseProjectScale,
        defaultQuantizeUnit,
        defaultSmoothness,
    ]);

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("mean_quantize_title")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        const parsed = Math.abs(Math.round(Number(toleranceCents) || 0));
                        const parsedUnit = Math.abs(Number(quantizeUnit) || 0);
                        onConfirm?.(
                            valueMode ? "value" : unit,
                            scaleValue,
                            parsed,
                            valueMode ? parsedUnit : undefined,
                            Math.max(0, Math.min(100, Number(smoothness) || 0)),
                        );
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <div ref={meanQuantizeWheelGuard}>
                <AppForm>
                    {!valueMode && (
                        <AppField label={tAny("quantize_unit")}>
                            <Select.Root
                                value={unit}
                                size="2"
                                onValueChange={(v) => setUnit(v as "semitone" | "scale")}
                            >
                                <Select.Trigger
                                    onWheel={(event) => {
                                        applySelectWheelChange({
                                            event,
                                            currentValue: unit,
                                            options: ["semitone", "scale"],
                                            onChange: (next) =>
                                                setUnit(next as "semitone" | "scale"),
                                        });
                                    }}
                                />
                                <Select.Content>
                                    <Select.Item value="semitone">
                                        {tAny("quantize_semitone")}
                                    </Select.Item>
                                    <Select.Item value="scale">
                                        {tAny("quantize_scale")}
                                    </Select.Item>
                                </Select.Content>
                            </Select.Root>
                        </AppField>
                    )}
                    {!valueMode && unit === "scale" && (
                        <AppField label={tAny("base_scale")}>
                            <Select.Root value={scaleValue} size="2" onValueChange={setScaleValue}>
                                <Select.Trigger
                                    onWheel={(event) => {
                                        applySelectWheelChange({
                                            event,
                                            currentValue: scaleValue,
                                            options: scaleSelectGroups.wheelOptions,
                                            onChange: setScaleValue,
                                        });
                                    }}
                                />
                                <Select.Content>
                                    <Select.Item value={scaleSelectGroups.projectOption.value}>
                                        {scaleSelectGroups.projectOption.label}
                                    </Select.Item>
                                    <Select.Separator />
                                    {scaleSelectGroups.builtinOptions.map((option) => (
                                        <Select.Item key={option.value} value={option.value}>
                                            {option.label}
                                        </Select.Item>
                                    ))}
                                    {scaleSelectGroups.customOptions.length > 0 && (
                                        <Select.Separator />
                                    )}
                                    {scaleSelectGroups.customOptions.map((option) => (
                                        <Select.Item key={option.value} value={option.value}>
                                            {option.label}
                                        </Select.Item>
                                    ))}
                                </Select.Content>
                            </Select.Root>
                        </AppField>
                    )}
                    {valueMode && (
                        <AppField label={tAny("quantize_unit")}>
                            <TextField.Root
                                size="2"
                                type="number"
                                value={quantizeUnit}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                    setQuantizeUnit(e.target.value)
                                }
                            />
                        </AppField>
                    )}
                    <AppField
                        label={
                            valueMode ? tAny("quantize_tolerance") : tAny("pitch_snap_tolerance")
                        }
                    >
                        <TextField.Root
                            size="2"
                            type="number"
                            value={toleranceCents}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setToleranceCents(e.target.value)
                            }
                        />
                    </AppField>
                    <AppField label={tAny("edge_smoothness")}>
                        <Flex align="center" gap="2">
                            <input
                                type="range"
                                min={0}
                                max={100}
                                step={1}
                                value={Math.round(Number(smoothness) || 0)}
                                onWheel={(e) => {
                                    // 阻止默认滚动由 Dialog.Content 上的原生非被动
                                    // 守卫完成（React onWheel 的 preventDefault 是
                                    // no-op，见 useWheelScrollGuard）。
                                    const fine = isModifierActive(paramFineAdjustKb, e.nativeEvent);
                                    const step = fine ? 1 : 5;
                                    const dir = e.deltaY < 0 ? 1 : -1;
                                    const current = Math.round(Number(smoothness) || 0);
                                    const next = Math.max(0, Math.min(100, current + dir * step));
                                    setSmoothness(String(next));
                                }}
                                onChange={(e) => setSmoothness(e.currentTarget.value)}
                                style={{ flex: 1 }}
                            />
                            <Text size="1" style={{ minWidth: 40, textAlign: "right" }}>
                                {Math.round(Number(smoothness) || 0)}%
                            </Text>
                        </Flex>
                    </AppField>
                </AppForm>
            </div>
        </AppDialog>
    );
}
