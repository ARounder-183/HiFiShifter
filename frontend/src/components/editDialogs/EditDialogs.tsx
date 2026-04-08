import { useState, useEffect, useMemo } from "react";
import { Dialog, Flex, Text, TextField, Button, Select } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import type { ScaleKey } from "../../utils/musicalScales";
import { useAppSelector } from "../../app/hooks";
import { isModifierActive, selectKeybinding } from "../../features/keybindings/keybindingsSlice";
import { applySelectWheelChange } from "../../utils/selectWheel";
import { buildScaleSelectGroups } from "../../utils/scaleSelection";

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
        if (open) setSmoothness(String(Math.round(defaultSmoothness)));
    }, [open, defaultSmoothness]);

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 340 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("menu_transpose_cents")}</Dialog.Title>
                <Flex direction="column" gap="3" mt="3">
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 80 }}>
                            {tAny("dlg_cents")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            value={cents}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setCents(e.target.value)
                            }
                            style={{ flex: 1 }}
                        />
                    </Flex>
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 80 }}>
                            {tAny("edge_smoothness")}
                        </Text>
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
                </Flex>
                <Flex justify="end" gap="2" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("cancel")}
                        </Button>
                    </Dialog.Close>
                    <Button
                        onClick={() => {
                            onConfirm?.(
                                Number(cents) || 0,
                                Math.max(0, Math.min(100, Number(smoothness) || 0)),
                            );
                            onOpenChange(false);
                        }}
                    >
                        {tAny("ok")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
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
            setScaleValue(defaultUseProjectScale ? "__project__" : defaultScale);
            setSmoothness(String(Math.round(defaultSmoothness)));
        }
    }, [open, defaultScale, defaultSmoothness, defaultUseProjectScale]);

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 360 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("menu_transpose_degrees")}</Dialog.Title>
                <Flex direction="column" gap="3" mt="3">
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 80 }}>
                            {tAny("transpose_degrees_amount")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            value={degrees}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setDegrees(e.target.value)
                            }
                            style={{ flex: 1 }}
                        />
                    </Flex>
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 80 }}>
                            {tAny("base_scale")}
                        </Text>
                        <Select.Root value={scaleValue} size="2" onValueChange={setScaleValue}>
                            <Select.Trigger
                                style={{ flex: 1 }}
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
                    </Flex>
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 80 }}>
                            {tAny("edge_smoothness")}
                        </Text>
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
                </Flex>
                <Flex justify="end" gap="2" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("cancel")}
                        </Button>
                    </Dialog.Close>
                    <Button
                        onClick={() => {
                            onConfirm?.(
                                Number(degrees) || 0,
                                scaleValue,
                                Math.max(0, Math.min(100, Number(smoothness) || 0)),
                            );
                            onOpenChange(false);
                        }}
                    >
                        {tAny("ok")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
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
            setSmoothness(String(Math.round(defaultSmoothness)));
            setNote(String(defaultValue));
        }
    }, [open, defaultSmoothness, defaultValue]);

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 340 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{titleText ?? tAny("menu_set_pitch")}</Dialog.Title>
                <Flex direction="column" gap="3" mt="3">
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 100 }}>
                            {valueLabelText ?? tAny("dlg_midi_note")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            value={note}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setNote(e.target.value)
                            }
                            style={{ flex: 1 }}
                        />
                    </Flex>
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 100 }}>
                            {tAny("edge_smoothness")}
                        </Text>
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
                </Flex>
                <Flex justify="end" gap="2" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("cancel")}
                        </Button>
                    </Dialog.Close>
                    <Button
                        onClick={() => {
                            const parsedNote = Number(note);
                            const nextValue = Number.isFinite(parsedNote)
                                ? parsedNote
                                : defaultValue;
                            onConfirm?.(
                                nextValue,
                                Math.max(0, Math.min(100, Number(smoothness) || 0)),
                            );
                            onOpenChange(false);
                        }}
                    >
                        {tAny("ok")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
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
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 400 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("menu_average")}</Dialog.Title>
                <Flex direction="column" gap="3" mt="3">
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 72 }}>
                            {tAny("dlg_average_strength")}
                        </Text>
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
                </Flex>
                <Flex justify="end" gap="2" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("cancel")}
                        </Button>
                    </Dialog.Close>
                    <Button
                        onClick={() => {
                            onConfirm?.(
                                Math.max(0, Math.min(100, Math.round(Number(strength) || 0))),
                            );
                            onOpenChange(false);
                        }}
                    >
                        {tAny("ok")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
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
        if (open) setStrength(Math.max(0, Math.min(100, Math.round(defaultSmoothness))));
    }, [open, defaultSmoothness]);

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 400 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("menu_smooth")}</Dialog.Title>
                <Flex direction="column" gap="3" mt="3">
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 72 }}>
                            {tAny("dlg_smoothness")}
                        </Text>
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
                </Flex>
                <Flex justify="end" gap="2" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("cancel")}
                        </Button>
                    </Dialog.Close>
                    <Button
                        onClick={() => {
                            onConfirm?.(Math.max(0, Math.min(100, Math.round(strength))));
                            onOpenChange(false);
                        }}
                    >
                        {tAny("ok")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
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
    // nsf-hifigan 气声音量参数范围为 0~2，此时默认振幅钳制为 1
    const isBreathGain =
        !isPitch && paramRange != null && paramRange.min === 0 && paramRange.max === 2;
    const defaultAmplitude = isPitch ? "30" : isBreathGain ? "1" : "30";

    const [amplitude, setAmplitude] = useState(defaultAmplitude);
    const [rate, setRate] = useState("5.5");
    const [attack, setAttack] = useState("50");
    const [release, setRelease] = useState("50");
    const [phase, setPhase] = useState("0");

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 380 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("menu_add_vibrato")}</Dialog.Title>
                <Flex direction="column" gap="3" mt="3">
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 120 }}>
                            {isPitch ? tAny("dlg_amplitude_cents") : tAny("dlg_amplitude")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            value={amplitude}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setAmplitude(e.target.value)
                            }
                            style={{ flex: 1 }}
                        />
                    </Flex>
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 120 }}>
                            {tAny("dlg_rate_hz")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            value={rate}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setRate(e.target.value)
                            }
                            style={{ flex: 1 }}
                        />
                    </Flex>
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 120 }}>
                            {tAny("dlg_attack_ms")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            value={attack}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setAttack(e.target.value)
                            }
                            style={{ flex: 1 }}
                        />
                    </Flex>
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 120 }}>
                            {tAny("dlg_release_ms")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            value={release}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setRelease(e.target.value)
                            }
                            style={{ flex: 1 }}
                        />
                    </Flex>
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 120 }}>
                            {tAny("dlg_phase_deg")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            value={phase}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setPhase(e.target.value)
                            }
                            style={{ flex: 1 }}
                        />
                    </Flex>
                </Flex>
                <Flex justify="end" gap="2" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("cancel")}
                        </Button>
                    </Dialog.Close>
                    <Button
                        onClick={() => {
                            onConfirm?.(
                                Number(amplitude) || 30,
                                Number(rate) || 5.5,
                                Number(attack) || 50,
                                Number(release) || 50,
                                Number(phase) || 0,
                            );
                            onOpenChange(false);
                        }}
                    >
                        {tAny("ok")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
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
    onConfirm?: (
        unit: "semitone" | "scale" | "value",
        scaleValue: string,
        toleranceCents: number,
        quantizeUnit?: number,
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
    onConfirm,
}: QuantizeProps) {
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

    useEffect(() => {
        if (open) {
            setScaleValue(defaultUseProjectScale ? "__project__" : defaultScale);
            setToleranceCents(String(toleranceDefault));
            setQuantizeUnit(String(defaultQuantizeUnit));
        }
    }, [open, defaultScale, toleranceDefault, defaultUseProjectScale, defaultQuantizeUnit]);

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 360 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("menu_quantize")}</Dialog.Title>
                <Flex direction="column" gap="3" mt="3">
                    {!valueMode && (
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 80 }}>
                                {tAny("quantize_unit")}
                            </Text>
                            <Select.Root
                                value={unit}
                                size="2"
                                onValueChange={(v) => setUnit(v as "semitone" | "scale")}
                            >
                                <Select.Trigger
                                    style={{ flex: 1 }}
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
                        </Flex>
                    )}
                    {!valueMode && unit === "scale" && (
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 80 }}>
                                {tAny("base_scale")}
                            </Text>
                            <Select.Root value={scaleValue} size="2" onValueChange={setScaleValue}>
                                <Select.Trigger
                                    style={{ flex: 1 }}
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
                        </Flex>
                    )}
                    {valueMode && (
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 80 }}>
                                {tAny("quantize_unit")}
                            </Text>
                            <TextField.Root
                                size="2"
                                type="number"
                                value={quantizeUnit}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                    setQuantizeUnit(e.target.value)
                                }
                                style={{ flex: 1 }}
                            />
                        </Flex>
                    )}
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 80 }}>
                            {valueMode ? tAny("quantize_tolerance") : tAny("pitch_snap_tolerance")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            value={toleranceCents}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setToleranceCents(e.target.value)
                            }
                            style={{ flex: 1 }}
                        />
                    </Flex>
                </Flex>
                <Flex justify="end" gap="2" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("cancel")}
                        </Button>
                    </Dialog.Close>
                    <Button
                        onClick={() => {
                            const parsed = Math.abs(Math.round(Number(toleranceCents) || 0));
                            const parsedUnit = Math.abs(Number(quantizeUnit) || 0);
                            onConfirm?.(
                                valueMode ? "value" : unit,
                                scaleValue,
                                parsed,
                                valueMode ? parsedUnit : undefined,
                            );
                            onOpenChange(false);
                        }}
                    >
                        {tAny("ok")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
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
    onConfirm?: (
        unit: "semitone" | "scale" | "value",
        scaleValue: string,
        toleranceCents: number,
        quantizeUnit?: number,
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
    onConfirm,
}: MeanQuantizeProps) {
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

    useEffect(() => {
        if (open) {
            setScaleValue(defaultUseProjectScale ? "__project__" : defaultScale);
            setToleranceCents(String(toleranceDefault));
            setQuantizeUnit(String(defaultQuantizeUnit));
        }
    }, [open, defaultScale, toleranceDefault, defaultUseProjectScale, defaultQuantizeUnit]);

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 360 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("mean_quantize_title")}</Dialog.Title>
                <Flex direction="column" gap="3" mt="3">
                    {!valueMode && (
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 80 }}>
                                {tAny("quantize_unit")}
                            </Text>
                            <Select.Root
                                value={unit}
                                size="2"
                                onValueChange={(v) => setUnit(v as "semitone" | "scale")}
                            >
                                <Select.Trigger
                                    style={{ flex: 1 }}
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
                        </Flex>
                    )}
                    {!valueMode && unit === "scale" && (
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 80 }}>
                                {tAny("base_scale")}
                            </Text>
                            <Select.Root value={scaleValue} size="2" onValueChange={setScaleValue}>
                                <Select.Trigger
                                    style={{ flex: 1 }}
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
                        </Flex>
                    )}
                    {valueMode && (
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 80 }}>
                                {tAny("quantize_unit")}
                            </Text>
                            <TextField.Root
                                size="2"
                                type="number"
                                value={quantizeUnit}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                    setQuantizeUnit(e.target.value)
                                }
                                style={{ flex: 1 }}
                            />
                        </Flex>
                    )}
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 80 }}>
                            {valueMode ? tAny("quantize_tolerance") : tAny("pitch_snap_tolerance")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            value={toleranceCents}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setToleranceCents(e.target.value)
                            }
                            style={{ flex: 1 }}
                        />
                    </Flex>
                </Flex>
                <Flex justify="end" gap="2" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("cancel")}
                        </Button>
                    </Dialog.Close>
                    <Button
                        onClick={() => {
                            const parsed = Math.abs(Math.round(Number(toleranceCents) || 0));
                            const parsedUnit = Math.abs(Number(quantizeUnit) || 0);
                            onConfirm?.(
                                valueMode ? "value" : unit,
                                scaleValue,
                                parsed,
                                valueMode ? parsedUnit : undefined,
                            );
                            onOpenChange(false);
                        }}
                    >
                        {tAny("ok")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
