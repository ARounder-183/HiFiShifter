import { useEffect, useMemo, useState } from "react";
import { Button, Flex, Select, Text, TextField } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    persistUiSettings,
    removeCustomScalePreset,
    setProjectBaseScaleRemote,
    setProjectCustomScaleRemote,
    upsertCustomScalePreset,
} from "../../features/session/sessionSlice";
import {
    CHROMATIC_NOTE_LABELS,
    createCustomScaleId,
    formatScaleNotes,
    sanitizeCustomScalePreset,
} from "../../utils/customScales";
import { SCALE_KEYS, SCALE_LABELS, resolveScaleNotes } from "../../utils/musicalScales";
import { applySelectWheelChange } from "../../utils/selectWheel";
import { getSelectedCustomScaleId } from "./customScaleDialogLogic";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

const BUILTIN_TEMPLATE_PREFIX = "builtin:";
const CUSTOM_TEMPLATE_PREFIX = "custom:";

export function CustomScaleDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const s = useAppSelector((state: RootState) => state.session);

    const [name, setName] = useState("");
    const [notes, setNotes] = useState<number[]>([]);
    const [editingPresetId, setEditingPresetId] = useState<string | null>(null);
    const [templateValue, setTemplateValue] = useState<string>(`${BUILTIN_TEMPLATE_PREFIX}C`);

    const customPresetOptions = useMemo(() => s.customScalePresets, [s.customScalePresets]);
    const selectedCustomPresetId = useMemo(() => {
        return getSelectedCustomScaleId(
            templateValue,
            customPresetOptions.map((preset) => preset.id),
        );
    }, [templateValue, customPresetOptions]);

    useEffect(() => {
        if (!open) return;
        const current =
            s.project.useCustomScale && s.project.customScale ? s.project.customScale : null;
        if (current) {
            const normalized = sanitizeCustomScalePreset(current);
            // eslint-disable-next-line react-hooks/set-state-in-effect -- 对话框打开时用 props 初始化局部 state（既有模式；重构会改变打开时序）
            setName(normalized.name);
            setNotes(normalized.notes);
            setEditingPresetId(normalized.id);
            setTemplateValue(`${CUSTOM_TEMPLATE_PREFIX}${normalized.id}`);
            return;
        }

        const fallbackScale = s.project.baseScale;
        setName(tAny("custom_scale_default_name"));
        setNotes(resolveScaleNotes(fallbackScale));
        setEditingPresetId(null);
        setTemplateValue(`${BUILTIN_TEMPLATE_PREFIX}${fallbackScale}`);
    }, [open, s.project.baseScale, s.project.customScale, s.project.useCustomScale, tAny]);

    function toggleNote(pc: number) {
        setNotes((prev) => {
            const has = prev.includes(pc);
            const next = has ? prev.filter((n) => n !== pc) : [...prev, pc];
            if (next.length === 0) return prev;
            return next.sort((a, b) => a - b);
        });
    }

    function applyTemplate(value: string) {
        setTemplateValue(value);
        if (value.startsWith(BUILTIN_TEMPLATE_PREFIX)) {
            const key = value.slice(BUILTIN_TEMPLATE_PREFIX.length);
            if ((SCALE_KEYS as readonly string[]).includes(key)) {
                setNotes(resolveScaleNotes(key as (typeof SCALE_KEYS)[number]));
                setEditingPresetId(null);
            }
            return;
        }
        if (value.startsWith(CUSTOM_TEMPLATE_PREFIX)) {
            const id = value.slice(CUSTOM_TEMPLATE_PREFIX.length);
            const preset = customPresetOptions.find((item) => item.id === id);
            if (!preset) return;
            const normalized = sanitizeCustomScalePreset(preset);
            setName(normalized.name);
            setNotes(normalized.notes);
            setEditingPresetId(normalized.id);
        }
    }

    function handleSave() {
        const nextId = editingPresetId ?? createCustomScaleId();
        const preset = sanitizeCustomScalePreset({ id: nextId, name, notes });
        dispatch(upsertCustomScalePreset(preset));
        void dispatch(persistUiSettings());
        void dispatch(setProjectCustomScaleRemote(preset));
        onOpenChange(false);
    }

    function handleDeleteSelectedPreset() {
        if (!selectedCustomPresetId) return;

        const preset = customPresetOptions.find((item) => item.id === selectedCustomPresetId);
        if (!preset) return;

        const isCurrentProjectCustom =
            s.project.useCustomScale && s.project.customScale?.id === selectedCustomPresetId;

        dispatch(removeCustomScalePreset(selectedCustomPresetId));
        void dispatch(persistUiSettings());

        if (isCurrentProjectCustom) {
            dispatch(setProjectBaseScaleRemote(s.project.baseScale));
        }

        if (editingPresetId === selectedCustomPresetId) {
            const fallbackScale = s.project.baseScale;
            setEditingPresetId(null);
            setName(tAny("custom_scale_default_name"));
            setNotes(resolveScaleNotes(fallbackScale));
            setTemplateValue(`${BUILTIN_TEMPLATE_PREFIX}${fallbackScale}`);
        }
    }

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("custom_scale_dialog_title")}
            size="md"
            actions={[
                {
                    id: "delete",
                    label: tAny("custom_scale_delete"),
                    intent: "danger",
                    align: "start",
                    disabled: !selectedCustomPresetId,
                    // 异步包装：删除不关闭对话框，避免页脚表单重新提交触发默认动作。
                    onClick: async () => {
                        handleDeleteSelectedPreset();
                    },
                },
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("custom_scale_save_apply"),
                    intent: "primary",
                    onClick: handleSave,
                },
            ]}
        >
            <AppForm>
                <AppField label={tAny("custom_scale_template")}>
                    <Select.Root value={templateValue} onValueChange={applyTemplate} size="2">
                        <Select.Trigger
                            onWheel={(event) => {
                                applySelectWheelChange({
                                    event,
                                    currentValue: templateValue,
                                    options: [
                                        ...SCALE_KEYS.map((k) => `${BUILTIN_TEMPLATE_PREFIX}${k}`),
                                        ...customPresetOptions.map(
                                            (preset) => `${CUSTOM_TEMPLATE_PREFIX}${preset.id}`,
                                        ),
                                    ],
                                    onChange: applyTemplate,
                                });
                            }}
                        />
                        <Select.Content>
                            <Select.Group>
                                {SCALE_KEYS.map((k) => (
                                    <Select.Item key={k} value={`${BUILTIN_TEMPLATE_PREFIX}${k}`}>
                                        {SCALE_LABELS[k]}
                                    </Select.Item>
                                ))}
                            </Select.Group>
                            {customPresetOptions.length > 0 ? (
                                <>
                                    <Select.Separator />
                                    <Select.Group>
                                        {customPresetOptions.map((preset) => (
                                            <Select.Item
                                                key={preset.id}
                                                value={`${CUSTOM_TEMPLATE_PREFIX}${preset.id}`}
                                            >
                                                {`${preset.name} (${formatScaleNotes(preset.notes)})`}
                                            </Select.Item>
                                        ))}
                                    </Select.Group>
                                </>
                            ) : null}
                        </Select.Content>
                    </Select.Root>
                </AppField>

                <AppField label={tAny("custom_scale_name")}>
                    <TextField.Root
                        size="2"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                    />
                </AppField>

                <Flex direction="column" gap="2">
                    <Text size="2">{tAny("custom_scale_notes")}</Text>
                    <Flex wrap="wrap" gap="2">
                        {CHROMATIC_NOTE_LABELS.map((label, pc) => {
                            const selected = notes.includes(pc);
                            return (
                                <Button
                                    key={label}
                                    type="button"
                                    size="1"
                                    variant={selected ? "solid" : "soft"}
                                    color={selected ? "amber" : "gray"}
                                    onClick={() => toggleNote(pc)}
                                >
                                    {label}
                                </Button>
                            );
                        })}
                    </Flex>
                </Flex>
            </AppForm>
        </AppDialog>
    );
}
