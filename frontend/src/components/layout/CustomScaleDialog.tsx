import { useEffect, useMemo, useState } from "react";
import { Button, Dialog, Flex, Select, Text, TextField } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    persistUiSettings,
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

    useEffect(() => {
        if (!open) return;
        const current = s.project.useCustomScale && s.project.customScale
            ? s.project.customScale
            : null;
        if (current) {
            const normalized = sanitizeCustomScalePreset(current);
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

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 520 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("custom_scale_dialog_title")}</Dialog.Title>

                <Flex direction="column" gap="3" mt="3">
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 112 }}>{tAny("custom_scale_template")}</Text>
                        <Select.Root value={templateValue} onValueChange={applyTemplate} size="2">
                            <Select.Trigger style={{ flex: 1 }} />
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
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 112 }}>{tAny("custom_scale_name")}</Text>
                        <TextField.Root
                            size="2"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            style={{ flex: 1 }}
                        />
                    </Flex>

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
                </Flex>

                <Flex justify="end" gap="2" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray">{tAny("cancel")}</Button>
                    </Dialog.Close>
                    <Button onClick={handleSave}>{tAny("custom_scale_save_apply")}</Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
