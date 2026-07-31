import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { Button, Dialog, Flex, Select, Text, TextField } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import {
    devicesLoaded,
    loadRecordingDevices,
    loadRecordingSettings,
    saveRecordingSettings,
} from "../../features/recording/recordingSlice";
import {
    DEFAULT_RECORDING_SETTINGS,
    type RecordingSettings,
} from "../../services/api/recording";
import { webApi } from "../../services/webviewApi";

interface RecordingSettingsDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

function clampGain(raw: number): number {
    if (!Number.isFinite(raw)) return 0;
    return Math.min(24, Math.max(-24, Math.round(raw * 10) / 10));
}

export function RecordingSettingsDialog({
    open,
    onOpenChange,
}: RecordingSettingsDialogProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const savedSettings = useAppSelector((state) => state.recording.settings);
    const devices = useAppSelector((state) => state.recording.devices);
    const [draft, setDraft] = useState<RecordingSettings>(savedSettings);
    const [submitting, setSubmitting] = useState(false);
    const [errorText, setErrorText] = useState("");
    const pathInputRef = useRef<HTMLInputElement | null>(null);

    useEffect(() => {
        if (!open) return;
        setErrorText("");
        void dispatch(loadRecordingSettings());
        void dispatch(loadRecordingDevices());
    }, [open, dispatch]);

    useEffect(() => {
        if (open) {
            setDraft((prev) => ({
                ...savedSettings,
                // 保留用户正在输入但尚未保存的路径模板。
                pathTemplate:
                    prev.pathTemplate && prev.pathTemplate !== DEFAULT_RECORDING_SETTINGS.pathTemplate
                        ? prev.pathTemplate
                        : savedSettings.pathTemplate,
            }));
        }
    }, [open, savedSettings]);

    function getPathInputElement(): HTMLInputElement | null {
        const input = pathInputRef.current;
        if (!input?.isConnected) {
            pathInputRef.current = null;
            return null;
        }
        return input;
    }

    function insertPathToken(token: string) {
        const input = getPathInputElement();
        if (!input) return;
        const start = input.selectionStart ?? input.value.length;
        const end = input.selectionEnd ?? input.value.length;
        const nextValue = `${input.value.slice(0, start)}${token}${input.value.slice(end)}`;
        setDraft((prev) => ({ ...prev, pathTemplate: nextValue }));
        window.requestAnimationFrame(() => {
            input.focus();
            const nextPos = start + token.length;
            input.setSelectionRange(nextPos, nextPos);
        });
    }

    async function refreshDevices() {
        try {
            const result = await webApi.getRecordingDevices();
            if (result.devices) {
                dispatch(devicesLoaded(result.devices));
            }
        } catch {
            setErrorText(tAny("recording_error_load_devices"));
        }
    }

    async function handleSave() {
        setErrorText("");
        setSubmitting(true);
        const nextSettings: RecordingSettings = {
            ...draft,
            sourceDevice: draft.sourceDevice?.trim() || "default",
            sampleRate: Number(draft.sampleRate) || 48_000,
            bitDepth: (Number(draft.bitDepth) === 16 || Number(draft.bitDepth) === 32
                ? Number(draft.bitDepth)
                : 24) as 16 | 24 | 32,
            channels: Number(draft.channels) === 1 ? 1 : 2,
            inputGainDb: clampGain(Number(draft.inputGainDb)),
            monitorGainDb: clampGain(Number(draft.monitorGainDb)),
            countdownSec: Math.min(10, Math.max(0, Math.floor(Number(draft.countdownSec) || 0))),
            pathTemplate: String(draft.pathTemplate ?? "").trim(),
        };
        try {
            await dispatch(saveRecordingSettings(nextSettings)).unwrap();
            onOpenChange(false);
        } catch {
            setErrorText(tAny("recording_error_save_settings"));
        } finally {
            setSubmitting(false);
        }
    }

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content
                style={{ maxWidth: 760 }}
                onKeyDown={(event) => event.stopPropagation()}
            >
                <Dialog.Title>{tAny("menu_recording_settings")}</Dialog.Title>
                <Dialog.Description>{tAny("recording_settings_desc")}</Dialog.Description>

                <Flex direction="column" gap="3" mt="3">
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 132 }}>
                            {tAny("recording_device")}
                        </Text>
                        <Select.Root
                            value={draft.sourceDevice}
                            onValueChange={(value) =>
                                setDraft((prev) => ({ ...prev, sourceDevice: value }))
                            }
                        >
                            <Select.Trigger style={{ minWidth: 260 }} />
                            <Select.Content>
                                <Select.Item value="default">
                                    {tAny("recording_device_default")}
                                </Select.Item>
                                {devices
                                    .filter((device) => !device.isDefault)
                                    .map((device) => (
                                        <Select.Item key={device.id} value={device.id}>
                                            {device.isLoopback
                                                ? `${tAny("recording_system_sound")} - ${device.name}`
                                                : device.name}
                                        </Select.Item>
                                    ))}
                            </Select.Content>
                        </Select.Root>
                        <Button
                            size="1"
                            variant="ghost"
                            color="gray"
                            onClick={() => void refreshDevices()}
                        >
                            {tAny("recording_refresh_devices")}
                        </Button>
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 132 }}>
                            {tAny("recording_sample_rate")}
                        </Text>
                        <Select.Root
                            value={String(draft.sampleRate)}
                            onValueChange={(value) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    sampleRate: Number(value),
                                }))
                            }
                        >
                            <Select.Trigger style={{ minWidth: 120 }} />
                            <Select.Content>
                                {[44_100, 48_000, 88_200, 96_000].map((rate) => (
                                    <Select.Item key={rate} value={String(rate)}>
                                        {rate} Hz
                                    </Select.Item>
                                ))}
                            </Select.Content>
                        </Select.Root>
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 132 }}>
                            {tAny("recording_bit_depth")}
                        </Text>
                        <Select.Root
                            value={String(draft.bitDepth)}
                            onValueChange={(value) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    bitDepth: Number(value) as 16 | 24 | 32,
                                }))
                            }
                        >
                            <Select.Trigger style={{ minWidth: 120 }} />
                            <Select.Content>
                                <Select.Item value="16">16-bit</Select.Item>
                                <Select.Item value="24">24-bit</Select.Item>
                                <Select.Item value="32">32-bit float</Select.Item>
                            </Select.Content>
                        </Select.Root>

                        <Text size="2" ml="4">
                            {tAny("recording_channels")}
                        </Text>
                        <Select.Root
                            value={String(draft.channels)}
                            onValueChange={(value) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    channels: Number(value) === 1 ? 1 : 2,
                                }))
                            }
                        >
                            <Select.Trigger style={{ minWidth: 100 }} />
                            <Select.Content>
                                <Select.Item value="1">{tAny("recording_mono")}</Select.Item>
                                <Select.Item value="2">{tAny("recording_stereo")}</Select.Item>
                            </Select.Content>
                        </Select.Root>
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 132 }}>
                            {tAny("recording_input_gain")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            min={-24}
                            max={24}
                            step={0.1}
                            value={String(draft.inputGainDb)}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    inputGainDb: Number(event.target.value),
                                }))
                            }
                            style={{ width: 120 }}
                        />
                        <Text size="1" color="gray">
                            dB
                        </Text>
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 132 }}>
                            {tAny("recording_countdown")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            min={0}
                            max={10}
                            step={1}
                            value={String(draft.countdownSec)}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    countdownSec: Number(event.target.value),
                                }))
                            }
                            style={{ width: 120 }}
                        />
                        <Text size="1" color="gray">
                            {tAny("recording_countdown_unit")}
                        </Text>
                    </Flex>

                    <label className="flex items-center gap-2 text-sm text-qt-text">
                        <input
                            type="checkbox"
                            checked={draft.monitorEnabled}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    monitorEnabled: event.target.checked,
                                }))
                            }
                        />
                        <span>{tAny("recording_monitor_enabled")}</span>
                    </label>

                    {draft.monitorEnabled ? (
                        <Flex align="center" gap="2" pl="6">
                            <Text size="2" style={{ minWidth: 120 }}>
                                {tAny("recording_monitor_gain")}
                            </Text>
                            <TextField.Root
                                size="2"
                                type="number"
                                min={-24}
                                max={24}
                                step={0.1}
                                value={String(draft.monitorGainDb)}
                                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                    setDraft((prev) => ({
                                        ...prev,
                                        monitorGainDb: Number(event.target.value),
                                    }))
                                }
                                style={{ width: 120 }}
                            />
                            <Text size="1" color="gray">
                                dB
                            </Text>
                        </Flex>
                    ) : null}

                    <label className="flex items-center gap-2 text-sm text-qt-text">
                        <input
                            type="checkbox"
                            checked={draft.autoNormalize}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    autoNormalize: event.target.checked,
                                }))
                            }
                        />
                        <span>{tAny("recording_auto_normalize")}</span>
                    </label>

                    <label className="flex items-center gap-2 text-sm text-qt-text">
                        <input
                            type="checkbox"
                            checked={draft.autoStopAtSelectionEnd}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    autoStopAtSelectionEnd: event.target.checked,
                                }))
                            }
                        />
                        <span>{tAny("recording_auto_stop_selection")}</span>
                    </label>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 132 }}>
                            {tAny("recording_path_template")}
                        </Text>
                        <TextField.Root
                            size="2"
                            value={draft.pathTemplate}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    pathTemplate: event.target.value,
                                }))
                            }
                            onFocus={(event) => {
                                pathInputRef.current = event.target as HTMLInputElement;
                            }}
                            style={{ flex: 1 }}
                        />
                    </Flex>

                    <Flex gap="2" wrap="wrap" align="center">
                        <Text size="1" color="gray">
                            {tAny("auto_backup_placeholders")}
                        </Text>
                        {["<ProjectFolder>", "<ProjectName>"].map((token) => (
                            <Button
                                key={token}
                                size="1"
                                variant="ghost"
                                color="gray"
                                onClick={() => insertPathToken(token)}
                            >
                                {token}
                            </Button>
                        ))}
                    </Flex>

                    <Text size="1" color="gray">
                        {tAny("auto_backup_time_format_hint")}
                    </Text>

                    {errorText ? (
                        <Text size="2" color="red">
                            {errorText}
                        </Text>
                    ) : null}
                </Flex>

                <Flex justify="end" gap="2" mt="4">
                    <Button variant="soft" color="gray" onClick={() => onOpenChange(false)}>
                        {tAny("cancel")}
                    </Button>
                    <Button onClick={() => void handleSave()} disabled={submitting}>
                        {tAny("recording_save_settings")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
