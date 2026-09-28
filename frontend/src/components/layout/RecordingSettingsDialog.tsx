import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { Flex, TextField } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import {
    appsLoaded,
    devicesLoaded,
    loadRecordingDevices,
    loadRecordingApps,
    loadRecordingSettings,
    saveRecordingSettings,
} from "../../features/recording/recordingSlice";
import {
    DEFAULT_RECORDING_SETTINGS,
    type RecordingAppInfo,
    type RecordingSettings,
} from "../../services/api/recording";
import { webApi } from "../../services/webviewApi";
import { AppButton } from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm, AppSwitchRow } from "../../ui/Field";
import { AppNumberField, AppSelect } from "../../ui";

interface RecordingSettingsDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

function clampGain(raw: number): number {
    if (!Number.isFinite(raw)) return 0;
    return Math.min(24, Math.max(-24, Math.round(raw * 10) / 10));
}

export function RecordingSettingsDialog({ open, onOpenChange }: RecordingSettingsDialogProps) {
    const dispatch = useAppDispatch();
    const { tf } = useI18n();
    const savedSettings = useAppSelector((state) => state.recording.settings);
    const devices = useAppSelector((state) => state.recording.devices);
    const apps = useAppSelector((state) => state.recording.apps);
    const [draft, setDraft] = useState<RecordingSettings>(savedSettings);
    const [submitting, setSubmitting] = useState(false);
    const [errorText, setErrorText] = useState("");
    const pathInputRef = useRef<HTMLInputElement | null>(null);

    useEffect(() => {
        if (!open) return;
        setErrorText("");
        void dispatch(loadRecordingSettings());
        void dispatch(loadRecordingDevices());
        void dispatch(loadRecordingApps());
    }, [open, dispatch]);

    // 草稿只在“打开”这一时机初始化一次：loadRecordingSettings() 稍后回填
    // savedSettings 会再次触发本 effect —— 若以 savedSettings 为依赖，用户
    // 在加载间隙已修改的采样率/增益/勾选会被静默还原（pathTemplate 已有
    // 输入中保护，其余字段没有）。经 ref 读取打开瞬间的最新已存值。
    const savedSettingsRef = useRef(savedSettings);
    savedSettingsRef.current = savedSettings;
    useEffect(() => {
        if (!open) return;
        const saved = savedSettingsRef.current;
        setDraft((prev) => ({
            ...saved,
            // 保留用户正在输入但尚未保存的路径模板（跨关闭/重开仍保留）。
            pathTemplate:
                prev.pathTemplate && prev.pathTemplate !== DEFAULT_RECORDING_SETTINGS.pathTemplate
                    ? prev.pathTemplate
                    : saved.pathTemplate,
        }));
    }, [open]);

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
            setErrorText(tf("recording_error_load_devices"));
        }
    }

    async function refreshApps() {
        try {
            const result = await webApi.getRecordingApps();
            if (result.apps) {
                dispatch(appsLoaded(result.apps));
            }
        } catch {
            setErrorText(tf("recording_error_load_apps"));
        }
    }

    function appNameById(id: string): string {
        const app = apps.find((item) => item.id === id);
        return app?.name ?? draft.captureAppName ?? id;
    }

    const missingApp = Boolean(
        draft.captureAppId && !apps.some((item) => item.id === draft.captureAppId),
    );
    const loopbackValue =
        draft.loopbackDevice === "default" ? "loopback:default" : draft.loopbackDevice;
    const inputDevices = devices.filter((device) => !device.isLoopback);
    const loopbackDevices = devices.filter((device) => device.isLoopback);

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
            setErrorText(tf("recording_error_save_settings"));
        } finally {
            setSubmitting(false);
        }
    }

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("menu_recording_settings")}
            description={tf("recording_settings_desc")}
            size="xl"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "save",
                    label: tf("recording_save_settings"),
                    intent: "primary",
                    disabled: submitting,
                    onClick: handleSave,
                },
            ]}
        >
            <AppForm>
                <AppField label={tf("recording_source_mode")}>
                    <AppSelect
                        value={draft.captureMode}
                        onValueChange={(value) =>
                            setDraft((prev) => ({
                                ...prev,
                                captureMode: value as RecordingSettings["captureMode"],
                            }))
                        }
                        options={[
                            { value: "device", label: tf("recording_mode_device") },
                            { value: "loopback", label: tf("recording_mode_loopback") },
                            { value: "application", label: tf("recording_mode_application") },
                        ]}
                    />
                </AppField>

                {draft.captureMode === "device" ? (
                    <AppField label={tf("recording_device")}>
                        <Flex align="center" gap="2">
                            <AppSelect
                                fullWidth={false}
                                // 不定长文本（设备名/应用名），给下限防止塌缩与切换时宽度跳动
                                minWidth={260}
                                value={draft.sourceDevice}
                                onValueChange={(value) =>
                                    setDraft((prev) => ({ ...prev, sourceDevice: value }))
                                }
                                options={[
                                    { value: "default", label: tf("recording_device_default") },
                                    ...inputDevices
                                        .filter((device) => !device.isDefault)
                                        .map((device) => ({
                                            value: device.id,
                                            label: device.name,
                                        })),
                                ]}
                            />
                            <AppButton size="sm" onClick={() => void refreshDevices()}>
                                {tf("recording_refresh_devices")}
                            </AppButton>
                        </Flex>
                    </AppField>
                ) : null}

                {draft.captureMode === "loopback" ? (
                    <AppField label={tf("recording_loopback_device")}>
                        <Flex align="center" gap="2">
                            <AppSelect
                                fullWidth={false}
                                // 不定长文本（设备名/应用名），给下限防止塌缩与切换时宽度跳动
                                minWidth={260}
                                value={loopbackValue}
                                onValueChange={(value) =>
                                    setDraft((prev) => ({ ...prev, loopbackDevice: value }))
                                }
                                options={[
                                    {
                                        value: "loopback:default",
                                        label: tf("recording_loopback_default"),
                                    },
                                    ...loopbackDevices
                                        .filter((device) => device.id !== "loopback:default")
                                        .map((device) => ({
                                            value: device.id,
                                            label: device.name,
                                        })),
                                ]}
                            />
                            <AppButton size="sm" onClick={() => void refreshDevices()}>
                                {tf("recording_refresh_devices")}
                            </AppButton>
                        </Flex>
                    </AppField>
                ) : null}

                {draft.captureMode === "application" ? (
                    <>
                        <AppField label={tf("recording_application")}>
                            <Flex align="center" gap="2">
                                <AppSelect
                                    fullWidth={false}
                                    // 不定长文本（设备名/应用名），给下限防止塌缩与切换时宽度跳动
                                    minWidth={260}
                                    value={draft.captureAppId}
                                    onValueChange={(value) => {
                                        const app = apps.find((item) => item.id === value);
                                        setDraft((prev) => ({
                                            ...prev,
                                            captureAppId: value,
                                            captureAppName: app?.name ?? value,
                                            captureAppProcess: app?.processName ?? "",
                                        }));
                                    }}
                                    options={[
                                        ...(missingApp
                                            ? [
                                                  {
                                                      value: draft.captureAppId,
                                                      label: appNameById(draft.captureAppId),
                                                  },
                                              ]
                                            : []),
                                        ...apps.map((app: RecordingAppInfo) => ({
                                            value: app.id,
                                            label: app.name,
                                        })),
                                    ]}
                                />
                                <AppButton size="sm" onClick={() => void refreshApps()}>
                                    {tf("recording_refresh_apps")}
                                </AppButton>
                            </Flex>
                        </AppField>
                        <span className="hs-type-caption" style={{ paddingLeft: 20 }}>
                            {tf("recording_application_hint")}
                        </span>
                    </>
                ) : null}

                <AppField label={tf("recording_sample_rate")}>
                    <AppSelect
                        value={String(draft.sampleRate)}
                        onValueChange={(value) =>
                            setDraft((prev) => ({
                                ...prev,
                                sampleRate: Number(value),
                            }))
                        }
                        options={[44_100, 48_000, 88_200, 96_000].map((rate) => ({
                            value: String(rate),
                            label: `${rate} Hz`,
                        }))}
                    />
                </AppField>

                <AppField label={tf("recording_bit_depth")}>
                    <Flex align="center" gap="2">
                        <AppSelect
                            fullWidth={false}
                            // 不定长文本（设备名/应用名），给下限防止塌缩与切换时宽度跳动
                            minWidth={120}
                            value={String(draft.bitDepth)}
                            onValueChange={(value) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    bitDepth: Number(value) as 16 | 24 | 32,
                                }))
                            }
                            options={[
                                { value: "16", label: "16-bit" },
                                { value: "24", label: "24-bit" },
                                { value: "32", label: "32-bit float" },
                            ]}
                        />

                        <span className="hs-type-label ml-4">{tf("recording_channels")}</span>
                        <AppSelect
                            fullWidth={false}
                            // 不定长文本（设备名/应用名），给下限防止塌缩与切换时宽度跳动
                            minWidth={100}
                            value={String(draft.channels)}
                            onValueChange={(value) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    channels: Number(value) === 1 ? 1 : 2,
                                }))
                            }
                            options={[
                                { value: "1", label: tf("recording_mono") },
                                { value: "2", label: tf("recording_stereo") },
                            ]}
                        />
                    </Flex>
                </AppField>

                <AppField label={tf("recording_input_gain")}>
                    <AppNumberField
                        value={draft.inputGainDb}
                        unit="gainDb"
                        min={-24}
                        max={24}
                        width={120}
                        suffix="dB"
                        ariaLabel={tf("recording_input_gain")}
                        onCommit={(inputGainDb) =>
                            setDraft((prev) => ({
                                ...prev,
                                inputGainDb,
                            }))
                        }
                    />
                </AppField>

                <AppField label={tf("recording_countdown")}>
                    <AppNumberField
                        value={draft.countdownSec}
                        unit="integer"
                        min={0}
                        max={10}
                        width={120}
                        suffix={tf("recording_countdown_unit")}
                        ariaLabel={tf("recording_countdown")}
                        onCommit={(countdownSec) =>
                            setDraft((prev) => ({
                                ...prev,
                                countdownSec,
                            }))
                        }
                    />
                </AppField>

                <AppSwitchRow
                    control="checkbox"
                    label={tf("recording_monitor_enabled")}
                    checked={draft.monitorEnabled}
                    onCheckedChange={(monitorEnabled) =>
                        setDraft((prev) => ({
                            ...prev,
                            monitorEnabled,
                        }))
                    }
                />

                {draft.monitorEnabled ? (
                    <AppField label={tf("recording_monitor_gain")} className="pl-6">
                        <AppNumberField
                            value={draft.monitorGainDb}
                            unit="gainDb"
                            min={-24}
                            max={24}
                            width={120}
                            suffix="dB"
                            ariaLabel={tf("recording_monitor_gain")}
                            onCommit={(monitorGainDb) =>
                                setDraft((prev) => ({
                                    ...prev,
                                    monitorGainDb,
                                }))
                            }
                        />
                    </AppField>
                ) : null}

                <AppSwitchRow
                    control="checkbox"
                    label={tf("recording_auto_normalize")}
                    checked={draft.autoNormalize}
                    onCheckedChange={(autoNormalize) =>
                        setDraft((prev) => ({
                            ...prev,
                            autoNormalize,
                        }))
                    }
                />

                <AppSwitchRow
                    control="checkbox"
                    label={tf("recording_auto_stop_selection")}
                    checked={draft.autoStopAtSelectionEnd}
                    onCheckedChange={(autoStopAtSelectionEnd) =>
                        setDraft((prev) => ({
                            ...prev,
                            autoStopAtSelectionEnd,
                        }))
                    }
                />

                <AppField label={tf("recording_path_template")}>
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
                    />
                </AppField>

                <Flex gap="2" wrap="wrap" align="center">
                    <span className="hs-type-caption">{tf("auto_backup_placeholders")}</span>
                    {["<ProjectFolder>", "<ProjectName>"].map((token) => (
                        <AppButton key={token} size="sm" onClick={() => insertPathToken(token)}>
                            {token}
                        </AppButton>
                    ))}
                </Flex>

                <span className="hs-type-caption">{tf("auto_backup_time_format_hint")}</span>

                {errorText ? (
                    <span className="hs-type-body" style={{ color: "var(--qt-danger-text)" }}>
                        {errorText}
                    </span>
                ) : null}
            </AppForm>
        </AppDialog>
    );
}
