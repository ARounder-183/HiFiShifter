/*
 * 指针设备（触控板 / 数位板 / 触控笔 / 触摸）输入设置对话框。
 *
 * 背景：手势层原先只按"鼠标 + 键盘修饰键"设计 —— 精细调整必须另一只手按 Ctrl、
 * 次级手势只能靠右键、连续调节只有滚轮。这在触控板（没有侧键 / 中键）、数位笔
 * （另一只手扶着板子）、触屏（根本没有修饰键）上分别是难用、不实用、不存在。
 *
 * 本对话框把那些设备专属的补偿**集中暴露并可关**：任何一项关掉后，行为退回
 * "和鼠标一样"。设备**能力**（有没有压感通道）不在这里配，而是由
 * `utils/inputProfile.ts` 在运行时按事件判定。
 *
 * 交互约定（与其它设置对话框一致）：
 * - 设置以草稿形式编辑，点「保存」才落盘；
 * - 草稿只在"打开"这一时机初始化一次，避免保存后 effect 重跑清掉提示；
 * - 下拉框与数字输入框统一走 `AppSelect` / `AppNumberField` 原语，滚轮步进与
 *   「精细调整」修饰键由原语内建。
 */

import { useEffect, useRef, useState } from "react";
import { Separator } from "@radix-ui/themes";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import {
    normalizePenInputSettings,
    type ContactReadoutMode,
    type PenInputSettings,
    type PointerDeviceDeclaration,
} from "../../services/api/settings";
import { persistUiSettings } from "../../features/session/thunks/runtimeThunks";
import { setPenInputSettings } from "../../features/session/sessionSlice";
import { AppNumberField, AppSelect, AppSwitchRow } from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";

interface PenInputDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

const DEVICE_OPTIONS: ReadonlyArray<{ value: PointerDeviceDeclaration; labelKey: string }> = [
    { value: "auto", labelKey: "pen_input_device_auto" },
    { value: "mouse", labelKey: "pen_input_device_mouse" },
    { value: "trackpad", labelKey: "pen_input_device_trackpad" },
    { value: "pen", labelKey: "pen_input_device_pen" },
    { value: "touch", labelKey: "pen_input_device_touch" },
];

const READOUT_OPTIONS: ReadonlyArray<{ value: ContactReadoutMode; labelKey: string }> = [
    { value: "off", labelKey: "pen_input_contact_readout_off" },
    { value: "touchOnly", labelKey: "pen_input_contact_readout_touch_only" },
    { value: "always", labelKey: "pen_input_contact_readout_always" },
];

export function PenInputDialog({ open, onOpenChange }: PenInputDialogProps) {
    const dispatch = useAppDispatch();
    const { tf } = useI18n();
    const saved = useAppSelector((state) => state.session.penInput);

    const [draft, setDraft] = useState<PenInputSettings>(saved);
    const [saving, setSaving] = useState(false);
    const [notice, setNotice] = useState("");
    const [errorText, setErrorText] = useState("");

    /*
     * 草稿只在"打开"这一时机初始化一次：保存后 Redux 中的设置会更新，若把
     * `saved` 放进依赖，effect 会立刻重跑并把"设置已保存"的提示清掉。
     */
    const savedRef = useRef(saved);
    savedRef.current = saved;
    useEffect(() => {
        if (!open) {
            setNotice("");
            return;
        }
        setDraft({ ...savedRef.current });
        setNotice("");
        setErrorText("");
    }, [open]);

    function patch(partial: Partial<PenInputSettings>) {
        setDraft((prev) => ({ ...prev, ...partial }));
    }

    async function handleSave() {
        setErrorText("");
        setSaving(true);
        try {
            const normalized = normalizePenInputSettings(draft);
            dispatch(setPenInputSettings(normalized));
            await dispatch(persistUiSettings());
            setDraft(normalized);
            setNotice(tf("render_cache_settings_saved"));
        } catch {
            setErrorText(tf("render_cache_settings_save_failed"));
        } finally {
            setSaving(false);
        }
    }

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("pen_input_dialog_title")}
            description={tf("pen_input_dialog_desc")}
            size="md"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "save",
                    label: tf("pen_input_save"),
                    intent: "primary",
                    disabled: saving,
                    onClick: handleSave,
                },
            ]}
        >
            <AppForm>
                <Separator size="4" />

                {/* ── 设备声明 ───────────────────────────────────────── */}
                <AppField label={tf("pen_input_device")} hint={tf("pen_input_device_hint")}>
                    <AppSelect
                        value={draft.device}
                        onValueChange={(value) =>
                            patch({ device: value as PointerDeviceDeclaration })
                        }
                        options={DEVICE_OPTIONS.map((option) => ({
                            value: option.value,
                            label: tf(option.labelKey),
                        }))}
                    />
                </AppField>

                <Separator size="4" />

                {/* ── 压感 ───────────────────────────────────────────── */}
                <AppSwitchRow
                    label={tf("pen_input_pressure_enabled")}
                    hint={tf("pen_input_pressure_hint")}
                    checked={draft.pressureEnabled}
                    onCheckedChange={(checked) => patch({ pressureEnabled: checked })}
                />

                {draft.pressureEnabled && (
                    <>
                        <span className="hs-type-label font-medium">
                            {tf("pen_input_pressure_curve")}
                        </span>

                        <AppField label={tf("pen_input_pressure_dead_zone")}>
                            <AppNumberField
                                value={draft.pressureDeadZone}
                                unit="pressureFactor"
                                min={0}
                                max={0.5}
                                width={120}
                                ariaLabel={tf("pen_input_pressure_dead_zone")}
                                onCommit={(next) => patch({ pressureDeadZone: next })}
                            />
                        </AppField>

                        <AppField label={tf("pen_input_pressure_ceiling")}>
                            <AppNumberField
                                value={draft.pressureCeiling}
                                unit="pressureFactor"
                                min={0.1}
                                max={4}
                                width={120}
                                ariaLabel={tf("pen_input_pressure_ceiling")}
                                onCommit={(next) => patch({ pressureCeiling: next })}
                            />
                        </AppField>

                        <AppField label={tf("pen_input_pressure_min_gain")}>
                            <AppNumberField
                                value={draft.pressureMinGain}
                                unit="pressureFactor"
                                min={0.02}
                                max={4}
                                width={120}
                                ariaLabel={tf("pen_input_pressure_min_gain")}
                                onCommit={(next) => patch({ pressureMinGain: next })}
                            />
                        </AppField>

                        <AppField label={tf("pen_input_pressure_max_gain")}>
                            <AppNumberField
                                value={draft.pressureMaxGain}
                                unit="pressureFactor"
                                min={0.02}
                                max={8}
                                width={120}
                                ariaLabel={tf("pen_input_pressure_max_gain")}
                                onCommit={(next) => patch({ pressureMaxGain: next })}
                            />
                        </AppField>

                        <AppField label={tf("pen_input_pressure_gamma")}>
                            <AppNumberField
                                value={draft.pressureGamma}
                                unit="pressureFactor"
                                min={0.2}
                                max={4}
                                width={120}
                                ariaLabel={tf("pen_input_pressure_gamma")}
                                onCommit={(next) => patch({ pressureGamma: next })}
                            />
                        </AppField>
                    </>
                )}

                <Separator size="4" />

                {/* ── 触控板 / 触摸 ─────────────────────────────────── */}
                <AppSwitchRow
                    label={tf("pen_input_trackpad_pinch")}
                    checked={draft.trackpadPinchZoom}
                    onCheckedChange={(checked) => patch({ trackpadPinchZoom: checked })}
                />
                <AppSwitchRow
                    label={tf("pen_input_touch_ramp")}
                    hint={tf("pen_input_touch_ramp_hint")}
                    checked={draft.touchPrecisionRamp}
                    onCheckedChange={(checked) => patch({ touchPrecisionRamp: checked })}
                />
                <AppSwitchRow
                    label={tf("pen_input_tilt_enabled")}
                    hint={tf("pen_input_tilt_hint")}
                    checked={draft.tiltEnabled}
                    onCheckedChange={(checked) => patch({ tiltEnabled: checked })}
                />

                <Separator size="4" />

                {/* ── 接触读数 ───────────────────────────────────────── */}
                <AppField label={tf("pen_input_contact_readout")}>
                    <AppSelect
                        value={draft.contactReadout}
                        onValueChange={(value) =>
                            patch({ contactReadout: value as ContactReadoutMode })
                        }
                        options={READOUT_OPTIONS.map((option) => ({
                            value: option.value,
                            label: tf(option.labelKey),
                        }))}
                    />
                </AppField>

                {errorText ? (
                    <span className="text-qt-sm text-qt-danger">{errorText}</span>
                ) : notice ? (
                    <span className="text-qt-sm text-qt-text-muted">{notice}</span>
                ) : null}
            </AppForm>
        </AppDialog>
    );
}
