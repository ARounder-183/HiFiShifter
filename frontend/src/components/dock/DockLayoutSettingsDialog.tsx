/*
 * 布局设置对话框。
 *
 * 与仓库既有的设置对话框同一形态（`TimelineDisplaySettingsDialog`）：
 * Radix `Dialog` + 行式布局，`onKeyDown` 必须 `stopPropagation`，否则全局
 * 快捷键会在输入时抢走按键。
 *
 * 【写入策略】每项改动立即 `dispatch(setDockSettings(...))`，由 `App` 的去抖
 * 副作用负责落盘 —— 对话框不做"确定/取消"两态。这类开关的语义就是即时生效，
 * 给它们加一层待提交状态只会制造"改了没反应"的困惑。
 */

import { Flex, Select, Text, TextField } from "@radix-ui/themes";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { setDockSettings, setTabPosition } from "../../features/dock/dockSlice";
import type { DockSettings } from "../../features/dock/dockSettings";
import { useI18n } from "../../i18n/I18nProvider";
import { applySelectWheelChange } from "../../utils/selectWheel";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm, AppSwitchRow } from "../../ui/Field";

export interface DockLayoutSettingsDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

const DOCK_MODIFIERS = ["primary", "alt", "shift", "none"] as const;

export function DockLayoutSettingsDialog({ open, onOpenChange }: DockLayoutSettingsDialogProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const settings = useAppSelector((state) => state.dock.settings);
    const layoutTabPosition = useAppSelector((state) => state.dock.layout.tabPosition);
    const presetNames = useAppSelector((state) => Object.keys(state.dock.layout.presets ?? {}));

    const patch = (next: DockSettings) => dispatch(setDockSettings(next));

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("layout_settings_title")}
            description={
                <Text size="2" color="gray">
                    {tAny("layout_settings_hint")}
                </Text>
            }
            size="md"
            actions={[{ id: "close", label: t("close"), onClick: () => onOpenChange(false) }]}
        >
            <AppForm>
                <AppField label={tAny("layout_setting_dock_modifier")}>
                    <Select.Root
                        value={settings.dockModifier}
                        onValueChange={(value) =>
                            patch({ dockModifier: value as DockSettings["dockModifier"] })
                        }
                    >
                        <Select.Trigger
                            onWheel={(event) =>
                                applySelectWheelChange({
                                    event,
                                    currentValue: settings.dockModifier,
                                    options: DOCK_MODIFIERS,
                                    onChange: (dockModifier) => patch({ dockModifier }),
                                })
                            }
                        />
                        <Select.Content>
                            <Select.Item value="primary">
                                {tAny("layout_modifier_primary")}
                            </Select.Item>
                            <Select.Item value="alt">{tAny("layout_modifier_alt")}</Select.Item>
                            <Select.Item value="shift">{tAny("layout_modifier_shift")}</Select.Item>
                            <Select.Item value="none">{tAny("layout_modifier_none")}</Select.Item>
                        </Select.Content>
                    </Select.Root>
                </AppField>
                <Text size="1" color="gray">
                    {tAny("layout_setting_dock_modifier_hint")}
                </Text>

                <AppField label={tAny("layout_setting_edge_band")}>
                    <NumberField
                        value={settings.edgeBandPx}
                        min={8}
                        max={120}
                        suffix={tAny("layout_unit_px")}
                        onChange={(edgeBandPx) => patch({ edgeBandPx })}
                    />
                </AppField>

                <AppField label={tAny("layout_setting_snap_px")}>
                    <NumberField
                        value={settings.floatSnapThresholdPx}
                        min={0}
                        max={64}
                        suffix={tAny("layout_unit_px")}
                        onChange={(floatSnapThresholdPx) => patch({ floatSnapThresholdPx })}
                    />
                </AppField>

                <AppField label={tAny("layout_setting_tab_position")}>
                    <Select.Root
                        value={layoutTabPosition}
                        onValueChange={(value) =>
                            dispatch(setTabPosition(value === "top" ? "top" : "bottom"))
                        }
                    >
                        <Select.Trigger />
                        <Select.Content>
                            <Select.Item value="bottom">
                                {tAny("layout_tab_position_bottom")}
                            </Select.Item>
                            <Select.Item value="top">{tAny("layout_tab_position_top")}</Select.Item>
                        </Select.Content>
                    </Select.Root>
                </AppField>

                <AppField label={tAny("layout_setting_save_delay")}>
                    <NumberField
                        value={settings.saveDebounceMs}
                        min={0}
                        max={5000}
                        suffix={tAny("layout_unit_ms")}
                        onChange={(saveDebounceMs) => patch({ saveDebounceMs })}
                    />
                </AppField>

                <AppField label={tAny("layout_setting_double_click")}>
                    <Select.Root
                        value={settings.doubleClickHeaderAction}
                        onValueChange={(value) =>
                            patch({
                                doubleClickHeaderAction:
                                    value as DockSettings["doubleClickHeaderAction"],
                            })
                        }
                    >
                        <Select.Trigger />
                        <Select.Content>
                            <Select.Item value="toggleFloat">{tAny("layout_dc_float")}</Select.Item>
                            <Select.Item value="maximize">{tAny("layout_dc_maximize")}</Select.Item>
                            <Select.Item value="collapse">{tAny("layout_dc_collapse")}</Select.Item>
                            <Select.Item value="none">{tAny("layout_dc_none")}</Select.Item>
                        </Select.Content>
                    </Select.Root>
                </AppField>

                <AppField label={tAny("layout_setting_startup")}>
                    <Select.Root
                        value={settings.startupLayout}
                        onValueChange={(value) => patch({ startupLayout: value })}
                    >
                        <Select.Trigger />
                        <Select.Content>
                            <Select.Item value="last">{tAny("layout_startup_last")}</Select.Item>
                            {presetNames.map((name) => (
                                <Select.Item key={name} value={name}>
                                    {name}
                                </Select.Item>
                            ))}
                        </Select.Content>
                    </Select.Root>
                </AppField>

                <AppSwitchRow
                    label={tAny("layout_setting_show_preview")}
                    checked={settings.showDropPreview}
                    onCheckedChange={(showDropPreview) => patch({ showDropPreview })}
                />
                <AppSwitchRow
                    label={tAny("layout_setting_tab_bar_single")}
                    checked={settings.showTabBarWhenSingle}
                    onCheckedChange={(showTabBarWhenSingle) => patch({ showTabBarWhenSingle })}
                />
                <AppSwitchRow
                    label={tAny("layout_setting_tab_icons")}
                    checked={settings.tabIcons}
                    onCheckedChange={(tabIcons) => patch({ tabIcons })}
                />
                <AppSwitchRow
                    label={tAny("layout_setting_float_snap")}
                    checked={settings.floatSnapEnabled}
                    onCheckedChange={(floatSnapEnabled) => patch({ floatSnapEnabled })}
                />
                <AppSwitchRow
                    label={tAny("layout_setting_restore_floats")}
                    checked={settings.floatRestoreOnStartup}
                    onCheckedChange={(floatRestoreOnStartup) => patch({ floatRestoreOnStartup })}
                />
                <AppSwitchRow
                    label={tAny("layout_setting_confirm_reset")}
                    checked={settings.confirmResetLayout}
                    onCheckedChange={(confirmResetLayout) => patch({ confirmResetLayout })}
                />
            </AppForm>
        </AppDialog>
    );
}

/** 数值输入：与 `RenderCacheDialog` 同一形态（Radix `TextField.Root` + 单位后缀）。 */
function NumberField({
    value,
    min,
    max,
    suffix,
    onChange,
}: {
    value: number;
    min: number;
    max: number;
    suffix: string;
    onChange: (value: number) => void;
}) {
    return (
        <Flex align="center" gap="2">
            <TextField.Root
                size="2"
                type="number"
                min={min}
                max={max}
                value={String(value)}
                onChange={(event) => {
                    const next = Number(event.target.value);
                    if (!Number.isFinite(next)) return;
                    onChange(Math.min(max, Math.max(min, Math.round(next))));
                }}
                style={{ width: 110 }}
            />
            <Text size="1" color="gray">
                {suffix}
            </Text>
        </Flex>
    );
}
