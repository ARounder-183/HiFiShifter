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

import { Text } from "@radix-ui/themes";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { setDockSettings, setTabPosition } from "../../features/dock/dockSlice";
import type { DockSettings } from "../../features/dock/dockSettings";
import { useI18n } from "../../i18n/I18nProvider";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm, AppSwitchRow } from "../../ui/Field";
import { AppNumberField, AppSelect } from "../../ui";

export interface DockLayoutSettingsDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function DockLayoutSettingsDialog({ open, onOpenChange }: DockLayoutSettingsDialogProps) {
    const dispatch = useAppDispatch();
    const { t, tf } = useI18n();
    const settings = useAppSelector((state) => state.dock.settings);
    const layoutTabPosition = useAppSelector((state) => state.dock.layout.tabPosition);
    const presetNames = useAppSelector((state) => Object.keys(state.dock.layout.presets ?? {}));

    const patch = (next: DockSettings) => dispatch(setDockSettings(next));

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("layout_settings_title")}
            description={tf("layout_settings_hint")}
            size="md"
            actions={[{ id: "close", label: t("close"), onClick: () => onOpenChange(false) }]}
        >
            {/* 混排表单：字段与开关共用标签列，因此显式声明 aligned */}
            <AppForm booleanRow="aligned">
                <AppField label={tf("layout_setting_dock_modifier")}>
                    <AppSelect
                        value={settings.dockModifier}
                        onValueChange={(value) =>
                            patch({ dockModifier: value as DockSettings["dockModifier"] })
                        }
                        options={[
                            { value: "primary", label: tf("layout_modifier_primary") },
                            { value: "alt", label: tf("layout_modifier_alt") },
                            { value: "shift", label: tf("layout_modifier_shift") },
                            { value: "none", label: tf("layout_modifier_none") },
                        ]}
                    />
                </AppField>
                <Text size="1" color="gray">
                    {tf("layout_setting_dock_modifier_hint")}
                </Text>

                <AppField label={tf("layout_setting_edge_band")}>
                    <AppNumberField
                        value={settings.edgeBandPx}
                        unit="pixels"
                        min={8}
                        max={120}
                        width={110}
                        suffix={tf("layout_unit_px")}
                        ariaLabel={tf("layout_setting_edge_band")}
                        onCommit={(edgeBandPx) => patch({ edgeBandPx })}
                    />
                </AppField>

                <AppField label={tf("layout_setting_snap_px")}>
                    <AppNumberField
                        value={settings.floatSnapThresholdPx}
                        unit="pixels"
                        min={0}
                        max={64}
                        width={110}
                        suffix={tf("layout_unit_px")}
                        ariaLabel={tf("layout_setting_snap_px")}
                        onCommit={(floatSnapThresholdPx) => patch({ floatSnapThresholdPx })}
                    />
                </AppField>

                <AppField label={tf("layout_setting_tab_position")}>
                    <AppSelect
                        value={layoutTabPosition}
                        onValueChange={(value) =>
                            dispatch(setTabPosition(value === "top" ? "top" : "bottom"))
                        }
                        options={[
                            { value: "bottom", label: tf("layout_tab_position_bottom") },
                            { value: "top", label: tf("layout_tab_position_top") },
                        ]}
                    />
                </AppField>

                <AppField label={tf("layout_setting_save_delay")}>
                    <AppNumberField
                        value={settings.saveDebounceMs}
                        unit="milliseconds"
                        min={0}
                        max={5000}
                        width={110}
                        suffix={tf("layout_unit_ms")}
                        ariaLabel={tf("layout_setting_save_delay")}
                        onCommit={(saveDebounceMs) => patch({ saveDebounceMs })}
                    />
                </AppField>

                <AppField label={tf("layout_setting_double_click")}>
                    <AppSelect
                        value={settings.doubleClickHeaderAction}
                        onValueChange={(value) =>
                            patch({
                                doubleClickHeaderAction:
                                    value as DockSettings["doubleClickHeaderAction"],
                            })
                        }
                        options={[
                            { value: "toggleFloat", label: tf("layout_dc_float") },
                            { value: "maximize", label: tf("layout_dc_maximize") },
                            { value: "collapse", label: tf("layout_dc_collapse") },
                            { value: "none", label: tf("layout_dc_none") },
                        ]}
                    />
                </AppField>

                <AppField label={tf("layout_setting_startup")}>
                    <AppSelect
                        value={settings.startupLayout}
                        onValueChange={(value) => patch({ startupLayout: value })}
                        options={[
                            { value: "last", label: tf("layout_startup_last") },
                            ...presetNames.map((name) => ({ value: name, label: name })),
                        ]}
                    />
                </AppField>

                <AppSwitchRow
                    label={tf("layout_setting_show_preview")}
                    checked={settings.showDropPreview}
                    onCheckedChange={(showDropPreview) => patch({ showDropPreview })}
                />
                <AppSwitchRow
                    label={tf("layout_setting_tab_bar_single")}
                    checked={settings.showTabBarWhenSingle}
                    onCheckedChange={(showTabBarWhenSingle) => patch({ showTabBarWhenSingle })}
                />
                <AppSwitchRow
                    label={tf("layout_setting_tab_icons")}
                    checked={settings.tabIcons}
                    onCheckedChange={(tabIcons) => patch({ tabIcons })}
                />
                <AppSwitchRow
                    label={tf("layout_setting_float_snap")}
                    checked={settings.floatSnapEnabled}
                    onCheckedChange={(floatSnapEnabled) => patch({ floatSnapEnabled })}
                />
                <AppSwitchRow
                    label={tf("layout_setting_restore_floats")}
                    checked={settings.floatRestoreOnStartup}
                    onCheckedChange={(floatRestoreOnStartup) => patch({ floatRestoreOnStartup })}
                />
                <AppSwitchRow
                    label={tf("layout_setting_confirm_reset")}
                    checked={settings.confirmResetLayout}
                    onCheckedChange={(confirmResetLayout) => patch({ confirmResetLayout })}
                />
            </AppForm>
        </AppDialog>
    );
}
