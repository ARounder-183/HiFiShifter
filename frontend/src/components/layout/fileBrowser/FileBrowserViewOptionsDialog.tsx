/**
 * 文件浏览器视图选项。
 *
 * 【为什么是对话框而不是全塞进右键菜单】这套选项有 10 项。常用的三四个（排序、
 * 降序、文件夹优先、隐藏文件）留在右键菜单里随手可切；其余"调完就不动"的项收进
 * 这个对话框 —— 与 `SearchTranslitToggle` 把宽严档位放进右键菜单是同一条理由：
 * 高频的占左键/顶层，低频的不占位。
 *
 * 【为什么没有「应用」按钮】与其它设置对话框一致：改动即写全局设置并持久化。
 */

import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import type { RootState } from "../../../app/store";
import { useI18n } from "../../../i18n/I18nProvider";
import {
    FILE_BROWSER_DENSITY_LABEL_KEY,
    FILE_BROWSER_DETAILS_LABEL_KEY,
    FILE_BROWSER_SORT_LABEL_KEY,
    type FileBrowserDensity,
    type FileBrowserDetailsColumn,
    type FileBrowserSortKey,
    type FileBrowserViewOptions,
} from "../../../features/fileBrowser/fileBrowserViewOptions";
import { persistUiSettings, setFileBrowserView } from "../../../features/session/sessionSlice";
import { AppSelect } from "../../../ui";
import { AppDialog } from "../../../ui/Dialog";
import { AppField, AppForm, AppSwitchRow } from "../../../ui/Field";

const SORT_KEYS: readonly FileBrowserSortKey[] = ["name", "date", "size"];
const DENSITIES: readonly FileBrowserDensity[] = ["compact", "comfortable"];
const DETAILS_COLUMNS: readonly FileBrowserDetailsColumn[] = ["size", "date", "none"];

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function FileBrowserViewOptionsDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    const view = useAppSelector((state: RootState) => state.session.fileBrowserView);
    const { t } = useI18n();

    /** 改动即写入全局设置并持久化。 */
    const update = (patch: Partial<FileBrowserViewOptions>) => {
        dispatch(setFileBrowserView(patch));
        void dispatch(persistUiSettings());
    };

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={t("fb_view_options")}
            size="sm"
            actions={[{ id: "close", label: t("close"), onClick: () => onOpenChange(false) }]}
        >
            <AppForm booleanRow="leading">
                <AppField label={t("fb_sort_label")}>
                    <AppSelect
                        value={view.sortMode}
                        onValueChange={(value) => update({ sortMode: value as FileBrowserSortKey })}
                        options={SORT_KEYS.map((key) => ({
                            value: key,
                            label: t(FILE_BROWSER_SORT_LABEL_KEY[key]),
                        }))}
                    />
                </AppField>
                <AppSwitchRow
                    control="checkbox"
                    label={t("fb_sort_descending")}
                    checked={view.sortDescending}
                    onCheckedChange={(checked) => update({ sortDescending: checked })}
                />
                <AppSwitchRow
                    control="checkbox"
                    label={t("fb_folders_first")}
                    checked={view.foldersFirst}
                    onCheckedChange={(checked) => update({ foldersFirst: checked })}
                />

                {/*
                  字段标签说明"这一项在设置什么"，选项标签才是"取哪个值"。
                  此前这里错把第一个选项当成了字段标签，界面上于是出现
                  「大小：大小 / 修改日期 / 无」与「紧凑：紧凑 / 舒适」——
                  同一句话里既当问题又当答案。
                */}
                <AppField label={t("fb_details_column")}>
                    <AppSelect
                        value={view.detailsColumn}
                        onValueChange={(value) =>
                            update({ detailsColumn: value as FileBrowserDetailsColumn })
                        }
                        options={DETAILS_COLUMNS.map((column) => ({
                            value: column,
                            label: t(FILE_BROWSER_DETAILS_LABEL_KEY[column]),
                        }))}
                    />
                </AppField>

                <AppField label={t("fb_density")}>
                    <AppSelect
                        value={view.density}
                        onValueChange={(value) => update({ density: value as FileBrowserDensity })}
                        options={DENSITIES.map((density) => ({
                            value: density,
                            label: t(FILE_BROWSER_DENSITY_LABEL_KEY[density]),
                        }))}
                    />
                </AppField>

                <AppSwitchRow
                    control="checkbox"
                    label={t("fb_show_hidden")}
                    checked={view.showHiddenFiles}
                    onCheckedChange={(checked) => update({ showHiddenFiles: checked })}
                />
                <AppSwitchRow
                    control="checkbox"
                    label={t("fb_show_path_hint")}
                    checked={view.showPathHint}
                    onCheckedChange={(checked) => update({ showPathHint: checked })}
                />
                <AppSwitchRow
                    control="checkbox"
                    label={t("fb_audio_only")}
                    checked={view.mediaOnly}
                    onCheckedChange={(checked) => update({ mediaOnly: checked })}
                />
                <AppSwitchRow
                    control="checkbox"
                    label={t("fb_preview_on_click")}
                    checked={view.previewOnClick}
                    onCheckedChange={(checked) => update({ previewOnClick: checked })}
                />
                <AppSwitchRow
                    control="checkbox"
                    label={t("fb_preview_on_navigate")}
                    checked={view.previewOnNavigate}
                    onCheckedChange={(checked) => update({ previewOnNavigate: checked })}
                />
                <AppSwitchRow
                    control="checkbox"
                    label={t("fb_status_bar")}
                    checked={view.statusBarVisible}
                    onCheckedChange={(checked) => update({ statusBarVisible: checked })}
                />
            </AppForm>
        </AppDialog>
    );
}
