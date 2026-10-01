/**
 * 搜索与匹配设置。
 *
 * 【为什么是独立对话框】这份设置同时作用于文件浏览器、快速搜索、快捷键面板与
 * 字体列表 —— 它不属于任何**一个**面板，因此不挂在某个面板的设置页里，而是与
 * 「吸附/网格设置」「时间显示设置」同级的全局设置：选项菜单里一个入口，
 * 打开后是一个小对话框。
 *
 * 【为什么「关闭」与「智能/模糊」共处一个下拉】内部状态确实是两个字段
 * （`translit` 总开关 + `mode` 宽严），但用户脑子里只有「关 / 智能 / 模糊」三档。
 * 合并后：选「关闭」只关总开关、`mode` 原样保留，再打开时回到用户上次的宽严 ——
 * 这比把两件事拆成两个控件更好回答「关闭到底该不该清掉 mode」。
 */

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    SEARCH_MODE_LABEL_KEY,
    effectiveSearchMode,
    searchModePatch,
    type SearchMode,
    type SearchSettings,
} from "../../features/search/searchSettings";
import { persistUiSettings, setSearchSettings } from "../../features/session/sessionSlice";
import { AppSelect } from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm, AppSwitchRow } from "../../ui/Field";

const MODES: readonly SearchMode[] = ["off", "smart", "fuzzy"];

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function SearchSettingsDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    const settings = useAppSelector((state: RootState) => state.session.searchSettings);
    const { tf } = useI18n();

    /** 改动即写入全局设置并持久化（与其它设置对话框一致，不需要「应用」按钮）。 */
    const update = (patch: Partial<SearchSettings>) => {
        dispatch(setSearchSettings(patch));
        void dispatch(persistUiSettings());
    };

    const activeMode = effectiveSearchMode(settings);

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("search_settings_title")}
            description={tf("search_settings_desc")}
            size="sm"
            actions={[{ id: "close", label: tf("close"), onClick: () => onOpenChange(false) }]}
        >
            <AppForm booleanRow="leading">
                <AppField label={tf("search_match_mode")}>
                    <AppSelect
                        value={activeMode}
                        onValueChange={(value) => update(searchModePatch(value as SearchMode))}
                        options={MODES.map((mode) => ({
                            value: mode,
                            label: tf(SEARCH_MODE_LABEL_KEY[mode]),
                        }))}
                    />
                </AppField>

                {/*
                  三个子开关各自独立：韩文初声对中文用户是纯噪音，日文长音对韩文用户
                  是纯噪音。让它们各自可关，比一个「宽松匹配」总开关更精确。
                */}
                <AppSwitchRow
                    control="checkbox"
                    label={tf("search_translit_heteronym")}
                    checked={settings.heteronym}
                    disabled={!settings.translit}
                    onCheckedChange={(checked) => update({ heteronym: checked })}
                />
                <AppSwitchRow
                    control="checkbox"
                    label={tf("search_translit_long_vowel")}
                    checked={settings.japaneseLongVowel}
                    disabled={!settings.translit}
                    onCheckedChange={(checked) => update({ japaneseLongVowel: checked })}
                />
                <AppSwitchRow
                    control="checkbox"
                    label={tf("search_translit_choseong")}
                    checked={settings.koreanChoseong}
                    disabled={!settings.translit}
                    onCheckedChange={(checked) => update({ koreanChoseong: checked })}
                />
                <AppSwitchRow
                    control="checkbox"
                    label={tf("search_show_match_reason")}
                    checked={settings.showMatchReason}
                    onCheckedChange={(checked) => update({ showMatchReason: checked })}
                />
            </AppForm>
        </AppDialog>
    );
}
