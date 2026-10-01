/**
 * 「匹配方式」弹出菜单：转写开关、宽严、各语言子开关。
 *
 * 【为什么是一个图标按钮 + 弹出菜单，而不是常驻下拉】文件浏览器与快速搜索的搜索行
 * 已经很挤（正则、仅音频、排序、加载指示）。再塞一个常驻下拉会挤掉排序控件；
 * 而三处搜索面各放一个下拉又是三份控件、三份状态。一个图标 + 共用组件最轻。
 *
 * 【为什么「关闭」与「智能/模糊」共处一个单选组】内部状态确实是两个字段
 * （`translit` 总开关 + `mode` 宽严），但界面上把它们合成一个三选一：
 * 用户脑子里只有「关 / 智能 / 模糊」三档，而分开之后「关闭」到底该不该清掉
 * `mode` 是个没有答案的问题。合并后：选「关闭」只关总开关，`mode` 原样保留，
 * 再打开时回到用户上次的宽严。
 */

import { DropdownMenu } from "@radix-ui/themes";

import { useI18n } from "../../../i18n/I18nProvider";
import {
    SEARCH_MODE_LABEL_KEY,
    effectiveSearchMode,
    type SearchSettings,
} from "../../../features/search/searchSettings";
import { AppIconButton } from "../../../ui";

export interface SearchModeMenuProps {
    settings: SearchSettings;
    /** 部分更新；调用方负责持久化。 */
    onChange: (patch: Partial<SearchSettings>) => void;
    /**
     * 正则模式是否激活。正则作用于**原文**，与转写互斥 —— 激活时菜单项置灰，
     * 免得用户改了开关却看不到任何效果。
     */
    regexActive?: boolean;
    /** 「在设置中管理…」的跳转。 */
    onOpenSettings?: () => void;
}

export function SearchModeMenu({
    settings,
    onChange,
    regexActive = false,
    onOpenSettings,
}: SearchModeMenuProps) {
    const { t } = useI18n();
    const activeMode = effectiveSearchMode(settings);
    const tooltip = t("search_match_mode");

    return (
        <DropdownMenu.Root>
            <DropdownMenu.Trigger>
                <AppIconButton
                    icon="Aa"
                    tooltip={tooltip}
                    // 正则模式下置为非激活：此时搜索根本不走转写，按钮亮着会误导。
                    active={activeMode !== "off" && !regexActive}
                    className="search-mode-menu__trigger"
                    style={{
                        fontFamily: "monospace",
                        fontSize: "var(--qt-fs-micro)",
                        width: 22,
                        height: 22,
                    }}
                />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content variant="soft" color="gray" align="end">
                {regexActive && (
                    <DropdownMenu.Label className="hs-type-caption">
                        {t("search_regex_disables_translit")}
                    </DropdownMenu.Label>
                )}
                <DropdownMenu.RadioGroup
                    value={activeMode}
                    onValueChange={(value) => {
                        // 「关闭」只动总开关，保留 `mode` —— 再打开时回到上次的宽严。
                        if (value === "off") {
                            onChange({ translit: false });
                            return;
                        }
                        onChange({ translit: true, mode: value as SearchSettings["mode"] });
                    }}
                >
                    {(["off", "smart", "fuzzy"] as const).map((mode) => (
                        <DropdownMenu.RadioItem key={mode} value={mode} disabled={regexActive}>
                            {t(SEARCH_MODE_LABEL_KEY[mode])}
                        </DropdownMenu.RadioItem>
                    ))}
                </DropdownMenu.RadioGroup>
                <DropdownMenu.Separator />
                <DropdownMenu.CheckboxItem
                    checked={settings.heteronym}
                    disabled={regexActive || activeMode === "off"}
                    onCheckedChange={(checked) => onChange({ heteronym: checked === true })}
                >
                    {t("search_translit_heteronym")}
                </DropdownMenu.CheckboxItem>
                <DropdownMenu.CheckboxItem
                    checked={settings.japaneseLongVowel}
                    disabled={regexActive || activeMode === "off"}
                    onCheckedChange={(checked) => onChange({ japaneseLongVowel: checked === true })}
                >
                    {t("search_translit_long_vowel")}
                </DropdownMenu.CheckboxItem>
                <DropdownMenu.CheckboxItem
                    checked={settings.koreanChoseong}
                    disabled={regexActive || activeMode === "off"}
                    onCheckedChange={(checked) => onChange({ koreanChoseong: checked === true })}
                >
                    {t("search_translit_choseong")}
                </DropdownMenu.CheckboxItem>
                <DropdownMenu.Separator />
                <DropdownMenu.CheckboxItem
                    checked={settings.showMatchReason}
                    onCheckedChange={(checked) => onChange({ showMatchReason: checked === true })}
                >
                    {t("search_show_match_reason")}
                </DropdownMenu.CheckboxItem>
                {onOpenSettings && (
                    <>
                        <DropdownMenu.Separator />
                        <DropdownMenu.Item onSelect={onOpenSettings}>
                            {t("search_open_settings")}
                        </DropdownMenu.Item>
                    </>
                )}
            </DropdownMenu.Content>
        </DropdownMenu.Root>
    );
}
