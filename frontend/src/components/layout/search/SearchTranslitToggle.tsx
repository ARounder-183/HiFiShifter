/**
 * 「拼音匹配」开关（左键开/关，右键出完整菜单）。
 *
 * 【为什么左键是开关】搜索行上的三个按钮必须能用同一种方式操作 —— 正则、仅显示
 * 媒体文件都是「点击 = 开/关」。原先这里是个三选一菜单，同一行里混着两种交互：
 * 用户得先猜「这个按钮点下去是会切换，还是会弹出一层东西」。现在三个按钮的左键
 * 语义完全一致。
 *
 * 【为什么右键放回原来的菜单】搜索时真正会改的只有「开 / 关」这一个决定，所以它
 * 值得占左键；但宽严（智能 / 模糊）与各语言子开关是**调完之后就不再动**的，把它们
 * 塞进左键会让每次点击都要先弹出菜单、再选一项。右键是「同一个控件的完整设置」，
 * 既不占位、也不打断左键的单一语义。
 *
 * 右键菜单的内容与原三选一菜单一致，外加「在设置中管理…」—— 那份设置同时作用于
 * 文件浏览器、快速搜索、快捷键面板与字体列表，需要一个常驻入口。
 *
 * 【图标为什么是「文A」而不是「Aa」】`Aa` 在 VS Code / JetBrains 一类编辑器里
 * 是**区分大小写**，放在搜索框旁边会被直接读错。「文A」是维基百科语言切换器用的
 * 那个记号，表示「不同文字之间」，与本功能（打拉丁字母找中日韩文字）正好对应；
 * 也和本行正则按钮的 `.*` 一样，是「一看就知道是什么的字符记号」而不是图形。
 *
 * 【为什么悬停提示里要写档位】开关只能表达「开 / 关」，表达不了「智能还是模糊」。
 * 不写出来，用户就得打开菜单才知道当前是哪一档。
 */

import { useState } from "react";

import { useI18n } from "../../../i18n/I18nProvider";
import {
    SEARCH_MODE_LABEL_KEY,
    effectiveSearchMode,
    searchModePatch,
    type SearchMode,
    type SearchSettings,
} from "../../../features/search/searchSettings";
import { AppContextMenu, AppIconButton } from "../../../ui";
import type { AppMenuItemSpec } from "../../../ui";

const MODES: readonly SearchMode[] = ["off", "smart", "fuzzy"];

export interface SearchTranslitToggleProps {
    settings: SearchSettings;
    /** 部分更新；调用方负责持久化。 */
    onChange: (patch: Partial<SearchSettings>) => void;
    /**
     * 正则模式是否激活。正则作用于**原文**，与转写互斥 —— 此时开关仍然显示真实
     * 设置（点击它切换的也仍是那个设置），只在提示里说明「此刻不生效」。
     * 让按钮显示成「关」会是假话：用户点一下它反而打开了转写。
     */
    regexActive?: boolean;
    /** 按钮边长（px）。同一行里的按钮应取同一尺寸。 */
    size?: number;
    /** 「在设置中管理…」的跳转；省略则菜单里不出现该项。 */
    onOpenSettings?: () => void;
}

export function SearchTranslitToggle({
    settings,
    onChange,
    regexActive = false,
    size = 22,
    onOpenSettings,
}: SearchTranslitToggleProps) {
    const { t } = useI18n();
    const [menuAt, setMenuAt] = useState<{ x: number; y: number } | null>(null);
    const mode = effectiveSearchMode(settings);
    const tooltip = `${t("search_match_mode")}: ${
        regexActive ? t("search_regex_disables_translit") : t(SEARCH_MODE_LABEL_KEY[mode])
    } · ${t("search_right_click_hint")}`;

    const items: AppMenuItemSpec[] = [
        ...MODES.map((candidate) => ({
            key: `mode-${candidate}`,
            label: t(SEARCH_MODE_LABEL_KEY[candidate]),
            checked: mode === candidate,
            onSelect: () => onChange(searchModePatch(candidate)),
        })),
        {
            key: "heteronym",
            label: t("search_translit_heteronym"),
            checked: settings.heteronym,
            disabled: !settings.translit,
            separatorBefore: true,
            onSelect: () => onChange({ heteronym: !settings.heteronym }),
        },
        {
            key: "longVowel",
            label: t("search_translit_long_vowel"),
            checked: settings.japaneseLongVowel,
            disabled: !settings.translit,
            onSelect: () => onChange({ japaneseLongVowel: !settings.japaneseLongVowel }),
        },
        {
            key: "choseong",
            label: t("search_translit_choseong"),
            checked: settings.koreanChoseong,
            disabled: !settings.translit,
            onSelect: () => onChange({ koreanChoseong: !settings.koreanChoseong }),
        },
        {
            key: "matchReason",
            label: t("search_show_match_reason"),
            checked: settings.showMatchReason,
            separatorBefore: true,
            onSelect: () => onChange({ showMatchReason: !settings.showMatchReason }),
        },
        ...(onOpenSettings
            ? [
                  {
                      key: "settings",
                      label: t("search_settings_title"),
                      separatorBefore: true,
                      onSelect: onOpenSettings,
                  },
              ]
            : []),
    ];

    return (
        <>
            <AppIconButton
                icon="文A"
                tooltip={tooltip}
                active={settings.translit}
                // 左键只翻总开关：`mode` 原样保留，所以关掉再打开会回到用户上次的宽严。
                onClick={() => onChange({ translit: !settings.translit })}
                onContextMenu={(event) => {
                    // 右键不是「另一种点击」而是「打开这个控件的完整设置」。
                    event.preventDefault();
                    event.stopPropagation();
                    setMenuAt({ x: event.clientX, y: event.clientY });
                }}
                className="search-translit-toggle"
                style={{ fontSize: "var(--qt-fs-micro)", width: size, height: size }}
            />
            {menuAt && (
                <AppContextMenu
                    x={menuAt.x}
                    y={menuAt.y}
                    ariaLabel={t("search_match_mode")}
                    items={items}
                    onClose={() => setMenuAt(null)}
                />
            )}
        </>
    );
}
