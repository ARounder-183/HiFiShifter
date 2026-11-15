/*
 * 状态栏的「参数曲线取数中」指示片。
 *
 * 【为什么单独成一个组件】这个状态只服务这一处显示，却曾经住在包裹整个
 * AppInner 的 Context 里 —— 于是它每翻转一次，整棵应用树（时间轴、参数编辑器、
 * 全部面板）就重渲染一次，把用户的绘制笔画打断。抽成独立组件并只订阅
 * interaction 切片里的派生子集后，重渲染被限制在这一个 span 上。
 *
 * 【为什么选择器返回布尔而不是整张表】react-redux 默认按引用比较；选择器每次
 * 现算 anyLoading，只有 true/false 翻转才触发本组件重渲染。
 */

import { useAppSelector } from "../../app/hooks";
import { selectAnyPianoRollLoading } from "../../features/session/interactionSlice";
import { useI18n } from "../../i18n/I18nProvider";
import { AppStatusChip } from "../../ui";

export function ParamDataLoadingChip() {
    const { t } = useI18n();
    const loading = useAppSelector(selectAnyPianoRollLoading);

    if (!loading) return null;

    return <AppStatusChip tone="accent">{t("common_loading")}</AppStatusChip>;
}
