// 参数刷新显示策略：插件同作用域保留已提交曲线，独立App和切作用域仍使用原清窗规则。
/** 强制重新取数不等于必须先清空画面；插件DSP等待期保持当前scope的可见数据。 */
export function shouldClearParamRefresh(scopeChanged: boolean, plugin: boolean): boolean {
    return scopeChanged || !plugin;
}
