// App/插件共用时间轴折叠入口：复用停靠系统隐藏/恢复，面板实例和参数选轨保持存活。
import { Button } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector, useAppStore } from "../../app/hooks";
import { openPanelById } from "../../features/dock/dockApi";
import { collapseForm, restoreCollapsedForm } from "../../features/dock/dockSlice";
import { isFormVisible } from "../../features/dock/dockTree";
import { PANEL_TIMELINE, PANEL_PARAM_EDITOR } from "../dock/registerBuiltinPanels";

/** 收起时确保参数面板可见；不重建Timeline、不切换轨道，两种运行模式行为一致。 */
export function TimelinePanelToggle() {
    const dispatch=useAppDispatch();const store=useAppStore();
    const visible=useAppSelector(state=>state.dock.layout.order.some(id=>
        state.dock.layout.forms[id]?.panelId===PANEL_TIMELINE&&isFormVisible(state.dock.layout,id)));
    return <Button size="1" variant="soft" aria-expanded={visible} aria-label={visible?"折叠轨道面板":"展开轨道面板"}
        onClick={()=>{
            if (visible) {
                openPanelById(dispatch,store.getState,PANEL_PARAM_EDITOR);
                const layout=store.getState().dock.layout;
                const formId=layout.order.find(id=>layout.forms[id]?.panelId===PANEL_TIMELINE&&isFormVisible(layout,id));
                if (formId) dispatch(collapseForm(formId));
            } else {
                const dock=store.getState().dock;
                const formId=Object.keys(dock.collapsedForms).find(id=>dock.layout.forms[id]?.panelId===PANEL_TIMELINE);
                if (formId) dispatch(restoreCollapsedForm(formId));
                else openPanelById(dispatch,store.getState,PANEL_TIMELINE);
            }
        }}>{visible?"折叠轨道面板":"展开轨道面板"}</Button>;
}
