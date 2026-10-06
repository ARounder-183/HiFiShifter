// 插件共享粘贴按最终目标授权：换轨不禁参数粘贴，但仍禁止宿主片段写操作。
// @vitest-environment jsdom
import {afterEach,expect,test} from "vitest";
import {pluginAllowsAction,pluginAllowsEditChannel,isHostGeometryReadOnly,canEditHostClips} from "./hostCapabilities";
import {resolveActionByFocus,resolvePasteRoute} from "../features/keybindings/focusRouting";
import {DEFAULT_KEYBINDINGS} from "../features/keybindings/defaultKeybindings";

afterEach(()=>{delete window.__HFS_PLUGIN_BOOTSTRAP__;});

test("Ctrl+V after a host track switch reaches parameter paste before channel authorization",()=>{
    window.__HFS_PLUGIN_BOOTSTRAP__={version:1,viewId:"paste"};
    for(const surface of ["pianoRoll","trackHeader","timeline",null] as const) {
        const action=resolveActionByFocus(new KeyboardEvent("keydown",{key:"v",ctrlKey:true}),
            DEFAULT_KEYBINDINGS,surface,"select");
        expect(action).not.toBeNull();
        expect(pluginAllowsAction(action!,surface)).toBe(true);
        expect(pluginAllowsEditChannel(resolvePasteRoute("param",surface))).toBe(true);
        expect(pluginAllowsEditChannel(resolvePasteRoute("clips",surface))).toBe(false);
    }
});

test("standalone clipboard routes stay allowed; unrelated plugin geometry shortcuts stay blocked",()=>{
    expect(pluginAllowsEditChannel("hifi:timelineEditOp")).toBe(true);
    expect(pluginAllowsEditChannel(null)).toBe(false);
    window.__HFS_PLUGIN_BOOTSTRAP__={version:1,viewId:"paste"};
    expect(pluginAllowsAction("clip.delete","timeline")).toBe(false);
    expect(pluginAllowsAction("track.add","pianoRoll")).toBe(false);
    expect(pluginAllowsEditChannel("hifi:timelineEditOp")).toBe(false);
});

test("geometry editing requires explicit native write capability and standalone stays editable",()=>{
    expect(isHostGeometryReadOnly()).toBe(false);
    window.__HFS_PLUGIN_BOOTSTRAP__={version:1,viewId:"host-geometry"};
    expect(isHostGeometryReadOnly()).toBe(true);expect(canEditHostClips()).toBe(false);
    window.__HFS_PLUGIN_BOOTSTRAP__.clipEditing=true;
    expect(isHostGeometryReadOnly()).toBe(false);expect(canEditHostClips()).toBe(true);
});
