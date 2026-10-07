// 原GUI插件播放能力只由native宿主声明开放；录音/本地设备始终不开放。
// @vitest-environment jsdom
import {afterEach, expect, test} from "vitest";
import {canControlHostTransport,pluginAllowsAction} from "./hostCapabilities";
afterEach(()=>{delete window.__HFS_PLUGIN_BOOTSTRAP__;});
test("host capability enables play/pause but never recording",()=>{
    window.__HFS_PLUGIN_BOOTSTRAP__={version:1,viewId:"host-view",transportControl:true};
    expect(canControlHostTransport()).toBe(true);
    expect(pluginAllowsAction("playback.toggle",null)).toBe(true);
    expect(pluginAllowsAction("playback.stop",null)).toBe(true);
    expect(pluginAllowsAction("recording.start",null)).toBe(false);
});
test("optional ARA control absent leaves plugin transport disabled",()=>{
    window.__HFS_PLUGIN_BOOTSTRAP__={version:1,viewId:"host-view"};
    expect(canControlHostTransport()).toBe(false);
    expect(pluginAllowsAction("playback.toggle",null)).toBe(false);
});
