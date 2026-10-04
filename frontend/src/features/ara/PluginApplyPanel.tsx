// REAPER内原GUI的自动应用状态；没有实例选择、外部app或手动提交按钮。
import { useEffect, useRef, useState } from "react";
import { Button, Flex } from "@radix-ui/themes";
import { invoke } from "../../services/invoke";
import { listen } from "../../services/hostEvents";
type ApplyState = { generation: number; applied_generation: number; pending: boolean;
    connected: boolean; ready?: boolean; error: string | null };

/** 显示真正后台状态；初始化等待只读重试，冲突刷新必须显式确认替换本地曲线。 */
export function PluginApplyPanel({ onTimelineChanged }: { onTimelineChanged: () => Promise<unknown> }) {
    const [state, setState] = useState<ApplyState | null>(null);
    const [failure, setFailure] = useState("");
    const [confirm, setConfirm] = useState(false);
    const [busy, setBusy] = useState(false);
    const timelineChanged = useRef(onTimelineChanged);
    timelineChanged.current = onTimelineChanged;
    useEffect(() => {
        let disposed = false;
        let inFlight = false;
        const subscription = listen<ApplyState>("plugin_apply_state", (event) => {
            if (!disposed) setState(event.payload);
        });
        async function poll() {
            if (inFlight || disposed) return;
            inFlight = true;
            try {
                const current = await invoke<ApplyState>("plugin_get_apply_state");
                if (!disposed) { setState(current); setFailure(""); }
                if (!current.ready && !disposed) await timelineChanged.current().catch(() => undefined);
            } catch (error) { if (!disposed) setFailure(String(error)); }
            finally { inFlight = false; }
        }
        void poll();
        const timer = window.setInterval(() => void poll(), 1000);
        return () => { disposed = true; clearInterval(timer); void subscription.then((off) => off()).catch(() => {}); };
    }, []);
    async function refresh(force = false) {
        if (state?.pending && !force) { setConfirm(true); return; }
        setBusy(true); setConfirm(false);
        try {
            await invoke("plugin_refresh", force);
            await onTimelineChanged();
            setState(await invoke<ApplyState>("plugin_get_apply_state"));
            setFailure("");
        } catch (error) { setFailure(String(error)); }
        finally { setBusy(false); }
    }
    const error = failure || state?.error;
    return <div aria-label="ARA 自动应用" style={{ flexShrink: 0, borderBottom: "1px solid var(--gray-6)", padding: "5px 12px" }}>
        <Flex align="center" gap="2" wrap="wrap">
            <span className="hs-type-label font-bold">HiFiShifter · REAPER / ARA</span>
            <span className="hs-type-label" role="status" style={{color:error ? "var(--qt-danger-text)" : state?.pending ? "var(--qt-text)" : "var(--qt-text-muted)"}}>
                {error ? `尚未应用：${error}` : !state?.ready ? "等待宿主音频" : state.pending ? "正在自动应用…" : "已应用"}
            </span>
            {state && <span className="hs-type-caption">编辑 {state.generation} / 音频 {state.applied_generation}</span>}
            <Button size="1" variant="soft" disabled={busy} onClick={() => void refresh()}>重新载入宿主</Button>
            <span className="hs-type-caption">文件、片段位置及播放由 REAPER 控制 · 不支持倒放</span>
        </Flex>
        {confirm && <Flex role="alertdialog" aria-label="重新载入宿主" align="center" gap="2" style={{ marginTop: 6 }}>
            <span className="hs-type-label">当前仍有未应用编辑。重新载入会替换本地曲线，继续？</span>
            <Button size="1" color="red" onClick={() => void refresh(true)}>确认重新载入</Button>
            <Button size="1" variant="soft" onClick={() => setConfirm(false)}>取消</Button>
        </Flex>}
    </div>;
}
