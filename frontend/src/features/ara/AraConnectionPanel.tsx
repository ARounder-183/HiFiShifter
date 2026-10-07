// ARA 宿主会话面板：复用主窗口参数编辑器。
import { useEffect, useState } from "react";
import { Button, Flex } from "@radix-ui/themes";
import { ReloadIcon, Link2Icon, UploadIcon, Cross2Icon } from "@radix-ui/react-icons";
import { araApi, araError, type AraInstance, type AraResult } from "./araApi";

export function AraConnectionPanel({
    dirty,
    onTimelineChanged,
}: {
    dirty: boolean;
    onTimelineChanged: () => Promise<unknown>;
}) {
    const [instances, setInstances] = useState<AraInstance[]>([]);
    const [selected, setSelected] = useState("");
    const [session, setSession] = useState<AraResult | null>(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const [status, setStatus] = useState("");
    const [replacement, setReplacement] = useState<"connect" | "refresh" | null>(null);

    async function run(action: () => Promise<void>) {
        setBusy(true);
        setError("");
        setStatus("");
        try {
            await action();
        } catch (err) {
            setError(araError(err));
        } finally {
            setBusy(false);
        }
    }
    async function list() {
        const found = await araApi.list();
        setInstances(found);
        setSelected((value) =>
            found.some((item) => item.instance_id === value)
                ? value
                : (found[0]?.instance_id ?? ""),
        );
    }
    useEffect(() => {
        void run(list);
    }, []);

    async function operate(kind: "connect" | "refresh", force = false) {
        if (dirty && !force) {
            setReplacement(kind);
            return;
        }
        setReplacement(null);
        await run(async () => {
            let result: AraResult;
            try {
                result =
                    kind === "connect"
                        ? await araApi.connect(selected, force)
                        : await araApi.refresh(force);
                if (!result.ok) throw new Error(result.error ?? "ARA 连接失败");
            } catch (err) {
                if (!force && araError(err).startsWith("dirty_project:")) setReplacement(kind);
                throw err;
            }
            setSession(result);
            await onTimelineChanged();
            setStatus(kind === "connect" ? "已连接" : "已刷新");
        });
    }

    return (
        <div
            aria-label="ARA / REAPER"
            style={{ flexShrink: 0, borderBottom: "1px solid var(--gray-6)", padding: "5px 12px" }}
        >
            <Flex align="center" gap="2" wrap="wrap">
                <span className="hs-type-label font-bold">
                    ARA / REAPER
                </span>
                <select
                    aria-label="ARA 实例"
                    value={selected}
                    disabled={busy || !!session}
                    onChange={(event) => setSelected(event.target.value)}
                    style={{
                        minWidth: 150,
                        maxWidth: "min(280px, 100%)",
                        height: 26,
                        background: "var(--color-panel-solid)",
                        color: "var(--gray-12)",
                        border: "1px solid var(--gray-7)",
                        borderRadius: 4,
                        fontSize: "var(--qt-fs-xs)",
                    }}
                >
                    {!instances.length && <option value="">未发现实例</option>}
                    {instances.map((instance) => (
                        <option key={instance.instance_id} value={instance.instance_id}>
                            {instance.name} ({instance.pid})
                        </option>
                    ))}
                </select>
                <Button
                    size="1"
                    variant="soft"
                    disabled={busy}
                    onClick={() => void run(list)}
                    title="刷新实例"
                >
                    <ReloadIcon />
                    刷新实例
                </Button>
                <Button
                    size="1"
                    variant="soft"
                    disabled={busy || !selected || !!session}
                    onClick={() => void operate("connect")}
                >
                    <Link2Icon />
                    连接
                </Button>
                <Button
                    size="1"
                    disabled={busy || !session}
                    onClick={() =>
                        void run(async () => {
                            const result = await araApi.submit();
                            if (!result.ok) throw new Error(result.error ?? "ARA 提交失败");
                            setSession(result);
                            await onTimelineChanged();
                            setStatus("已提交");
                        })
                    }
                >
                    <UploadIcon />
                    提交到REAPER
                </Button>
                <Button
                    size="1"
                    variant="soft"
                    disabled={busy || !session}
                    onClick={() => void operate("refresh")}
                >
                    <ReloadIcon />
                    刷新宿主
                </Button>
                <Button
                    size="1"
                    variant="soft"
                    disabled={busy || !session}
                    onClick={() =>
                        void run(async () => {
                            const result = await araApi.disconnect();
                            if (!result.ok) throw new Error(result.error ?? "ARA 断开失败");
                            setSession(null);
                            setStatus("已断开");
                        })
                    }
                >
                    <Cross2Icon />
                    断开
                </Button>
                <span className="hs-type-caption">
                    不支持倒放
                </span>
                <span className="hs-type-caption" role="status">
                    {busy ? "处理中..." : status}
                    {session && ` · r${session.revision} / m${session.model_revision}`}
                </span>
            </Flex>
            {replacement && (
                <Flex
                    align="center"
                    gap="2"
                    wrap="wrap"
                    role="alertdialog"
                    aria-label="替换未保存工程"
                    style={{ marginTop: 6 }}
                >
                    <span className="hs-type-label">当前工程有未保存修改，替换为宿主快照？</span>
                    <Button
                        size="1"
                        color="red"
                        disabled={busy}
                        onClick={() => void operate(replacement, true)}
                    >
                        替换未保存工程
                    </Button>
                    <Button
                        size="1"
                        variant="soft"
                        disabled={busy}
                        onClick={() => setReplacement(null)}
                    >
                        取消
                    </Button>
                </Flex>
            )}
            {error && (
                <span className="hs-type-label"
                    role="alert"
                    style={{ display: "block", overflowWrap: "anywhere", marginTop: 4 }}
                >
                    {error}
                </span>
            )}
        </div>
    );
}
