// @vitest-environment jsdom
import { act, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, test, vi } from "vitest";

const { forms } = vi.hoisted(() => ({
    forms: {
        timeline: { panelId: "timeline" },
        editor: { panelId: "paramEditor" },
    },
}));
vi.mock("../../app/hooks", () => ({ useAppSelector: () => forms }));
vi.mock("../../app/store", () => ({
    store: { getState: () => ({ dock: { layout: { forms } } }) },
}));
vi.mock("../../features/dock/panelRegistry", () => ({
    getPanel: () => ({ defaultWidth: 400, defaultHeight: 300 }),
}));

import { useDockSlot } from "./useDockSlot";
import { acquirePanelHost, resetPanelHostsForTests } from "./panelHostRegistry";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root | null = null;
let mount: HTMLDivElement | null = null;

function Slot({ formId }: { formId: string }) {
    const ref = useDockSlot(formId);
    return <div data-test-slot ref={ref} />;
}

afterEach(async () => {
    if (root) await act(async () => root!.unmount());
    root = null;
    mount?.remove();
    mount = null;
    resetPanelHostsForTests();
});

test("old tab cleanup never parks the newly attached tab", async () => {
    const timeline = acquirePanelHost("timeline");
    const editor = acquirePanelHost("editor");
    mount = document.createElement("div");
    document.body.append(mount);
    root = createRoot(mount);
    for (const formId of ["timeline", "editor", "timeline", "editor"]) {
        await act(async () => root!.render(<Slot formId={formId} />));
        const current = formId === "timeline" ? timeline : editor;
        const previous = formId === "timeline" ? editor : timeline;
        expect(current.parentElement).toBe(mount.querySelector("[data-test-slot]"));
        expect(previous.parentElement?.dataset.dockParking).toBe("1");
    }
});

test("StrictMode keeps a connected panel; real unmount parks it", async () => {
    const timeline = acquirePanelHost("timeline");
    mount = document.createElement("div");
    document.body.append(mount);
    root = createRoot(mount);
    await act(async () =>
        root!.render(
            <StrictMode>
                <Slot formId="timeline" />
            </StrictMode>,
        ),
    );
    expect(timeline.parentElement).toBe(mount.querySelector("[data-test-slot]"));
    await act(async () => root!.unmount());
    root = null;
    expect(timeline.isConnected).toBe(true);
    expect(timeline.parentElement?.dataset.dockParking).toBe("1");
});
