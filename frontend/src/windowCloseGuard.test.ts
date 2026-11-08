import { describe, expect, it, vi } from "vitest";
import { createWindowCloseGuard } from "./windowCloseGuard";

function harness() {
    const state = { dirty: false, disposed: false, allowed: false };
    const getProjectMeta = vi.fn(async () => ({ dirty: false }));
    const promptSave = vi.fn();
    const closeWindow = vi.fn(async () => {});
    const guard = createWindowCloseGuard({
        isDisposed: () => state.disposed,
        consumeAllowClose: () => {
            const allowed = state.allowed;
            state.allowed = false;
            return allowed;
        },
        isProjectDirty: () => state.dirty,
        getProjectMeta,
        promptSave,
        closeWindow,
    });
    const request = async () => {
        const preventDefault = vi.fn();
        await guard({ preventDefault });
        return preventDefault;
    };
    return { state, getProjectMeta, promptSave, closeWindow, request };
}

describe("standalone window close guard", () => {
    it("reads edits made after the listener was registered", async () => {
        const h = harness();
        h.state.dirty = true;
        expect(await h.request()).toHaveBeenCalledOnce();
        expect(h.promptSave).toHaveBeenCalledOnce();
        expect(h.getProjectMeta).not.toHaveBeenCalled();
        expect(h.closeWindow).not.toHaveBeenCalled();
    });

    it("prompts when backend block edits have not reached the frontend dirty flag", async () => {
        const h = harness();
        h.getProjectMeta.mockResolvedValue({ dirty: true });
        expect(await h.request()).toHaveBeenCalledOnce();
        expect(h.promptSave).toHaveBeenCalledOnce();
        expect(h.closeWindow).not.toHaveBeenCalled();
    });

    it("only closes after both sides confirm the project is clean", async () => {
        const h = harness();
        expect(await h.request()).toHaveBeenCalledOnce();
        expect(h.closeWindow).toHaveBeenCalledOnce();
        expect(h.promptSave).not.toHaveBeenCalled();
    });

    it("keeps the window open if backend verification fails", async () => {
        const h = harness();
        h.getProjectMeta.mockRejectedValue(new Error("IPC failed"));
        expect(await h.request()).toHaveBeenCalledOnce();
        expect(h.promptSave).toHaveBeenCalledOnce();
        expect(h.closeWindow).not.toHaveBeenCalled();
    });

    it("coalesces repeated clicks and rechecks edits made during verification", async () => {
        const h = harness();
        let finish!: (meta: { dirty: boolean }) => void;
        h.getProjectMeta.mockImplementation(
            () =>
                new Promise((resolve) => {
                    finish = resolve;
                }),
        );
        const first = h.request();
        expect(await h.request()).toHaveBeenCalledOnce();
        h.state.dirty = true;
        finish({ dirty: false });
        await first;
        expect(h.getProjectMeta).toHaveBeenCalledOnce();
        expect(h.promptSave).toHaveBeenCalledOnce();
        expect(h.closeWindow).not.toHaveBeenCalled();
    });

    it("a disposed listener cannot destroy the window or open a dialog", async () => {
        const h = harness();
        let finish!: (meta: { dirty: boolean }) => void;
        h.getProjectMeta.mockImplementation(
            () =>
                new Promise((resolve) => {
                    finish = resolve;
                }),
        );
        const first = h.request();
        h.state.disposed = true;
        finish({ dirty: false });
        await first;
        expect(await h.request()).toHaveBeenCalledOnce();
        expect(h.promptSave).not.toHaveBeenCalled();
        expect(h.closeWindow).not.toHaveBeenCalled();
    });

    it("allows the confirmed close fallback exactly once", async () => {
        const h = harness();
        h.state.dirty = true;
        h.state.allowed = true;
        expect(await h.request()).not.toHaveBeenCalled();
        expect(h.state.allowed).toBe(false);
        expect(await h.request()).toHaveBeenCalledOnce();
        expect(h.promptSave).toHaveBeenCalledOnce();
    });
});
