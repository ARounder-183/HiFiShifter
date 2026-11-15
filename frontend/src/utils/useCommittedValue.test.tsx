// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it } from "vitest";
import { useCommittedValue, type CommittedValue } from "./useCommittedValue";

describe("useCommittedValue", () => {
    it("publishes the latest value after commit", async () => {
        const host = document.createElement("div");
        const root = createRoot(host);
        let box: CommittedValue<number> | undefined;
        function Probe({ value }: { value: number }) {
            box = useCommittedValue(value);
            return null;
        }
        await act(async () => {
            root.render(<Probe value={1} />);
        });
        expect(box?.current.value).toBe(1);
        await act(async () => {
            root.render(<Probe value={2} />);
        });
        expect(box?.current.value).toBe(2);
        await act(async () => {
            root.unmount();
        });
    });
});
