import { describe, expect, it } from "vitest";
import { backendDetail } from "./invoke";

describe("backendDetail", () => {
    it("returns a string cause verbatim", () => {
        expect(backendDetail(new Error("outer", { cause: "backend said no" }))).toBe(
            "backend said no",
        );
    });

    it("returns the innermost message along an object cause chain", () => {
        const error = new Error("outer", {
            cause: new Error("middle", { cause: new Error("backend detail") }),
        });
        expect(backendDetail(error)).toBe("backend detail");
    });

    it("does not loop forever on cyclic causes", () => {
        const error = new Error("outer");
        (error as Error & { cause?: unknown }).cause = error;
        expect(backendDetail(error)).toBe("outer");
    });
});
