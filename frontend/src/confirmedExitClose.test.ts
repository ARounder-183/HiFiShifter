import { runConfirmedExitClose } from "./confirmedExitClose.js";

function assertEqual<T>(actual: T, expected: T): void {
    if (actual !== expected) {
        throw new Error(`Expected ${String(expected)}, received ${String(actual)}`);
    }
}

async function main() {
    const successCalls: string[] = [];

    await runConfirmedExitClose({
        markAllowClose: () => {
            successCalls.push("allow");
        },
        destroyWindow: async () => {
            successCalls.push("destroy");
        },
        closeWindow: async () => {
            successCalls.push("close");
        },
    });

    assertEqual(successCalls.join(","), "allow,destroy");

    const fallbackCalls: string[] = [];

    await runConfirmedExitClose({
        markAllowClose: () => {
            fallbackCalls.push("allow");
        },
        destroyWindow: async () => {
            fallbackCalls.push("destroy");
            throw new Error("destroy failed");
        },
        closeWindow: async () => {
            fallbackCalls.push("close");
        },
    });

    assertEqual(fallbackCalls.join(","), "allow,destroy,allow,close");
    console.log("confirmed exit close checks passed");
}

void main();
