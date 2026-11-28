// Talk to the app's WebView through the Chrome DevTools Protocol.
//
// [Why this exists] On a phone the only window into a Tauri app is Chrome DevTools,
// but "open chrome://inspect and look" is not something a script (or an agent) can do.
// The protocol underneath that UI is plain JSON over a WebSocket, and Android exposes
// it per WebView through a local abstract socket -- so this script attaches to the same
// endpoint Chrome would, and can read the console, evaluate JavaScript and take a
// screenshot without a human in the loop.
//
// It is the only way to assert things that are invisible from outside: which shell the
// app picked, what the native inset bridge reported, whether an invoke round-trip works.
//
//   node scripts/dev/android-webview.mjs --eval "document.documentElement.dataset.form"
//   node scripts/dev/android-webview.mjs --console
//   node scripts/dev/android-webview.mjs --screenshot out.png
//   node scripts/dev/android-webview.mjs --check
//
// The device must be reachable and the port forwarded; android-emulator.ps1 -Action
// devtools does that, or by hand:
//   adb forward tcp:9222 localabstract:webview_devtools_remote_<pid>

import { writeFileSync } from "node:fs";

const HOST = "http://127.0.0.1:9222";
const args = process.argv.slice(2);

const flag = (name) => args.includes(name);
const value = (name, fallback = "") => {
    const index = args.indexOf(name);
    return index >= 0 && index + 1 < args.length ? args[index + 1] : fallback;
};

async function listTargets() {
    const response = await fetch(HOST + "/json/list");
    if (!response.ok) throw new Error("devtools endpoint returned " + response.status);
    return response.json();
}

function connect(url) {
    return new Promise((resolve, reject) => {
        const socket = new WebSocket(url);
        socket.addEventListener("open", () => resolve(socket));
        socket.addEventListener("error", () => reject(new Error("cannot open " + url)));
        setTimeout(() => reject(new Error("timed out opening " + url)), 10000);
    });
}

let nextId = 1;
const pending = new Map();
const events = [];

function send(socket, method, params = {}) {
    const id = nextId++;
    socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        setTimeout(() => {
            if (pending.delete(id)) reject(new Error("timed out waiting for " + method));
        }, 30000);
    });
}

const targets = await listTargets();
// The app can expose more than one page (a second window, an iframe target); the main
// document is the one whose URL is the app's origin.
const page = targets.find((target) => target.type === "page") || targets[0];
if (!page) {
    console.error("no page target; is the app running and the port forwarded?");
    process.exit(2);
}

const socket = await connect(page.webSocketDebuggerUrl);
socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
        const { resolve, reject } = pending.get(message.id);
        pending.delete(message.id);
        if (message.error) reject(new Error(JSON.stringify(message.error)));
        else resolve(message.result);
    } else if (message.method) {
        events.push(message);
    }
});

await send(socket, "Runtime.enable");
await send(socket, "Log.enable");

async function evaluate(expression) {
    const result = await send(socket, "Runtime.evaluate", {
        expression,
        returnByValue: true,
        awaitPromise: true,
        userGesture: true,
    });
    if (result.exceptionDetails) {
        const detail = result.exceptionDetails;
        return { __error: detail.exception?.description || detail.text || "evaluation failed" };
    }
    return result.result.value;
}

function describeConsole() {
    const lines = [];
    for (const event of events) {
        if (event.method === "Runtime.consoleAPICalled") {
            const text = (event.params.args || [])
                .map((argument) => argument.value ?? argument.description ?? argument.type)
                .join(" ");
            lines.push({ level: event.params.type, text });
        } else if (event.method === "Runtime.exceptionThrown") {
            const detail = event.params.exceptionDetails;
            lines.push({
                level: "exception",
                text: detail.exception?.description || detail.text || "",
            });
        } else if (event.method === "Log.entryAdded") {
            lines.push({ level: event.params.entry.level, text: String(event.params.entry.text) });
        }
    }
    return lines;
}

if (flag("--eval")) {
    const expression = value("--eval");
    const result = await evaluate(expression);
    console.log(typeof result === "string" ? result : JSON.stringify(result, null, 2));
}

if (flag("--screenshot")) {
    const result = await send(socket, "Page.captureScreenshot", { format: "png" });
    const file = value("--screenshot", "webview.png");
    writeFileSync(file, Buffer.from(result.data, "base64"));
    console.log("screenshot: " + file);
}

if (flag("--console")) {
    // Deduplicated: a broken attribute in a render loop produces thousands of identical
    // lines, and "which distinct errors happened" is the question that matters.
    const seen = new Map();
    for (const line of describeConsole()) {
        const key = line.level + "|" + line.text;
        seen.set(key, (seen.get(key) || 0) + 1);
    }
    for (const [key, count] of seen) {
        const split = key.indexOf("|");
        const level = key.slice(0, split);
        const text = key.slice(split + 1);
        console.log("[" + level + "] x" + count + " " + text.slice(0, 500));
    }
}

if (flag("--check")) {
    // The assertions a phone cannot be asked for by hand. Each one has already been the
    // cause of a shipped defect: the shell picking the desktop layout, the inset bridge
    // reporting zeros, an invoke that never reaches Rust.
    const probe = await evaluate(`JSON.stringify((() => {
        const style = getComputedStyle(document.documentElement);
        return {
            form: document.documentElement.dataset.form || null,
            insetTop: style.getPropertyValue("--qt-inset-top").trim(),
            insetBottom: style.getPropertyValue("--qt-inset-bottom").trim(),
            safeTop: style.getPropertyValue("--qt-safe-top").trim(),
            hasTauri: typeof window.__TAURI_INTERNALS__ !== "undefined",
            title: document.title,
        };
    })())`);
    const state = JSON.parse(probe);
    const profile = await evaluate(
        "window.__TAURI_INTERNALS__.invoke('get_surface_profile').then((p) => JSON.stringify(p.viewport)).catch((e) => 'ERR ' + e)",
    );
    const insets = await evaluate(
        "window.__TAURI_INTERNALS__.invoke('get_surface_insets').then((i) => JSON.stringify(i)).catch((e) => 'ERR ' + e)",
    );
    const exceptions = describeConsole().filter(
        (line) => line.level === "exception" || line.level === "error",
    );

    const report = {
        ...state,
        viewportFromRust: profile,
        insetsFromRust: insets,
        errorCount: exceptions.length,
        distinctErrors: [...new Set(exceptions.map((line) => line.text.slice(0, 160)))].slice(
            0,
            10,
        ),
    };
    console.log(JSON.stringify(report, null, 2));

    const problems = [];
    if (!report.hasTauri)
        problems.push(
            "window.__TAURI_INTERNALS__ is missing: the Rust side never reached the page",
        );
    if (report.form !== "mobile")
        problems.push("shell form is '" + report.form + "', expected 'mobile'");
    if (!report.insetTop || report.insetTop === "0px")
        problems.push("--qt-inset-top is zero: the native inset bridge reported nothing");
    if (String(profile).startsWith("ERR")) problems.push("get_surface_profile failed: " + profile);
    if (String(insets).startsWith("ERR")) problems.push("get_surface_insets failed: " + insets);
    if (problems.length) {
        console.error("CHECK FAILED");
        for (const problem of problems) console.error("  - " + problem);
        process.exit(1);
    }
    console.log("CHECK OK");
}

socket.close();
