const fs = require("fs");
const path = require("path");
const files = [
    "frontend/src/i18n/en-US.ts",
    "frontend/src/i18n/ko-KR.ts",
    "frontend/src/i18n/ja-JP.ts",
    "frontend/src/i18n/zh-CN.ts",
    "frontend/src/i18n/zh-TW.ts",
];
function keysFrom(content) {
    const re = /^[ \t]*([a-zA-Z0-9_]+):/gm;
    const out = new Set();
    let m;
    while ((m = re.exec(content))) {
        out.add(m[1]);
    }
    return Array.from(out).sort();
}
const map = {};
for (const f of files) {
    try {
        const c = fs.readFileSync(path.join(process.cwd(), f), "utf8");
        map[f] = keysFrom(c);
    } catch (e) {
        map[f] = null;
    }
}
const en = map["frontend/src/i18n/en-US.ts"];
for (const f of files) {
    if (!map[f] || f === "frontend/src/i18n/en-US.ts") continue;
    const missing = en.filter((k) => !map[f].includes(k));
    console.log(f + " missing " + missing.length + " keys");
    missing.forEach((k) => console.log("  " + k));
}
