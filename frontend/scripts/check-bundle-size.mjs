/**
 * 前端构建体积预算。
 *
 * 【为什么需要】Vite 的 chunkSizeWarningLimit 只是单 chunk 告警，不覆盖
 * CSS，也不阻止主入口悄悄变大。这里把“首屏入口”和“CSS 总量”钉成硬上限：
 * - mount 是 App 的入口 chunk；超过 800 kB 说明又有重模块被静态拖回首屏；
 * - CSS 超过 800 kB 说明 Radix/Tailwind 或新增样式表失控。
 *
 * 预算不是审美：桌面端虽然本地加载，但解析/编译成本仍直接落在启动时间上。
 */
import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const assets = join(process.cwd(), "dist", "assets");
const files = readdirSync(assets);

const mount = files.find((file) => /^mount-.*\.js$/.test(file));
if (!mount) throw new Error("mount chunk not found in " + assets);
const mountBytes = statSync(join(assets, mount)).size;
const mountLimit = 800 * 1024;
if (mountBytes > mountLimit) {
    throw new Error("mount chunk " + mount + " is " + mountBytes + " bytes; limit " + mountLimit);
}

const cssFiles = files.filter((file) => file.endsWith(".css"));
const cssBytes = cssFiles.reduce((sum, file) => sum + statSync(join(assets, file)).size, 0);
const cssLimit = 800 * 1024;
if (cssBytes > cssLimit) {
    throw new Error("CSS budget exceeded: " + cssBytes + " bytes; limit " + cssLimit);
}

/*
 * 总量预算会放过「一个 chunk 吃掉几乎全部额度」的退化：当前最大的 CSS 已经
 * 746.73 kB、vendor chunk 633.95 kB。这里给单文件也钉一道只降不升的上限，
 * 失败信息直接点名文件，避免下一次增长在总量用尽之前无人察觉。
 */
const largestCss = cssFiles
    .map((file) => ({ file, bytes: statSync(join(assets, file)).size }))
    .sort((a, b) => b.bytes - a.bytes)[0];
const cssFileLimit = 780 * 1024;
if (largestCss && largestCss.bytes > cssFileLimit) {
    throw new Error(
        "largest CSS file " +
            largestCss.file +
            " is " +
            largestCss.bytes +
            " bytes; limit " +
            cssFileLimit,
    );
}

const jsFiles = files.filter((file) => file.endsWith(".js"));
const largestJs = jsFiles
    .map((file) => ({ file, bytes: statSync(join(assets, file)).size }))
    .sort((a, b) => b.bytes - a.bytes)[0];
const jsFileLimit = 700 * 1024;
if (largestJs && largestJs.bytes > jsFileLimit) {
    throw new Error(
        "largest JS chunk " +
            largestJs.file +
            " is " +
            largestJs.bytes +
            " bytes; limit " +
            jsFileLimit,
    );
}

console.log(
    "bundle size ok: " +
        mount +
        "=" +
        mountBytes +
        "B, css=" +
        cssBytes +
        "B, largestCss=" +
        (largestCss?.file ?? "none") +
        "=" +
        (largestCss?.bytes ?? 0) +
        "B, largestJs=" +
        (largestJs?.file ?? "none") +
        "=" +
        (largestJs?.bytes ?? 0) +
        "B",
);
