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

const cssBytes = files
    .filter((file) => file.endsWith(".css"))
    .reduce((sum, file) => sum + statSync(join(assets, file)).size, 0);
const cssLimit = 800 * 1024;
if (cssBytes > cssLimit) {
    throw new Error("CSS budget exceeded: " + cssBytes + " bytes; limit " + cssLimit);
}

console.log("bundle size ok: " + mount + "=" + mountBytes + "B, css=" + cssBytes + "B");
