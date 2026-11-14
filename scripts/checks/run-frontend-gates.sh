#!/usr/bin/env bash
# 前端门禁的唯一入口：lint 预算 / 格式 / 单测 / 生产构建。
#
# 【为什么合并成一条】同一组命令此前写在 build.yml 与 vst3.yml 两处，参数还不一样
# （后者只跑 lint:budget + format:check + test，不跑 build）。于是"前端能不能构建"
# 只有在 build.yml 上被验证过，而插件流水线用的正是同一份 frontend/dist。
#
#   bash scripts/checks/run-frontend-gates.sh
#
# 依赖已安装（工作流里先跑 npm ci）：本脚本只做门禁，不做安装。
set -euo pipefail

cd "$(dirname "$0")/../.."

# ESLint 10 / Vite 8 / Vitest 5 都要求 Node >= 22.12（见 frontend/package.json engines）。
# 混合环境（Windows 安装的 node_modules 被 WSL 的 node 解释）会报 util.styleText 缺失或
# ESM 解析错误；显式校验版本，把环境问题与源码回归区分开。
# 【为什么比较到 minor】只查 major>=22 会放过 22.0–22.11：这几个版本没有
# util.styleText，会在 rolldown 里直接崩，报错看起来像源码回归。
if ! command -v node >/dev/null 2>&1; then
    echo "run-frontend-gates: node not found on PATH; install Node >= 22.12" >&2
    exit 1
fi
NODE_MAJOR=$(node -p "Number(process.versions.node.split('.')[0])")
NODE_MINOR=$(node -p "Number(process.versions.node.split('.')[1])")
if [ "$NODE_MAJOR" -lt 22 ] || { [ "$NODE_MAJOR" -eq 22 ] && [ "$NODE_MINOR" -lt 12 ]; }; then
    echo "run-frontend-gates: Node >= 22.12 is required (found $(node --version))" >&2
    exit 1
fi

# lint:budget / format:check / typecheck 的判据已经绑在 npm test 内的
# buildGates.test.ts 与 lintBudget.test.ts（见 frontend/README.md）：直接跑测试时
# 也绕不开它们。这里不再重复调用，避免 CI 把同一组命令各跑两遍（实测每轮各约 70 秒）。
# 单元测试（含 i18n 完整性、lint 预算、格式/类型检查、文件体积预算等门禁型用例）。
npm --prefix frontend run test

# 生产构建：typecheck + vite build + SDK 类型产物。
npm --prefix frontend run build

# 体积预算：构建完 dist 才能测量；不再只作为没人调用的 package.json 脚本存在。
npm --prefix frontend run check:bundle
