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

# 预算门禁：警告数是只许降不许升的棘轮（现值见 frontend/tests/lintBudget.test.ts 的
# WARNING_BUDGET 与 package.json 的 lint:budget，两者由该测试钉住一致）。
npm --prefix frontend run lint:budget

npm --prefix frontend run format:check

# 单元测试（含 i18n 完整性、lint 预算、文件体积预算等门禁型用例）。
npm --prefix frontend run test

# 生产构建：typecheck + vite build + SDK 类型产物。
npm --prefix frontend run build
