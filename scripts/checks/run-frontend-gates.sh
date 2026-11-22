#!/usr/bin/env bash
# The single entry point for the frontend gates: lint budget / formatting / unit tests /
# production build.
#
# [Why merged into one] The same commands used to live in build.yml and vst3.yml with
# differing arguments (the latter ran only lint:budget + format:check + test, not build).
# So "can the frontend build" was only ever verified on build.yml — while the plugin
# pipelines consume the very same frontend/dist.
#
#   bash scripts/checks/run-frontend-gates.sh
#
# Dependencies are already installed (workflows run npm ci first): this script only gates,
# it does not install.
set -euo pipefail

cd "$(dirname "$0")/../.."

# ESLint 10 / Vite 8 / Vitest 5 all require Node >= 22.12 (see frontend/package.json engines).
# Mixed environments (node_modules installed on Windows but interpreted by WSL's node) fail
# with a missing util.styleText or ESM parse errors; checking the version explicitly
# separates environment problems from source regressions.
# [Why compare down to the minor] Checking only major >= 22 would let 22.0-22.11 slip
# through: those versions lack util.styleText and crash outright in rolldown, with errors
# that look like source regressions.
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

# The lint:budget / format:check / typecheck criteria are already bound inside npm test's
# buildGates.test.ts and lintBudget.test.ts (see frontend/README.md): running the tests
# cannot bypass them either. They are not invoked again here, so CI does not run the same
# commands twice (measured ~70 seconds each per round).
# Unit tests (including gate-style cases: i18n completeness, lint budget, format/type
# checks, file size budgets, and more).
npm --prefix frontend run test

# Production build: typecheck + vite build + SDK type artifacts.
npm --prefix frontend run build

# Size budget: only measurable once dist is built; no longer exists merely as an uncalled
# package.json script.
npm --prefix frontend run check:bundle
