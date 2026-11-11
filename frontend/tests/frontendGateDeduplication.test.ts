/**
 * 前端门禁入口不得与 npm test 内的门禁型用例重复跑同一命令。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

test("run-frontend-gates.sh 不重复执行 test 已覆盖的门禁", () => {
    const gate = readFileSync(
        join(process.cwd(), "../scripts/checks/run-frontend-gates.sh"),
        "utf8",
    );
    expect(gate).not.toContain("npm --prefix frontend run lint:budget");
    expect(gate).not.toContain("npm --prefix frontend run format:check");
    expect(gate).toContain("npm --prefix frontend run test");
    expect(gate).toContain("npm --prefix frontend run build");
    expect(gate).toContain("npm --prefix frontend run check:bundle");
});
