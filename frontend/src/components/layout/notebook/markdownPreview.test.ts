import { test } from "vitest";

import { renderMarkdownPreview } from "./markdownPreview.ts";

test("components/layout/notebook/markdownPreview.test.ts scripted checks", async () => {
    function assertIncludes(actual: string, expected: string, label: string): void {
        if (!actual.includes(expected)) {
            throw new Error(`${label}: expected to include ${expected}, received ${actual}`);
        }
    }

    const html = renderMarkdownPreview(`# Title

Paragraph with [link](https://example.com) and \`code\`.

- a
- b

> quote

\`\`\`ts
const x = 1;
\`\`\`
`);

    assertIncludes(html, "<h1>Title</h1>", "heading renders");
    assertIncludes(
        html,
        '<a href="https://example.com" target="_blank" rel="noreferrer">link</a>',
        "link renders",
    );
    assertIncludes(html, "<code>code</code>", "inline code renders");
    assertIncludes(html, "<ul>", "list renders");
    assertIncludes(html, "<blockquote>", "blockquote renders");
    assertIncludes(html, "<pre><code", "code fence renders");
    assertIncludes(
        renderMarkdownPreview("<script>alert(1)</script>"),
        "&lt;script&gt;alert(1)&lt;/script&gt;",
        "raw html is escaped",
    );

});
