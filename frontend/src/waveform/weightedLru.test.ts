import { WeightedLru } from "./weightedLru.ts";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    if (!Object.is(actual, expected)) {
        throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
    }
}

{
    const cache = new WeightedLru<string, string>(10);
    cache.set("a", "A", 6);
    cache.set("b", "B", 6);

    assertEqual(cache.has("a"), false, "oldest unpinned entry is evicted by bytes");
    assertEqual(cache.has("b"), true, "newest entry remains resident");
    assertEqual(cache.totalBytes, 6, "resident bytes reflect backing costs");
}

{
    const cache = new WeightedLru<string, string>(10);
    cache.set("pinned", "P", 8);
    const lease = cache.acquire("pinned");
    cache.set("new", "N", 8);

    assertEqual(cache.has("pinned"), true, "pinned entry survives eviction");
    assertEqual(cache.has("new"), false, "unpinned insertion yields to a pinned entry");
    lease?.release();
    assertEqual(cache.totalBytes <= 10, true, "lease release enforces the byte budget");
}

{
    const cache = new WeightedLru<string, string>(10);
    const evicted: string[] = [];
    cache.set("oversize", "O", 30, (value) => evicted.push(value));
    const lease = cache.acquire("oversize");

    assertEqual(lease?.value, "O", "sole oversize entry remains available for immediate use");
    assertEqual(cache.totalBytes, 30, "sole pinned oversize entry may exceed the budget");

    lease?.release();
    lease?.release();
    assertEqual(cache.totalBytes, 0, "released oversize entry is removed");
    assertEqual(evicted.length, 1, "idempotent lease release evicts exactly once");
}

{
    const cache = new WeightedLru<string, string>(12);
    cache.set("a", "A", 6);
    cache.set("b", "B", 6);
    assertEqual(cache.peek("a"), "A", "peek returns and refreshes an entry");
    cache.set("c", "C", 6);

    assertEqual(cache.has("a"), true, "recently read entry remains resident");
    assertEqual(cache.has("b"), false, "least recently read entry is evicted");
}

console.log("weightedLru checks passed");
