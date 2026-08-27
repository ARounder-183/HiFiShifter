interface WeightedLruEntry<V> {
    value: V;
    bytes: number;
    lastUsed: number;
    pins: number;
    evicted: boolean;
    onEvict?: (value: V) => void;
}

export interface WeightedLruLease<V> {
    value: V;
    release(): void;
}

export class WeightedLru<K, V> {
    readonly budgetBytes: number;
    private readonly entries = new Map<K, WeightedLruEntry<V>>();
    private sequence = 0;
    private residentBytes = 0;

    constructor(budgetBytes: number) {
        if (!Number.isFinite(budgetBytes) || budgetBytes < 0) {
            throw new RangeError("budgetBytes must be a finite non-negative number");
        }
        this.budgetBytes = budgetBytes;
    }

    get totalBytes(): number {
        return this.residentBytes;
    }

    get size(): number {
        return this.entries.size;
    }

    has(key: K): boolean {
        return this.entries.has(key);
    }

    set(key: K, value: V, bytes: number, onEvict?: (value: V) => void): void {
        if (!Number.isFinite(bytes) || bytes < 0) {
            throw new RangeError("entry bytes must be a finite non-negative number");
        }

        this.removeEntry(key);
        this.entries.set(key, {
            value,
            bytes,
            lastUsed: this.nextSequence(),
            pins: 0,
            evicted: false,
            onEvict,
        });
        this.residentBytes += bytes;
        this.enforceBudget(true);
    }

    peek(key: K): V | undefined {
        const entry = this.entries.get(key);
        if (!entry) return undefined;
        entry.lastUsed = this.nextSequence();
        return entry.value;
    }

    acquire(key: K): WeightedLruLease<V> | null {
        const entry = this.entries.get(key);
        if (!entry) return null;

        entry.lastUsed = this.nextSequence();
        entry.pins += 1;
        let released = false;

        return {
            value: entry.value,
            release: () => {
                if (released) return;
                released = true;
                entry.pins = Math.max(0, entry.pins - 1);
                if (!entry.evicted) this.enforceBudget(false);
            },
        };
    }

    delete(key: K): boolean {
        return this.removeEntry(key);
    }

    clear(): void {
        for (const [key] of this.entries) {
            this.removeEntry(key);
        }
    }

    private nextSequence(): number {
        this.sequence += 1;
        return this.sequence;
    }

    private enforceBudget(allowSoleOversize: boolean): void {
        while (this.residentBytes > this.budgetBytes) {
            if (allowSoleOversize && this.entries.size === 1) return;

            let candidateKey: K | undefined;
            let candidateSequence = Number.POSITIVE_INFINITY;
            for (const [key, entry] of this.entries) {
                if (entry.pins > 0 || entry.lastUsed >= candidateSequence) continue;
                candidateKey = key;
                candidateSequence = entry.lastUsed;
            }

            if (candidateKey === undefined) return;
            this.removeEntry(candidateKey);
        }
    }

    private removeEntry(key: K): boolean {
        const entry = this.entries.get(key);
        if (!entry) return false;

        this.entries.delete(key);
        this.residentBytes = Math.max(0, this.residentBytes - entry.bytes);
        if (!entry.evicted) {
            entry.evicted = true;
            entry.onEvict?.(entry.value);
        }
        return true;
    }
}
