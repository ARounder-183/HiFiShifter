export interface SyllableAnalysisClip {
    readonly id: string;
    readonly name: string;
}

export interface SyllableAnalysisResult {
    readonly clipId: string;
    readonly analyzed: boolean;
}

/** Keep one IPC analysis in flight, including across React effect reruns. */
export function createSyllableAnalysisQueue() {
    let tail: Promise<unknown> = Promise.resolve();
    return {
        runBatch<T extends SyllableAnalysisClip>(args: {
            readonly clips: readonly T[];
            readonly analyze: (clip: T) => Promise<boolean>;
            readonly shouldStart: () => boolean;
            readonly onStart: (clip: T) => void;
            readonly onSkipped: (clip: T) => void;
        }): Promise<SyllableAnalysisResult[]> {
            const run = tail.then(async () => {
                const results: SyllableAnalysisResult[] = [];
                for (const clip of args.clips) {
                    if (!args.shouldStart()) {
                        args.onSkipped(clip);
                        continue;
                    }
                    try {
                        args.onStart(clip);
                        results.push({ clipId: clip.id, analyzed: await args.analyze(clip) });
                    } catch {
                        results.push({ clipId: clip.id, analyzed: false });
                    }
                }
                return results;
            });
            tail = run.catch(() => undefined);
            return run;
        },
    };
}
