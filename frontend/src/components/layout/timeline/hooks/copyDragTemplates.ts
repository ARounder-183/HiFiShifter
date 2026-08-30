import type { ClipTemplate } from "../../../../features/session/sessionTypes";

export async function buildCopyDragTemplates(args: {
    templateInputs: Array<{
        id: string;
        initial: { startSec: number; trackId: string };
        now: {
            name: string;
            lengthSec: number;
            sourcePath?: string;
            durationSec?: number;
            gain?: number;
            muted?: boolean;
            sourceStartSec?: number;
            sourceEndSec?: number;
            playbackRate?: number;
            reversed?: boolean;
            loopEnabled?: boolean;
            fadeInSec?: number;
            fadeOutSec?: number;
            fadeInShape?: number;
            fadeInDir?: number;
            fadeOutShape?: number;
            fadeOutDir?: number;
        };
        targetTrackId: string;
    }>;
    deltaSec: number;
    linkedParamsResults: Array<{ ok?: boolean; linkedParams?: unknown }>;
}): Promise<ClipTemplate[]> {
    return args.templateInputs.map((input, index) => ({
        trackId: input.targetTrackId,
        name: String(input.now.name),
        startSec: Math.max(0, input.initial.startSec + args.deltaSec),
        lengthSec: Number(input.now.lengthSec),
        sourcePath: input.now.sourcePath,
        durationSec: input.now.durationSec,
        gain: Number(input.now.gain ?? 1) || 1,
        muted: Boolean(input.now.muted),
        sourceStartSec: Number(input.now.sourceStartSec ?? 0) || 0,
        sourceEndSec: Number(input.now.sourceEndSec ?? 0) || 0,
        playbackRate: Number(input.now.playbackRate ?? 1) || 1,
        reversed: Boolean(input.now.reversed),
        loopEnabled: Boolean(input.now.loopEnabled),
        fadeInSec: Number(input.now.fadeInSec ?? 0) || 0,
        fadeOutSec: Number(input.now.fadeOutSec ?? 0) || 0,
        fadeInShape: Number(input.now.fadeInShape ?? 0) || 0,
        fadeInDir: Number(input.now.fadeInDir ?? 0) || 0,
        fadeOutShape: Number(input.now.fadeOutShape ?? 0) || 0,
        fadeOutDir: Number(input.now.fadeOutDir ?? 0) || 0,
        linkedParams: args.linkedParamsResults[index]?.ok
            ? (args.linkedParamsResults[index]?.linkedParams as ClipTemplate["linkedParams"])
            : undefined,
    }));
}
