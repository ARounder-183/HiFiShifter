import { describe, expect, it } from "vitest";
import {
    applyExtensionToFileName,
    isSampleRateAllowed,
    nearestAllowedSampleRate,
    sampleRateOptions,
} from "./exportFormat";

describe("applyExtensionToFileName", () => {
    it("replaces known extensions case-insensitively", () => {
        expect(applyExtensionToFileName("<ProjectName>.wav", "mp3")).toBe("<ProjectName>.mp3");
        expect(applyExtensionToFileName("song.MP3", "flac")).toBe("song.flac");
        expect(applyExtensionToFileName("a.b.Flac", "wav")).toBe("a.b.wav");
    });

    it("appends when extension is missing or unknown", () => {
        expect(applyExtensionToFileName("track", "flac")).toBe("track.flac");
        expect(applyExtensionToFileName("track.bak", "mp3")).toBe("track.bak.mp3");
    });

    it("keeps directory components intact", () => {
        expect(applyExtensionToFileName("album.wav/track", "mp3")).toBe("album.wav/track.mp3");
        expect(applyExtensionToFileName("album/track.wav", "mp3")).toBe("album/track.mp3");
    });
});

describe("sample rate helpers", () => {
    it("restricts mp3 to the MPEG table", () => {
        expect(sampleRateOptions("mp3")).not.toContain(96000);
        expect(sampleRateOptions("wav")).toContain(96000);
        expect(isSampleRateAllowed("mp3", 48000)).toBe(true);
        expect(isSampleRateAllowed("mp3", 96000)).toBe(false);
        expect(isSampleRateAllowed("flac", 96000)).toBe(true);
    });

    it("halves onto the nearest MPEG rate", () => {
        expect(nearestAllowedSampleRate("mp3", 96000)).toBe(48000);
        expect(nearestAllowedSampleRate("mp3", 88200)).toBe(44100);
        expect(nearestAllowedSampleRate("mp3", 192000)).toBe(48000);
        expect(nearestAllowedSampleRate("mp3", 176400)).toBe(44100);
    });

    it("falls back to the log-nearest rate and returns null when already legal", () => {
        expect(nearestAllowedSampleRate("mp3", 50000)).toBe(48000);
        expect(nearestAllowedSampleRate("wav", 48000)).toBeNull();
        expect(nearestAllowedSampleRate("mp3", 48000)).toBeNull();
        expect(nearestAllowedSampleRate("mp3", Number.NaN)).toBeNull();
    });

    it("reduces to the smallest table entry for below-table rates", () => {
        expect(nearestAllowedSampleRate("mp3", 4000)).toBe(8000);
        expect(nearestAllowedSampleRate("mp3", 0)).toBeNull();
        expect(nearestAllowedSampleRate("mp3", -44100)).toBeNull();
        expect(nearestAllowedSampleRate("mp3", Number.POSITIVE_INFINITY)).toBeNull();
    });
});
