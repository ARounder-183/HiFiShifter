/**
 * 钢琴键点击音效播放器
 */

const MAX_VOICES = 5;

export class PianoKeySound {
    private audioContext: AudioContext | null = null;
    private compressor: DynamicsCompressorNode | null = null;
    private masterGain: GainNode | null = null;
    private activeOscillators: Map<
        number,
        { osc: OscillatorNode; filter: BiquadFilterNode; gain: GainNode }
    > = new Map();

    private midiToFreq(midiNote: number): number {
        return 440 * Math.pow(2, (midiNote - 69) / 12);
    }

    private ensureContext(): AudioContext {
        if (!this.audioContext) {
            const ctx = new AudioContext();
            this.audioContext = ctx;
            this.compressor = ctx.createDynamicsCompressor();
            this.compressor.threshold.value = -8;
            this.compressor.knee.value = 20;
            this.compressor.ratio.value = 6;
            this.compressor.attack.value = 0.005; 
            this.compressor.release.value = 0.1;
            this.masterGain = ctx.createGain();
            this.masterGain.gain.value = 1.3; 

            this.compressor.connect(this.masterGain);
            this.masterGain.connect(ctx.destination);
        }
        if (this.audioContext.state === "suspended") {
            void this.audioContext.resume();
        }
        return this.audioContext;
    }

    play(midiNote: number, velocity: number = 0.5): void {
        const ctx = this.ensureContext();

        if (this.activeOscillators.has(midiNote)) return;

        if (this.activeOscillators.size >= MAX_VOICES) {
            const oldestKey = this.activeOscillators.keys().next().value;
            if (oldestKey !== undefined) this.stop(oldestKey);
        }

        const freq = this.midiToFreq(midiNote);
        const osc = ctx.createOscillator();
        const filter = ctx.createBiquadFilter();
        const gain = ctx.createGain();

        osc.type = "triangle";
        osc.frequency.value = freq;

        filter.type = "lowpass";
        filter.frequency.value = Math.min(freq * 4.5, 15000); 
        filter.Q.value = 1.5;

        const now = ctx.currentTime;
        const startTime = now + 0.005;

        gain.gain.value = 0;
        gain.gain.setValueAtTime(0, startTime);
        gain.gain.linearRampToValueAtTime(velocity, startTime + 0.008);
        const sustainLevel = velocity * 0.75;
        gain.gain.setTargetAtTime(sustainLevel, startTime + 0.008, 0.05);

        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this.compressor!); 

        osc.start(startTime);

        this.activeOscillators.set(midiNote, { osc, filter, gain });
    }

    stop(midiNote: number): void {
        const entry = this.activeOscillators.get(midiNote);
        if (!entry) return;

        const { osc, filter, gain } = entry;
        const ctx = this.audioContext!;
        const now = ctx.currentTime;

        if (typeof gain.gain.cancelAndHoldAtTime === 'function') {
            gain.gain.cancelAndHoldAtTime(now);
        } else {
            gain.gain.cancelScheduledValues(now);
            gain.gain.setValueAtTime(gain.gain.value, now);
        }

        gain.gain.setTargetAtTime(0, now, 0.015);

        const stopTime = now + 0.1;
        osc.stop(stopTime);

        osc.onended = () => {
            gain.disconnect();
            filter.disconnect();
            osc.disconnect();
        };

        this.activeOscillators.delete(midiNote);
    }

    stopAll(): void {
        for (const midiNote of this.activeOscillators.keys()) {
            this.stop(midiNote);
        }
    }

    dispose(): void {
        this.stopAll();
        if (this.audioContext) {
            void this.audioContext.close();
            this.audioContext = null;
        }
    }
}

export const pianoKeySound = new PianoKeySound();