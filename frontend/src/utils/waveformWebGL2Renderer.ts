/**
 * 波形 WebGL2 渲染器（占位文件，Task 4-9 填充实现）
 *
 * 完整实现见后续任务：shader 编译、纹理管理、drawClipWaveform 主流程。
 */

import type { WaveformRenderer } from "./waveformRenderer";

export class WebGL2WaveformRenderer implements WaveformRenderer {
    readonly backend = "webgl2" as const;

    constructor(canvas: HTMLCanvasElement, gl: WebGL2RenderingContext) {
        // 占位：实际实现在 Task 4-9
        void canvas;
        void gl;
        throw new Error("WebGL2WaveformRenderer not implemented yet");
    }

    resize(): void {
        throw new Error("Not implemented");
    }

    clear(): void {
        throw new Error("Not implemented");
    }

    drawClipWaveform(): void {
        throw new Error("Not implemented");
    }

    dispose(): void {
        // 占位
    }
}
