import { Canvas2dWaveformRenderer, WebGl2WaveformRenderer } from "./surfaceRenderer";
import type { WaveformGeometry } from "./geometry";

function geometry(): WaveformGeometry {
    const values: number[] = [];
    for (let x = 16; x < 624; x += 2) {
        const amplitude = 8 + Math.abs(Math.sin(x * 0.035)) * 45;
        values.push(x, 60 - amplitude, 0.96, 0.98, 1, 0.92);
        values.push(x, 60 + amplitude, 0.96, 0.98, 1, 0.92);
    }
    return { vertices: new Float32Array(values), lineCount: values.length / 12, complete: true };
}

function countCanvasPixels(canvas: HTMLCanvasElement): number {
    const ctx = canvas.getContext("2d");
    if (!ctx) return 0;
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let count = 0;
    for (let index = 3; index < pixels.length; index += 4) {
        if ((pixels[index] ?? 0) > 0) count += 1;
    }
    return count;
}

function countWebGlPixels(canvas: HTMLCanvasElement): number {
    const gl = canvas.getContext("webgl2");
    if (!gl) return 0;
    const pixels = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let count = 0;
    for (let index = 3; index < pixels.length; index += 4) {
        if ((pixels[index] ?? 0) > 0) count += 1;
    }
    return count;
}

const webglCanvas = document.querySelector<HTMLCanvasElement>("#webgl");
const fallbackCanvas = document.querySelector<HTMLCanvasElement>("#canvas2d");
const result = document.querySelector<HTMLElement>("#result");

if (!webglCanvas || !fallbackCanvas || !result) throw new Error("Test fixture is incomplete");

try {
    const data = geometry();
    const webgl = new WebGl2WaveformRenderer(webglCanvas);
    const fallback = new Canvas2dWaveformRenderer(fallbackCanvas);
    webgl.render(data, 640, 120, window.devicePixelRatio || 1);
    fallback.render(data, 640, 120, window.devicePixelRatio || 1);

    const webglPixels = countWebGlPixels(webglCanvas);
    const fallbackPixels = countCanvasPixels(fallbackCanvas);
    const pass = webglPixels > 1000 && fallbackPixels > 1000;
    result.dataset.status = pass ? "pass" : "fail";
    result.dataset.webglPixels = String(webglPixels);
    result.dataset.canvasPixels = String(fallbackPixels);
    result.textContent = pass
        ? `PASS - WebGL2 ${webglPixels} pixels, Canvas 2D ${fallbackPixels} pixels`
        : `FAIL - WebGL2 ${webglPixels} pixels, Canvas 2D ${fallbackPixels} pixels`;
} catch (error) {
    result.dataset.status = "fail";
    result.textContent = `FAIL - ${error instanceof Error ? error.message : String(error)}`;
}
