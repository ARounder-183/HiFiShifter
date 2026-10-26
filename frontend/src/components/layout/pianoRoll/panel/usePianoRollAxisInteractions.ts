/**
 * 参数编辑器**左轴列**（音高参数是琴键列，其余参数是数值刻度）的三个交互。
 *
 * 三条通路共用同一个宿主元素（`axisWrapRef`）和同一套"y → 参数值"投影
 * （分母一律是**绘图区**高度 `viewSize.h`，不是轴列高度）：
 *
 * 1. **试听**：音高参数下按琴键发声（按下 / 拖动换键 / 抬起停止；窗口失焦也停止，
 *    见该 effect 里的说明）；非音高参数直接不注册监听——对着数值刻度发声是纯噪声。
 * 2. **单位切换**：支持线性倍率 ↔ dB 的参数，点击左轴切换展示单位并持久化。
 *    不支持时同样不注册监听，点击保持"无操作"。
 * 3. **悬浮读数**：指针在轴上时给出该 y 处的读数（音高只给音名，数值刻度给参数
 *    值），按帧合并 setState。
 *
 * 【为什么从面板模型里搬出来】这三段原本夹在滚动 / 同步 / 渲染内核的 effect 之间，
 * 但它们只读 `editParam` / `axisWrapRef` / `viewSizeRef` / `pitchViewRef`，
 * 不写任何面板状态（只通过传入的 setter 写一个浮窗），因此是**纯输入接线**，
 * 可以整体搬走而不牵动任何其它功能。
 */
import { useEffect } from "react";
import type { AppDispatch, RootState } from "../../../../app/store";
import { persistUiSettings, toggleParamAxisUnit } from "../../../../features/session/sessionSlice";
import { pianoKeySound } from "../../../../utils/PianoKeySound";
import { clamp } from "../../timeline";
import { PITCH_MAX_MIDI, PITCH_MIN_MIDI } from "../constants";
import { supportsParamAxisUnit } from "../paramAxisUnits";
import { midiToLabel } from "../utils";
import type { ParamName, ValueViewport } from "../types";

/** 本 hook 读取的会话切片（只声明真正用到的字段）。 */
type AxisSession = Pick<RootState["session"], "showParamValuePopup">;

type PianoRollAxisInputs = {
    /** 会话切片：只用到 `showParamValuePopup`（悬浮读数的总开关）。 */
    s: AxisSession;
    /** 当前编辑的参数（决定轴是琴键列还是数值刻度）。 */
    editParam: ParamName;
    /** 左轴列的宿主元素（滚轮 / 点击 / 悬浮都挂在它上面）。 */
    axisWrapRef: { current: HTMLDivElement | null };
    /** 绘图区尺寸（y 投影的分母，**不是**轴列高度）。 */
    viewSizeRef: { current: { w: number; h: number } };
    /** 音高视口（琴键 → MIDI 的投影来源）。 */
    pitchViewRef: { current: ValueViewport };
    dispatch: AppDispatch;
    /** y（绘图区坐标）→ 当前参数的参数值。 */
    yToValue: (param: ParamName, y: number, h: number) => number;
    /** 参数值 → 悬浮读数字符串（与曲线浮窗同一套口径）。 */
    formatParamValuePreview: (value: number) => string;
    /** 左轴读数浮窗的 setter（浮窗本身由面板渲染）。 */
    setAxisValuePreview: (
        value: {
            clientX: number;
            clientY: number;
            text: string;
            /** 轴列容器 rect 快照（flushPreview 内随读数同帧采集）。 */
            rectLeft: number;
            rectTop: number;
        } | null,
    ) => void;
};

/** 左轴列的三个交互。副作用顺序（试听 → 单位切换 → 读数）与拆分前一致。 */
export function usePianoRollAxisInteractions({
    s,
    editParam,
    axisWrapRef,
    viewSizeRef,
    pitchViewRef,
    dispatch,
    yToValue,
    formatParamValuePreview,
    setAxisValuePreview,
}: PianoRollAxisInputs) {
    // Piano keys (axis) hover: play sine wave sound when pointer moves over keys
    useEffect(() => {
        // 【只有音高参数的左轴才是钢琴卷帘】其余参数的左轴是数值刻度（音量 / 动态 /
        // 音分…），没有"对应的音高"可发声——对着刻度按下就响是纯粹的噪声，且音量 /
        // 动态的刻度点击另有语义（切换展示单位，见下一个 effect）。非音高时直接在
        // 注册监听之前返回，连手势状态都不建立。
        if (editParam !== "pitch") return;

        const el = axisWrapRef.current;
        if (!el) return;

        let isPointerDown = false;
        let activeMidiNote: number | null = null;

        const getMidiNoteFromY = (clientY: number): number => {
            const bounds = el.getBoundingClientRect();
            const y = clientY - bounds.top;
            // 同滚轮路径：分母取绘图区高度，与琴键实例的投影同源
            //（见内核的 AXIS_TICK_LABEL_DESCENT_PX 说明）。
            const h = Math.max(1, viewSizeRef.current.h);
            const t = 1 - clamp(y / h, 0, 1);
            const absMin = PITCH_MIN_MIDI;
            const absMax = PITCH_MAX_MIDI;
            const view = pitchViewRef.current;
            const span = clamp(view.span, 1e-6, absMax - absMin);
            const min = clamp(view.center - span / 2, absMin, absMax - span);
            // 使用 floor 与渲染逻辑一致
            return Math.floor(clamp(min + t * span, absMin, absMax));
        };

        const playNoteIfChanged = (midiNote: number) => {
            if (midiNote !== activeMidiNote) {
                if (activeMidiNote !== null) {
                    pianoKeySound.stop(activeMidiNote);
                }
                activeMidiNote = midiNote;
                pianoKeySound.play(midiNote, 0.25);
            }
        };

        const stopNote = () => {
            if (activeMidiNote !== null) {
                pianoKeySound.stop(activeMidiNote);
                activeMidiNote = null;
            }
        };

        const onPointerDown = (e: PointerEvent) => {
            if (e.button !== 0) return;
            isPointerDown = true;
            const midiNote = getMidiNoteFromY(e.clientY);
            playNoteIfChanged(midiNote);
        };

        const onPointerMove = (e: PointerEvent) => {
            if (!isPointerDown) return;
            const midiNote = getMidiNoteFromY(e.clientY);
            playNoteIfChanged(midiNote);
        };

        const onPointerUp = () => {
            isPointerDown = false;
            stopNote();
        };

        const onPointerLeave = () => {
            if (isPointerDown) {
                stopNote();
            }
        };

        el.addEventListener("pointerdown", onPointerDown);
        el.addEventListener("pointermove", onPointerMove);
        window.addEventListener("pointerup", onPointerUp);
        el.addEventListener("pointerleave", onPointerLeave);
        // 【失焦收尾】用户在按住琴键时 Alt+Tab 切走、又在窗口外松手，WebView2 不会
        // 送回 pointerup ⇒ `isPointerDown` 永远为真、正弦音一直响，切回来还会
        // "继续试听"。这一条与 utils/gestureFocusGuard 的门禁要求同源：装了
        // pointermove 的手势必须自己收尾。
        window.addEventListener("blur", onPointerUp);

        return () => {
            el.removeEventListener("pointerdown", onPointerDown);
            el.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerup", onPointerUp);
            el.removeEventListener("pointerleave", onPointerLeave);
            window.removeEventListener("blur", onPointerUp);
            stopNote();
        };
    }, [editParam, pitchViewRef]);

    // ── 纵轴标尺：左键点击切换展示单位（音量 / 动态）──────────────────────────
    //
    // 「倍率 ↔ dB」只对**线性幅值倍率**参数有意义（1× = 0 dB，见 paramAxisUnits）。
    // 其余参数的左轴单位由参数语义唯一确定（音高是音名、音分就是音分），因此支持
    // 判定不通过时**完全不注册监听**——点击左轴保持"无操作"，而不是静默写一个
    // 无意义的设置项。
    //
    // 用 `click` 而不是 pointerdown/up 自配对：`click` 只在按下与抬起都落在本元素
    // （或本元素内的子元素）上时触发，天然排除"从轴列拖到画布"这类手势；轴列上
    // 目前没有其它拖拽手势，因此不需要更复杂的裁决。
    useEffect(() => {
        const el = axisWrapRef.current;
        if (!el) return;
        if (!supportsParamAxisUnit(editParam)) return;
        const onClick = (e: MouseEvent) => {
            // 只认左键：中键留给内核的平移手势，右键留给上下文菜单。
            if (e.button !== 0) return;
            dispatch(toggleParamAxisUnit(editParam));
            void dispatch(persistUiSettings());
        };
        el.addEventListener("click", onClick);
        return () => el.removeEventListener("click", onClick);
    }, [dispatch, editParam]);

    // ── 纵轴标尺的悬浮读数（`弹出展示参数`）──────────────────────────────────
    //
    // 与曲线上的浮窗共用同一个开关（`showParamValuePopup`）。读数口径按左轴的**内容**
    // 分两种：
    // - 钢琴卷帘（音高）：**只给音名**（E4 / D4），不给音分 —— 左轴本来就是按琴键
    //   分行画出来的，"E4+12" 并不指向某个键，反而会被误读成另一个音；
    // - 数值刻度（音量 / 动态 / 音分 / 张力…）：给该 y 处的参数值，其中音量 / 动态
    //   按纵轴展示单位读数（切 dB 时就是 dB）。
    useEffect(() => {
        const el = axisWrapRef.current;
        if (!el) return;
        if (!s.showParamValuePopup) {
            setAxisValuePreview(null);
            return;
        }

        const describe = (clientY: number): string => {
            const bounds = el.getBoundingClientRect();
            // 分母必须是**绘图区**高度（`viewSize.h`），不是轴列高度：轴列比滚动
            // 视口高出标尺行与底部滚动条行，用列高会让读数整体偏低（与轴上滚轮
            // 处理同一条约束，见上方 wheel effect 的说明）。
            const h = Math.max(1, viewSizeRef.current.h);
            const y = clamp(clientY - bounds.top, 0, h);
            if (editParam === "pitch") {
                // 与琴键实例同一投影：向下取整到"这一行属于哪个键"。
                return midiToLabel(Math.floor(yToValue("pitch", y, h)));
            }
            return formatParamValuePreview(yToValue(editParam, y, h));
        };

        // 读数合帧：pointermove 高频触发，且 describe 内含 getBoundingClientRect
        // （布局读取）。每帧至多计算 + setState 一次，滞后一帧不可感知。
        let previewRafId: number | null = null;
        let pending:
            | {
                  clientX: number;
                  clientY: number;
                  text: string;
              }
            | null
            | undefined;
        const flushPreview = () => {
            previewRafId = null;
            if (pending === undefined) return;
            // 容器 rect 在这里（rAF 内、事件后）随读数同帧快照：EditorArea
            // 渲染期只消费数值，不再 getBoundingClientRect（渲染期强制同步布局）。
            if (pending) {
                const rect = el.getBoundingClientRect();
                setAxisValuePreview({ ...pending, rectLeft: rect.left, rectTop: rect.top });
            } else {
                setAxisValuePreview(pending);
            }
            pending = undefined;
        };

        const onPointerMove = (e: PointerEvent) => {
            const text = describe(e.clientY);
            pending = text.length === 0 ? null : { clientX: e.clientX, clientY: e.clientY, text };
            if (previewRafId == null) previewRafId = requestAnimationFrame(flushPreview);
        };
        const onPointerLeave = () => {
            pending = null;
            if (previewRafId == null) previewRafId = requestAnimationFrame(flushPreview);
        };

        el.addEventListener("pointermove", onPointerMove);
        el.addEventListener("pointerleave", onPointerLeave);
        return () => {
            el.removeEventListener("pointermove", onPointerMove);
            el.removeEventListener("pointerleave", onPointerLeave);
            if (previewRafId != null) cancelAnimationFrame(previewRafId);
            setAxisValuePreview(null);
        };
    }, [s.showParamValuePopup, editParam, yToValue, formatParamValuePreview]);
}
