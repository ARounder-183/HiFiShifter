/**
 * 一次"多步骤导入"的取消闸门。
 *
 * 【为什么必须有】目录导入的逐文件循环会持续数秒（上千个文件时更久），而
 * `begin_undo_group` 只负责"把多条命令合成一条历史"，它**不阻止用户在此期间撤销**。
 * 用户按下 Ctrl+Z 之后，循环若继续往已经消失的 trackId 上导 clip，后端 `add_clip`
 * 会在轨道不存在时凭空造一条 "Track" —— 撤销栈就此被写成"撤回后又冒出新东西"，
 * 用户再也回不到导入前。
 *
 * 【为什么是模块级而不是 Redux】闸门要能被**循环的下一轮同步读到**，而 Redux 的
 * 派发是异步的；且它只服务这几段循环，不参与任何渲染。与 `selectionEditInFlight`
 * 同一条理由。
 *
 * 【为什么用"代次"而不是每轮一个令牌】一次目录导入可能**委托**给
 * `importMultipleAudioAtPosition`（不建轨道组时），被委托的那段循环拿不到发起方的
 * 令牌。代次是全局的：每个循环在开始时记下当时的代次，`cancelActiveImports` 一加一，
 * 所有在途循环的下一轮都会发现"代次变了"，委托关系无需传递任何东西。
 */

/** 当前代次。每次"否决在途导入"加一。 */
let generation = 0;

/** 记下一次可取消导入所处的代次（在循环开始前调用）。 */
export function currentImportGeneration(): number {
    return generation;
}

/**
 * 否决所有在途导入。
 *
 * 由"历史位置被改变"的入口调用：撤销 / 重做 / 历史跳转，以及打开 / 新建工程
 * （历史被清空，继续灌没有意义）。
 */
export function cancelActiveImports(): void {
    generation += 1;
}

/** 该代次的导入是否已被否决（代次变了即为是）。 */
export function isImportCancelled(gen: number): boolean {
    return gen !== generation;
}

/** 在途的"多步骤导入"（见 `registerImportRun`）。 */
const inFlightRuns = new Set<Promise<unknown>>();

/** 一次多步骤导入的登记句柄；`finish()` 必须在其 `finally` 里调用。 */
export interface ImportRunRegistration {
    finish(): void;
}

/**
 * 登记一次"多步骤导入"，直到 `finish()` 为止都算在途。
 *
 * 【为什么需要】导入循环里的每条后端命令（建树 / 导 clip）都是独立的 IPC。若用户在
 * 某一条**在途期间**按下撤销，那条命令会在撤销**之后**才落地：建树会让轨道组重新
 * 出现，导 clip 会往已消失的轨道上写（后端 `add_clip` 会在轨道不存在时凭空造一条
 * "Track"）。前后端就此分叉，而用户看到的正是"撤销了，轨道却还在"。
 * 登记之后，`cancelActiveImportsAndDrain` 能等到这些命令全部收尾再跳转历史。
 */
export function registerImportRun(): ImportRunRegistration {
    let finish!: () => void;
    const settled = new Promise<void>((resolve) => {
        finish = resolve;
    });
    const tracked = settled.finally(() => {
        inFlightRuns.delete(tracked);
    });
    inFlightRuns.add(tracked);
    return { finish };
}

/**
 * 否决所有在途导入，并**等它们收尾**。
 *
 * 历史跳转前必须调用它（而不是裸的 `cancelActiveImports`）：等收尾之后才跳转，
 * 就不会有"撤销之后又落地一条导入命令"的交错。
 */
export async function cancelActiveImportsAndDrain(): Promise<void> {
    cancelActiveImports();
    await Promise.allSettled([...inFlightRuns]);
}

/** 仅测试用：把代次与在途登记复位，避免用例之间互相污染。 */
export function resetImportCancellationForTests(): void {
    generation = 0;
    inFlightRuns.clear();
}
