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

/** 仅测试用：把代次复位，避免用例之间互相污染。 */
export function resetImportCancellationForTests(): void {
    generation = 0;
}
