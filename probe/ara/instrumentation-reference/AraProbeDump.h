//------------------------------------------------------------------------------
//! \file       AraProbeDump.h
//!             ARA 模型图 dump —— HiFiShifter 探针专用
//! \project    HiFiShifter ARA2 probe
//!
//! 作用：把宿主（DAW）通过 ARA 提供的整个模型图序列化为 JSON 落盘，供
//!       `docs/superpowers/plans/2026-10-04-ara-bridge-probe.md` 的 Task 1
//!       与 Task 3 使用（Task 3 拿它当转换器测试夹具）。
//!
//! 与其他模块的关系：这是对 ARA SDK 官方示例 Test Plug-In 的**只增不改**式插桩。
//!       SDK 源码本身不动；本文件是新增翻译单元，只在
//!       `ARATestDocumentController` 的两个回调里被调用。
//!
//! 特殊说明：
//!   - 输出路径优先取环境变量 `ARA_PROBE_OUT`；未设时在若干候选位置中选第一个
//!     可写目录，落 `ara-model.json`。**每次调用整体重写**（快照语义）。
//!   - 本探针代码是一次性产物，不属于任何产品代码路径。
//------------------------------------------------------------------------------

#pragma once

namespace ARA
{
    namespace PlugIn
    {
        class DocumentController;
    }
}

//! 把 ARA 文档的当前模型图整体 dump 为 JSON。
//!
//! 流程：从 DocumentController 取 Document，遍历
//!       audioSources / musicalContexts(regionSequences) / audioModifications /
//!       playbackRegions，拼成 JSON 后写盘。
//! 作用：产出 Task 1 需要的"宿主到底给了什么"的一手证据。
//! 特殊说明：本函数在文档变更回调（主线程）中调用，可能被高频触发；
//!       实现内部做了节流，见 .cpp。失败只写日志，绝不抛异常影响宿主。
//! 参数：documentController —— 当前文档控制器；不得为 nullptr。
void AraProbeDumpToFile (ARA::PlugIn::DocumentController* documentController) noexcept;
