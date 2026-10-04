//! VocalShifter Library (vslib) FFI bindings.
//!
//! vslib v1.56 — by あっきー (ackiesound@hotmail.co.jp)
//! License: free for personal/non-commercial freeware only.
//!
//! Provides pitch analysis, formant shifting, time stretching,
//! breathiness control and waveform synthesis via VocalShifter's
//! proprietary engine or the WORLD vocoder backend.
//!
//! DLL location: third_party/vslib/vslib_x64.dll (Windows x64)

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_double, c_int, c_uint, c_void};

//--------------------------------------------------------------
// 宏定义常量
//--------------------------------------------------------------
pub const VSLIB_MAX_PATH: usize = 256;
pub const VSLIB_MAX_TRACK: usize = 64;
pub const VSLIB_MAX_ITEM: usize = 1024;

// 错误码
pub const VSERR_NOERR: c_int = 0;
pub const VSERR_PRM: c_int = 1;
pub const VSERR_PRJOPEN: c_int = 2;
pub const VSERR_PRJFORMAT: c_int = 3;
pub const VSERR_WAVEOPEN: c_int = 4;
pub const VSERR_WAVEFORMAT: c_int = 5;
pub const VSERR_FREQ: c_int = 6;
pub const VSERR_MAX: c_int = 7;
pub const VSERR_NOMEM: c_int = 8;

// 合成模式
pub const SYNTHMODE_M: c_int = 0; // 单音
pub const SYNTHMODE_MF: c_int = 1; // 单音 + 共振峰补偿
pub const SYNTHMODE_P: c_int = 2; // 和音

// 分析选项
pub const ANALYZE_OPTION_SKIP_PIT_ANALYZE: c_uint = 0x00000001;
pub const ANALYZE_OPTION_WORLD: c_uint = 0x00000000; // WORLD 分析引擎
pub const ANALYZE_OPTION_VOCAL_SHIFTER: c_uint = 0x00000002; // VocalShifter 自有引擎

//--------------------------------------------------------------
// 句柄
//--------------------------------------------------------------
#[repr(C)]
pub struct HVSPRJ__ {
    _unused: c_int,
}
pub type HVSPRJ = *mut HVSPRJ__;

//--------------------------------------------------------------
// 结构体
//--------------------------------------------------------------

/// 项目信息
#[repr(C)]
pub struct VSPRJINFO {
    pub masterVolume: c_double, // 主音量倍数
    pub sampFreq: c_int,        // 采样率 [Hz]
}

/// 轨道信息（基础）
#[repr(C)]
pub struct VSTRACKINFO {
    pub volume: c_double,   // 音量倍数
    pub pan: c_double,      // 声像 (-1.0 ~ 1.0)
    pub invPhaseFlg: c_int, // 反相标志
    pub soloFlg: c_int,     // 独奏标志
    pub muteFlg: c_int,     // 静音标志
}

/// 轨道信息（扩展）
#[repr(C)]
pub struct VSTRACKINFOEX {
    pub name: [c_char; 64],
    pub volume: c_double,
    pub pan: c_double,
    pub reserved1: [c_double; 2],
    pub invPhaseFlg: c_int,
    pub soloFlg: c_int,
    pub muteFlg: c_int,
    pub color: c_int,
    pub morphGroup: c_int,
    pub option: c_uint,
    pub reserved2: [c_int; 10],
}

/// 素材（Item）信息
#[repr(C)]
pub struct VSITEMINFO {
    pub fileName: [c_char; VSLIB_MAX_PATH],
    pub sampFreq: c_int,   // 采样率 [Hz]
    pub channel: c_int,    // 声道数
    pub sampleOrg: c_int,  // 原始样本数
    pub sampleEdit: c_int, // 编辑后样本数
    pub ctrlPntPs: c_int,  // 每秒控制点数
    pub ctrlPntNum: c_int, // 总控制点数
    pub synthMode: c_int,  // 合成模式 (R/W)
    pub trackNum: c_int,   // 轨道编号 (R/W)
    pub offset: c_int,     // 偏移量 [sample] (R/W)
}

/// 控制点信息（扩展）
#[repr(C)]
pub struct VSCPINFOEX {
    pub dynOrg: c_double,   // 编辑前动态倍数（只读）
    pub dynEdit: c_double,  // 编辑后动态倍数
    pub volume: c_double,   // 音量倍数
    pub pan: c_double,      // 声像 (-1.0 ~ 1.0)
    pub spcDyn: c_double,   // 频谱动态（只读）
    pub pitAna: c_int,      // 音高分析值 [cent]（只读）
    pub pitOrg: c_int,      // 编辑前音高 [cent]
    pub pitEdit: c_int,     // 编辑后目标音高 [cent]（核心写入字段）
    pub formant: c_int,     // 共振峰偏移 [cent]
    pub pitFlgOrg: c_int,   // 编辑前有声标志（只读）
    pub pitFlgEdit: c_int,  // 编辑后有声标志
    pub breathiness: c_int, // 气声强度 (-10000 ~ 10000)
    pub eq1: c_int,         // EQ1 (-10000 ~ 10000)
    pub eq2: c_int,         // EQ2 (-10000 ~ 10000)
}

/// 控制点信息（扩展 v2，含 HEQ）
#[repr(C)]
pub struct VSCPINFOEX2 {
    pub dynOrg: c_double,
    pub dynEdit: c_double,
    pub volume: c_double,
    pub pan: c_double,
    pub spcDyn: c_double,
    pub reserved1: c_double,
    pub pitAna: c_int,
    pub pitOrg: c_int,
    pub pitEdit: c_int,
    pub formant: c_int,
    pub pitFlgOrg: c_int,
    pub pitFlgEdit: c_int,
    pub breathiness: c_int,
    pub eq1: c_int,
    pub eq2: c_int,
    pub heq: c_int, // 谐波 EQ (-10000 ~ 10000)
    pub reserved2: [c_int; 6],
}

/// ASAnalyzer 分析参数
#[repr(C)]
pub struct AWDINFO {
    pub wavdatasize: c_int, // 数据大小 [样本]
    pub wavsampleps: c_int, // 采样率 [Hz]
    pub wavbit: c_int,      // 位深 (8 或 16)
    pub wavchannel: c_int,  // 声道数 (1 或 2)
    pub nnoffset: c_int,    // 最低分析音高 [Note Number]
    pub nnrange: c_int,     // 分析音域范围 [半音]
    pub blockpn: c_int,     // 每半音的块数 (1-20)
    pub dynsize: c_int,     // 音量检测区间 [样本]
    pub targetch: c_int,    // 目标声道
}

pub fn vslib_error_name(code: c_int) -> &'static str {
    match code {
        VSERR_NOERR => "VSERR_NOERR",
        VSERR_PRM => "VSERR_PRM",
        VSERR_PRJOPEN => "VSERR_PRJOPEN",
        VSERR_PRJFORMAT => "VSERR_PRJFORMAT",
        VSERR_WAVEOPEN => "VSERR_WAVEOPEN",
        VSERR_WAVEFORMAT => "VSERR_WAVEFORMAT",
        VSERR_FREQ => "VSERR_FREQ",
        VSERR_MAX => "VSERR_MAX",
        VSERR_NOMEM => "VSERR_NOMEM",
        _ => "VSERR_UNKNOWN",
    }
}

//--------------------------------------------------------------
// 外部函数声明 (Windows system ABI)
//--------------------------------------------------------------
#[link(name = "vslib_x64")]
extern "system" {
    // 库版本
    pub fn VslibGetVersion() -> c_int;

    // 项目生命周期
    pub fn VslibCreateProject(hVsprj: *mut HVSPRJ) -> c_int;
    pub fn VslibOpenProject(hVsprj: *mut HVSPRJ, fileName: *const c_char) -> c_int;
    pub fn VslibSaveProject(hVsprj: HVSPRJ, fileName: *const c_char) -> c_int;
    pub fn VslibDeleteProject(hVsprj: HVSPRJ) -> c_int;
    pub fn VslibGetProjectInfo(hVsprj: HVSPRJ, info: *mut VSPRJINFO) -> c_int;
    pub fn VslibSetProjectInfo(hVsprj: HVSPRJ, info: *mut VSPRJINFO) -> c_int;

    // 轨道操作
    pub fn VslibGetTrackMaxNum(hVsprj: HVSPRJ, trackMaxNum: *mut c_int) -> c_int;
    pub fn VslibAddTrack(hVsprj: HVSPRJ) -> c_int;
    pub fn VslibDeleteTrack(hVsprj: HVSPRJ, trackNum: c_int) -> c_int;
    pub fn VslibGetTrackInfo(hVsprj: HVSPRJ, trackNum: c_int, info: *mut VSTRACKINFO) -> c_int;
    pub fn VslibSetTrackInfo(hVsprj: HVSPRJ, trackNum: c_int, info: *mut VSTRACKINFO) -> c_int;
    pub fn VslibGetTrackInfoEx(hVsprj: HVSPRJ, trackNum: c_int, info: *mut VSTRACKINFOEX) -> c_int;
    pub fn VslibSetTrackInfoEx(hVsprj: HVSPRJ, trackNum: c_int, info: *mut VSTRACKINFOEX) -> c_int;

    // 素材操作
    pub fn VslibGetItemMaxNum(hVsprj: HVSPRJ, itemMaxNum: *mut c_int) -> c_int;
    pub fn VslibAddItem(hVsprj: HVSPRJ, fileName: *const c_char, itemNum: *mut c_int) -> c_int;
    pub fn VslibAddItemEx(
        hVsprj: HVSPRJ,
        fileName: *const c_char,
        itemNum: *mut c_int,
        nnOffset: c_int,
        nnRange: c_int,
        option: c_uint,
    ) -> c_int;
    pub fn VslibDeleteItem(hVsprj: HVSPRJ, itemNum: c_int) -> c_int;
    pub fn VslibGetItemInfo(hVsprj: HVSPRJ, itemNum: c_int, info: *mut VSITEMINFO) -> c_int;
    pub fn VslibSetItemInfo(hVsprj: HVSPRJ, itemNum: c_int, info: *mut VSITEMINFO) -> c_int;

    // 控制点读写
    pub fn VslibGetCtrlPntInfoEx(
        hVsprj: HVSPRJ,
        itemNum: c_int,
        pntNum: c_int,
        info: *mut VSCPINFOEX,
    ) -> c_int;
    pub fn VslibSetCtrlPntInfoEx(
        hVsprj: HVSPRJ,
        itemNum: c_int,
        pntNum: c_int,
        info: *mut VSCPINFOEX,
    ) -> c_int;
    pub fn VslibGetCtrlPntInfoEx2(
        hVsprj: HVSPRJ,
        itemNum: c_int,
        pntNum: c_int,
        info: *mut VSCPINFOEX2,
    ) -> c_int;
    pub fn VslibSetCtrlPntInfoEx2(
        hVsprj: HVSPRJ,
        itemNum: c_int,
        pntNum: c_int,
        info: *mut VSCPINFOEX2,
    ) -> c_int;

    // EQ
    pub fn VslibGetEQGain(hVsprj: HVSPRJ, itemNum: c_int, eqNum: c_int, gain: *mut c_int) -> c_int;
    pub fn VslibSetEQGain(hVsprj: HVSPRJ, itemNum: c_int, eqNum: c_int, gain: *mut c_int) -> c_int;
    pub fn VslibGetHEQGain(hVsprj: HVSPRJ, itemNum: c_int, gain: *mut c_int) -> c_int;
    pub fn VslibSetHEQGain(hVsprj: HVSPRJ, itemNum: c_int, gain: *mut c_int) -> c_int;

    // Timing 控制点（时间拉伸）
    pub fn VslibGetTimeCtrlPntNum(hVsprj: HVSPRJ, itemNum: c_int, num: *mut c_int) -> c_int;
    pub fn VslibGetTimeCtrlPnt(
        hVsprj: HVSPRJ,
        itemNum: c_int,
        pntNum: c_int,
        time1: *mut c_int,
        time2: *mut c_int,
    ) -> c_int;
    pub fn VslibSetTimeCtrlPnt(
        hVsprj: HVSPRJ,
        itemNum: c_int,
        pntNum: c_int,
        time1: c_int,
        time2: c_int,
    ) -> c_int;
    pub fn VslibAddTimeCtrlPnt(hVsprj: HVSPRJ, itemNum: c_int, time1: c_int, time2: c_int)
        -> c_int;
    pub fn VslibDeleteTimeCtrlPnt(hVsprj: HVSPRJ, itemNum: c_int, pntNum: c_int) -> c_int;
    pub fn VslibGetStretchOrgSec(
        hVsprj: HVSPRJ,
        itemNum: c_int,
        time_edt: c_double,
        time_org: *mut c_double,
    ) -> c_int;
    pub fn VslibGetStretchEditSec(
        hVsprj: HVSPRJ,
        itemNum: c_int,
        time_org: c_double,
        time_edt: *mut c_double,
    ) -> c_int;

    // 混音输出
    pub fn VslibGetStretchOrgSample(
        hVsprj: HVSPRJ,
        itemNum: c_int,
        time_edt: c_double,
        time_org: *mut c_double,
    ) -> c_int;
    pub fn VslibGetStretchEditSample(
        hVsprj: HVSPRJ,
        itemNum: c_int,
        time_org: c_double,
        time_edt: *mut c_double,
    ) -> c_int;
    pub fn VslibGetMixSample(hVsprj: HVSPRJ, mixSample: *mut c_int) -> c_int;
    pub fn VslibGetMixData(
        hVsprj: HVSPRJ,
        waveData: *mut c_void,
        bit: c_int,
        channel: c_int,
        index: c_int,
        size: c_int,
    ) -> c_int;
    pub fn VslibExportWaveFile(
        hVsprj: HVSPRJ,
        fileName: *const c_char,
        bit: c_int,
        channel: c_int,
    ) -> c_int;

    // 单位换算
    pub fn VslibCent2Freq(cent: c_int) -> c_double;
    pub fn VslibFreq2Cent(freq: c_double) -> c_int;
    pub fn VslibNoteNum2Freq(noteNum: c_int) -> c_double;
    pub fn VslibFreq2NoteNum(freq: c_double) -> c_int;

    // 批量音高写入
    pub fn VslibSetPitchArray(
        hVsprj: HVSPRJ,
        itemNum: c_int,
        pitData: *mut c_int,
        nPitData: c_int,
        interval: c_double,
    ) -> c_int;
}

//--------------------------------------------------------------
// 安全封装
//--------------------------------------------------------------

/// vslib 错误类型
#[derive(Debug)]
pub enum VslibError {
    Code(c_int),
}

impl std::fmt::Display for VslibError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VslibError::Code(c) => write!(f, "vslib error code {c} ({})", vslib_error_name(*c)),
        }
    }
}

impl std::error::Error for VslibError {}

/// 检查 vslib 返回码，非零转为 Err
pub fn check(code: c_int) -> Result<(), VslibError> {
    if code == VSERR_NOERR {
        Ok(())
    } else {
        Err(VslibError::Code(code))
    }
}

/// RAII 封装：自动调用 VslibDeleteProject
pub struct VsProject(pub HVSPRJ);

impl VsProject {
    pub fn create() -> Result<Self, VslibError> {
        let mut h: HVSPRJ = std::ptr::null_mut();
        check(unsafe { VslibCreateProject(&mut h) })?;
        Ok(VsProject(h))
    }

    pub fn item_info(&self, item_num: c_int) -> Result<VSITEMINFO, VslibError> {
        // SAFETY: VSITEMINFO is a plain C struct; hVsprj is valid while self is alive
        let mut info = unsafe { std::mem::zeroed::<VSITEMINFO>() };
        check(unsafe { VslibGetItemInfo(self.0, item_num, &mut info) })?;
        Ok(info)
    }

    pub fn ctrl_pnt_ex(&self, item_num: c_int, pnt: c_int) -> Result<VSCPINFOEX, VslibError> {
        let mut cp = unsafe { std::mem::zeroed::<VSCPINFOEX>() };
        check(unsafe { VslibGetCtrlPntInfoEx(self.0, item_num, pnt, &mut cp) })?;
        Ok(cp)
    }

    pub fn set_ctrl_pnt_ex(
        &self,
        item_num: c_int,
        pnt: c_int,
        cp: &mut VSCPINFOEX,
    ) -> Result<(), VslibError> {
        check(unsafe { VslibSetCtrlPntInfoEx(self.0, item_num, pnt, cp) })
    }

    pub fn mix_sample(&self) -> Result<c_int, VslibError> {
        let mut n: c_int = 0;
        check(unsafe { VslibGetMixSample(self.0, &mut n) })?;
        Ok(n)
    }

    /// 取混音 PCM 数据（16-bit 立体声）到给定 buffer。
    /// `buf` 长度须为帧数 × 2（channel=2 立体声 i16 交错）。
    /// size 参数 = 帧数（= buf.len() / 2），vslib 内部 × channel 计算写入量。
    pub fn mix_data_i16(&self, buf: &mut [i16], index: c_int) -> Result<(), VslibError> {
        check(unsafe {
            VslibGetMixData(
                self.0,
                buf.as_mut_ptr() as *mut c_void,
                16,
                2,
                index,
                (buf.len() / 2) as c_int, // 帧数 = 总 i16 / channel(2)
            )
        })
    }
}

impl Drop for VsProject {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { VslibDeleteProject(self.0) };
        }
    }
}
