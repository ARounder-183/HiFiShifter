//! 原算法参数能力命令的薄适配；DTO与描述符映射由共享编辑内核提供。
pub use hifishifter_kernel::editor::capabilities::ParamDescriptorDto;
pub(super) fn get_processor_params(algo:String)->Vec<ParamDescriptorDto> {
    hifishifter_kernel::editor::capabilities::get_processor_params(algo)
}