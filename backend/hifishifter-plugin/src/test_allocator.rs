//! 测试专用实时内存守卫：仅计数当前测试线程，不干扰并发测试的初始化/析构。

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

std::thread_local! {
    static OBSERVING: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
struct GuardedAllocator;
#[global_allocator]
static ALLOCATOR: GuardedAllocator = GuardedAllocator;

fn note() {
    let _=OBSERVING.try_with(|flag| { if flag.get() { let _=CALLS.try_with(|calls| calls.set(calls.get()+1)); } });
}
// SAFETY: 每个操作只计数然后原样委托给系统分配器，不改变地址、布局或所有权。
unsafe impl GlobalAlloc for GuardedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: 调用者的 GlobalAlloc 合约原样传给 System。
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: 原样传递合法布局。
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        note();
        // SAFETY: 指针与原布局来自本分配器（实际 System）。
        unsafe { System.dealloc(pointer,layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        note();
        // SAFETY: 合法重分配参数原样委托。
        unsafe { System.realloc(pointer,layout,size) }
    }
}
pub(crate) fn begin() { CALLS.with(|calls| calls.set(0)); OBSERVING.with(|flag| flag.set(true)); }
pub(crate) fn end() -> usize { OBSERVING.with(|flag| flag.set(false)); CALLS.with(Cell::get) }
