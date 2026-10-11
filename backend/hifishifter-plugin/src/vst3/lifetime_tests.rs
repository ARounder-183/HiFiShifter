//! `vst3.rs` 的生命周期测试（组件/控制器引用计数、初始化/终止竞态）。从
//! 内联模块纯搬到这里 —— `use super::*` 解析不变，被测代码整文件可读。

    use super::*;
    use ara2_bridge::companion::vst3::ffi::ara2_vst3_release;
    use std::sync::atomic::AtomicU32;

    /// 从真实工厂创建组件，返回其宿主 owning COM 引用。
    fn create_component() -> *mut c_void {
        let iid = uid_guid(IID_ICOMPONENT);
        let mut component = std::ptr::null_mut();
        // SAFETY: CID/IID 和输出槽均在同步工厂调用期间存活。
        let result = unsafe {
            factory_create_instance(
                std::ptr::null_mut(),
                PROCESSOR_CID.as_ptr().cast(),
                iid.as_ptr().cast(),
                &raw mut component,
            )
        };
        assert_eq!(result, K_RESULT_OK);
        component
    }

    /// 工厂交付控制器时必须消耗自己的初始引用：create + QI 之后，宿主引用应是
    /// 唯一引用，宿主一次 release 就归零并销毁。此前成功分支漏掉工厂侧 release，
    /// 每个控制器实例都确定性泄漏一份 ConnectionState（含 Arc 与两个 Mutex）。
    #[test]
    fn controller_instance_arrives_with_exactly_one_host_reference() {
        let iid = uid_guid(IID_IEDIT_CONTROLLER);
        let mut controller = std::ptr::null_mut();
        // SAFETY: CID/IID 和输出槽均在同步工厂调用期间存活。
        let result = unsafe {
            factory_create_instance(
                std::ptr::null_mut(),
                CONTROLLER_CID.as_ptr().cast(),
                iid.as_ptr().cast(),
                &raw mut controller,
            )
        };
        assert_eq!(result, K_RESULT_OK);
        assert!(!controller.is_null());
        let refcount =
            unsafe { (*controller.cast::<EditController>()).refcount.load(Ordering::Acquire) };
        assert_eq!(refcount, 1, "工厂交付后应只剩宿主那一份引用");
        // 宿主放手后对象必须真正销毁（release 返回 0 = 计数归零并 drop）。
        assert_eq!(unsafe { edit_controller_release(controller) }, 0);
    }

    /// 安全RED：只观察原生refcount，不在尚未保活的实现中重入最终release触发悬空。
    #[test]
    fn task38a_initialize_owns_a_short_component_reference_before_querying_host() {
        let component = create_component();
        let host = crate::host::MockHost::new();
        let observed = Arc::new(AtomicI32::new(0));
        let count = observed.clone();
        let raw = component as usize;
        *host.hook.borrow_mut() = Some((
            "api:GetPlayPositionEx".into(),
            Box::new(move || {
                count.store(
                    unsafe {
                        (*((raw as *mut c_void).cast::<Processor>()))
                            .refcount
                            .load(Ordering::Acquire)
                    },
                    Ordering::Release,
                );
            }),
        ));
        let result = unsafe { component_initialize(component, host.context()) };
        let count = observed.load(Ordering::Acquire);
        assert_eq!(unsafe { component_release(component) }, 0);
        assert_eq!(result, K_RESULT_OK);
        assert_eq!(count, 2, "宿主owning引用之外，本次初始化应独立保活插件存储");
    }

    /// terminate撤销授权与最终release不同；初始化不得把closed组件伪报成成功或重装host引用。
    #[test]
    fn task38a_initialize_reentrant_terminate_is_rejected_and_does_not_reinstall_host() {
        for terminate in [true, false] {
            let component = create_component();
            let host = crate::host::MockHost::new();
            let raw = component as usize;
            let owner =
                unsafe { Arc::downgrade(&(*component.cast::<Processor>()).extension_owner) };
            *host.hook.borrow_mut() = Some((
                "api:GetPlayPositionEx".into(),
                Box::new(move || {
                    if terminate {
                        unsafe {
                            component_terminate(raw as *mut c_void);
                        }
                    } else {
                        owner.upgrade().unwrap().stop_editor();
                    }
                }),
            ));
            let result = unsafe { component_initialize(component, host.context()) };
            let closed = unsafe { (*component.cast::<Processor>()).extension_owner.is_closed() };
            let references = host.references();
            let calls = host.calls();
            assert_eq!(unsafe { component_release(component) }, 0);
            assert_eq!(result, K_RESULT_FALSE);
            assert!(closed);
            assert_eq!(references, 1);
            assert_eq!(calls.last().unwrap(), "api:GetPlayPositionEx");
        }
    }

    /// 完成保活后才执行真正的重入最终release；调用结束不能泄漏该短引用。
    #[test]
    fn task38a_initialize_survives_reentrant_final_release_and_drops_its_short_reference() {
        let component = create_component();
        let host = crate::host::MockHost::new();
        let raw = component as usize;
        let weak = unsafe { Arc::downgrade(&(*component.cast::<Processor>()).extension_owner) };
        let during = weak.clone();
        let remaining = Arc::new(AtomicU32::new(99));
        let observed = remaining.clone();
        let alive = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observed_alive = alive.clone();
        *host.hook.borrow_mut() = Some((
            "api:GetPlayPositionEx".into(),
            Box::new(move || {
                let count = unsafe { component_release(raw as *mut c_void) };
                observed.store(count, Ordering::Release);
                observed_alive.store(during.upgrade().is_some(), Ordering::Release);
            }),
        ));
        assert_eq!(
            unsafe { component_initialize(component, host.context()) },
            K_RESULT_OK
        );
        assert_eq!(remaining.load(Ordering::Acquire), 1);
        assert!(alive.load(Ordering::Acquire));
        assert!(weak.upgrade().is_none());
        assert_eq!(host.references(), 1);
    }

    #[test]
    fn task38a_initialize_of_a_closed_component_does_not_call_host() {
        let component = create_component();
        let host = crate::host::MockHost::new();
        unsafe {
            component_terminate(component);
        }
        host.reset();
        let result = unsafe { component_initialize(component, host.context()) };
        let calls = host.calls();
        assert_eq!(unsafe { component_release(component) }, 0);
        assert_eq!(result, K_RESULT_FALSE);
        assert!(calls.is_empty());
        assert_eq!(host.references(), 1);
    }

    /// IHostApplication QI itself可重入terminate；失败初始化不能留下后来才存入的host owning引用。
    #[test]
    fn task38a_initialize_reentry_during_host_application_qi_releases_the_returned_host_reference()
    {
        let component = create_component();
        let host = crate::host::MockHost::new();
        host.enable_connection_host();
        let raw = component as usize;
        *host.hook.borrow_mut() = Some((
            "QI".into(),
            Box::new(move || unsafe {
                component_terminate(raw as *mut c_void);
            }),
        ));
        let result = unsafe { component_initialize(component, host.context()) };
        let refs = host.references();
        let calls = host.calls();
        assert_eq!(unsafe { component_release(component) }, 0);
        assert_eq!(result, K_RESULT_FALSE);
        assert_eq!(refs, 1, "关闭回调之后QI成功返回的引用也必须立即回收");
        assert_eq!(calls, ["QI"]);
    }

    /// 工厂不能遗留自己的初始引用，否则组件和扩展永远不会释放。
    #[test]
    fn host_component_release_drops_the_extension_owner() {
        let component = create_component();
        // SAFETY: component 是工厂返回的有效 Processor 基址。
        let owner = unsafe { Arc::downgrade(&(*component.cast::<Processor>()).extension_owner) };
        // SAFETY: 消耗唯一宿主组件引用，此后不再使用 component。
        assert_eq!(unsafe { component_release(component) }, 0);
        assert!(owner.upgrade().is_none());
    }

    /// native ARA entry 被宿主持有时，组件先释放也不能让 extension storage 悬空。
    #[test]
    fn host_entry_reference_keeps_the_extension_owner_alive() {
        let component = create_component();
        // SAFETY: 组件仍存活，query_interface 返回新 owning COM 引用。
        let (owner, entry) = unsafe {
            let processor = &*component.cast::<Processor>();
            (
                Arc::downgrade(&processor.extension_owner),
                processor
                    .entry
                    .as_ref()
                    .unwrap()
                    .query_interface(Ara2Vst3InterfaceKind::PluginEntry2)
                    .unwrap(),
            )
        };
        // SAFETY: 消耗宿主组件引用，entry 仍被单独持有。
        assert_eq!(unsafe { component_release(component) }, 0);
        assert!(owner.upgrade().is_some());
        let mut remaining = 0;
        // SAFETY: 最后消耗 entry owning COM 引用，不再访问其指针。
        assert_eq!(
            unsafe { ara2_vst3_release(entry, &raw mut remaining) },
            ARA2_VST3_OK
        );
        assert_eq!(remaining, 0);
        assert!(owner.upgrade().is_none());
    }

    /// 真COM组件、真实route租约与native entry同时持有；释放入口不能牵连同文档/另一文档actor。
    #[test]
    fn task34_real_processor_release_revokes_routes_without_retaining_document() {
        use crate::editor::{
            routing::EditorLink,
            session::{UiRequest, UiSink},
        };
        use ara2_bridge::{core::ApiGeneration, plugin::ExtensionRoles};
        use std::sync::{atomic::AtomicBool, mpsc};
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let weak_document = Arc::downgrade(&document);
        let other_model = crate::ara::model::ModelHandle::new();
        let other_document = other_model.session();
        let components = [create_component(), create_component(), create_component()];
        let mut links = Vec::new();
        let mut editors = Vec::new();
        let mut sinks = Vec::new();
        let mut receivers = Vec::new();
        for (index, component) in components.iter().enumerate() {
            // SAFETY: 三个工厂返回的COM owning引用在循环期间均存活。
            let processor = unsafe { &*component.cast::<Processor>() };
            processor
                .extension_owner
                .bind_to_document(
                    if index == 2 {
                        other_document.clone()
                    } else {
                        document.clone()
                    },
                    ApiGeneration::V2Final,
                    ExtensionRoles::all(),
                    ExtensionRoles::EDITOR_RENDERER,
                    None,
                )
                .unwrap();
            let link = Arc::new(EditorLink::default());
            link.bind(std::process::id() as i64, processor.route.token())
                .unwrap();
            let editor = processor.extension_owner.editor_session().unwrap();
            let (reply, rx) = mpsc::channel();
            let (events, _) = mpsc::sync_channel(8);
            let sink = UiSink {
                view_id: format!("task34-com-{index}"),
                reply,
                events,
                closed: Arc::new(AtomicBool::new(false)),
            };
            editor
                .enqueue(UiRequest {
                    id: 1,
                    command: "get_ui_settings".into(),
                    args: serde_json::json!({}),
                    sink: sink.clone(),
                    link: Some(link.clone()),
                })
                .unwrap();
            assert_eq!(
                rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap()["ok"],
                true
            );
            links.push(link);
            editors.push(editor);
            sinks.push(sink);
            receivers.push(rx);
        }
        assert!(Arc::ptr_eq(&editors[0], &editors[1]));
        assert!(!Arc::ptr_eq(&editors[0], &editors[2]));
        // ARA native entry保留owner，弱引用仍能升级也必须撤销processor入口。
        let (weak_owner, entry) = unsafe {
            let processor = &*components[0].cast::<Processor>();
            (
                Arc::downgrade(&processor.extension_owner),
                processor
                    .entry
                    .as_ref()
                    .unwrap()
                    .query_interface(Ara2Vst3InterfaceKind::PluginEntry2)
                    .unwrap(),
            )
        };
        assert_eq!(unsafe { component_release(components[0]) }, 0);
        assert!(weak_owner.upgrade().is_some());
        assert!(links[0].owner().is_err());
        assert!(sinks[0].closed.load(Ordering::Acquire));
        assert!(!sinks[1].closed.load(Ordering::Acquire));
        assert!(editors[0]
            .enqueue(UiRequest {
                id: 2,
                command: "get_ui_settings".into(),
                args: serde_json::json!({}),
                sink: sinks[0].clone(),
                link: Some(links[0].clone())
            })
            .is_err());
        document.close();
        assert!(sinks[1].closed.load(Ordering::Acquire));
        assert!(links[1].owner().is_err());
        assert!(links[2].owner().is_ok());
        editors[2]
            .enqueue(UiRequest {
                id: 3,
                command: "get_ui_settings".into(),
                args: serde_json::json!({}),
                sink: sinks[2].clone(),
                link: Some(links[2].clone()),
            })
            .unwrap();
        assert_eq!(
            receivers[2]
                .recv_timeout(std::time::Duration::from_secs(3))
                .unwrap()["ok"],
            true
        );
        drop(document);
        drop(model);
        assert!(
            weak_document.upgrade().is_none(),
            "native entry/actor/route不能强持document"
        );
        let mut remaining = 0;
        assert_eq!(
            unsafe { ara2_vst3_release(entry, &raw mut remaining) },
            ARA2_VST3_OK
        );
        assert_eq!(remaining, 0);
        assert!(weak_owner.upgrade().is_none());
        assert_eq!(unsafe { component_release(components[1]) }, 0);
        assert_eq!(unsafe { component_release(components[2]) }, 0);
        other_document.close();
        drop(other_document);
        drop(other_model);
    }

    /// 实际IComponent getState/setState经过IBStream ABI；短读写也必须保存共享actor最新值的组件范围。
    #[test]
    fn task34_real_component_state_stream_flushes_and_saves_only_its_scope() {
        use crate::editor::{
            routing::EditorLink,
            session::{UiRequest, UiSink},
        };
        use ara2_bridge::{core::ApiGeneration, plugin::ExtensionRoles};
        use std::sync::{atomic::AtomicBool, mpsc};
        #[repr(C)]
        struct Stream {
            vtable: *const Vtable,
            bytes: Vec<u8>,
            position: usize,
        }
        #[repr(C)]
        struct Vtable {
            query: usize,
            add_ref: usize,
            release: usize,
            read: unsafe extern "system" fn(*mut c_void, *mut c_void, i32, *mut i32) -> i32,
            write: unsafe extern "system" fn(*mut c_void, *mut c_void, i32, *mut i32) -> i32,
            seek: usize,
            tell: usize,
        }
        unsafe extern "system" fn read(
            this: *mut c_void,
            buffer: *mut c_void,
            count: i32,
            actual: *mut i32,
        ) -> i32 {
            let stream = unsafe { &mut *this.cast::<Stream>() };
            let size = (count as usize)
                .min(13)
                .min(stream.bytes.len() - stream.position);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    stream.bytes.as_ptr().add(stream.position),
                    buffer.cast(),
                    size,
                );
                *actual = size as i32;
            }
            stream.position += size;
            K_RESULT_OK
        }
        unsafe extern "system" fn write(
            this: *mut c_void,
            buffer: *mut c_void,
            count: i32,
            actual: *mut i32,
        ) -> i32 {
            let stream = unsafe { &mut *this.cast::<Stream>() };
            let size = (count as usize).min(11);
            stream.bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(buffer.cast::<u8>(), size)
            });
            unsafe {
                *actual = size as i32;
            }
            K_RESULT_OK
        }
        let vtable = Vtable {
            query: 0,
            add_ref: 0,
            release: 0,
            read,
            write,
            seek: 0,
            tell: 0,
        };
        let (model, old_owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        for owner in old_owners {
            owner.stop_editor();
        }
        let components = [create_component(), create_component()];
        let mut links = Vec::new();
        for (index, component) in components.iter().enumerate() {
            // SAFETY: 工厂返回的processor COM owning引用仍由本测试持有。
            let processor = unsafe { &*component.cast::<Processor>() };
            let raw = processor
                .extension_owner
                .bind_to_document(
                    document.clone(),
                    ApiGeneration::V2Final,
                    ExtensionRoles::all(),
                    ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
                    None,
                )
                .unwrap();
            let key = (&*ids[index] as *const u8) as u64;
            unsafe {
                let ext = &*raw;
                ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                    ext.playbackRendererRef,
                    key as *mut _,
                );
            }
            let link = Arc::new(EditorLink::default());
            link.bind(std::process::id() as i64, processor.route.token())
                .unwrap();
            links.push(link);
        }
        let editor = document.editor_session().unwrap();
        let (reply, rx) = mpsc::channel();
        let (events, _) = mpsc::sync_channel(128);
        let sink = UiSink {
            view_id: "task34-state-stream".into(),
            reply,
            events,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let request = |id, command: &str, args| {
            editor
                .enqueue(UiRequest {
                    id,
                    command: command.into(),
                    args,
                    sink: sink.clone(),
                    link: Some(links[0].clone()),
                })
                .unwrap();
            let response = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
            assert_eq!(response["ok"], true, "{response}");
            response["value"].clone()
        };
        let timeline = request(1, "get_timeline_state", serde_json::json!({}));
        for (index, volume) in [0.5, 0.25].into_iter().enumerate() {
            request(
                2,
                "set_track_state",
                serde_json::json!({"trackId":timeline["tracks"][index]["id"],"volume":volume}),
            );
        }
        for (index, id) in ["track", "b"].into_iter().enumerate() {
            let mut stream = Stream {
                vtable: &vtable,
                bytes: Vec::new(),
                position: 0,
            };
            assert_eq!(
                unsafe { (COMPONENT_VTBL.get_state)(components[index], (&raw mut stream).cast()) },
                K_RESULT_OK
            );
            let length = u32::from_le_bytes(stream.bytes[..4].try_into().unwrap()) as usize;
            assert_eq!(stream.bytes.len(), length + 4);
            let saved: serde_json::Value = serde_json::from_slice(&stream.bytes[4..]).unwrap();
            // source basis已随自动初始化进入v3；仍须证明IBStream仅保存当前组件范围。
            assert_eq!(saved["version"], 3);
            assert_eq!(saved["edits"]["tracks"].as_array().unwrap().len(), 1);
            assert_eq!(saved["edits"]["tracks"][0]["id"], id);
            assert_eq!(
                saved["edits"]["tracks"][0]["volume"],
                if index == 0 { 0.5 } else { 0.25 }
            );
            assert_eq!(saved["edits"]["bindings"].as_object().unwrap().len(), 1);
            let regions = saved["edits"]["atlas"]["regions"]
                .as_object()
                .expect("v3 source basis");
            assert_eq!(regions.len(), 1);
            assert!(regions.values().all(|region| region["root"] == id));
            assert_eq!(
                unsafe { (COMPONENT_VTBL.set_state)(components[index], (&raw mut stream).cast()) },
                K_RESULT_OK
            );
        }
        document.close();
        for component in components {
            assert_eq!(unsafe { component_release(component) }, 0);
        }
    }
