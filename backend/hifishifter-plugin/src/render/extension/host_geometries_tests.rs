//! 多 region 几何匹配的纯函数回归。

    use crate::sync::MutexExt;

    /// 把某条 region 调成与夹具 item 相容的窗口。
    ///
    /// 夹具的 item 几何是 `D_POSITION=1 / D_LENGTH=4 / D_STARTOFFS=0 / D_PLAYRATE=0.5`。
    fn align_with_fixture_item(region: &mut crate::ara::AraPlaybackRegion) {
        region.start_in_playback_time = 1.;
        region.duration_in_playback_time = 4.;
        region.start_in_modification_time = 0.;
        region.duration_in_modification_time = 2.;
    }

    fn fixture_host() -> Box<crate::host::MockHost> {
        let host = crate::host::MockHost::new();
        host.enable_media();
        host.clear_markers();
        host.inventory_enabled.set(true);
        host
    }

    /// 一组 region 里能对上的那些被绑定，对不上的保持未绑定（不猜、不兜底）。
    #[test]
    fn only_the_regions_matching_a_host_item_are_bound() {
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let owner = &owners[0];
        let first = (&*ids[0] as *const u8) as u64;
        let second = (&*ids[1] as *const u8) as u64;
        // 夹具只有一个 item，先让它与 first 相容；second 留在原位（对不上）。
        {
            let mut regions = document.regions.lock_or_recover();
            align_with_fixture_item(regions.get_mut(&first).unwrap());
        }
        // 把第二条 region 也分配给同一个 owner：这才是"一个实例管一组 region"。
        let raw = owner.binding.lock_or_recover().as_ref().unwrap().as_raw();
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                second as *mut _,
            );
        }
        let host = fixture_host();
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        let bound = owner.host_geometries().expect("至少一条 region 必须绑定");
        assert_eq!(bound.len(), 1, "对不上的 region 不得被绑定");
        assert_eq!(bound[0].region_key, first);
        assert!(bound[0].geometry.item_id.starts_with('{'));
        document.close();
    }

    /// 两条 region 都指向同一个 item ⇒ 只保留**一个**认领者，绝不留下两个指向同一
    /// 对象的绑定。
    ///
    /// 【为什么不"两个都丢"】旧实现把重复认领的候选**全部**丢弃，于是"同一位置两个
    /// 几何全等的 item"会让两个 clip 一起消失（用户报障："片段凭空没了"）。现在按
    /// region key 的稳定顺序保留第一个认领者：仍然满足"一个 item 只归一个 region"
    /// （后续写回不会互相覆盖），但不会一次丢掉全部。
    #[test]
    fn two_regions_claiming_one_item_keep_exactly_one_owner() {
        let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
        let document = model.session();
        let owner = &owners[0];
        let first = (&*ids[0] as *const u8) as u64;
        let second = (&*ids[1] as *const u8) as u64;
        {
            let mut regions = document.regions.lock_or_recover();
            align_with_fixture_item(regions.get_mut(&first).unwrap());
            align_with_fixture_item(regions.get_mut(&second).unwrap());
        }
        let raw = owner.binding.lock_or_recover().as_ref().unwrap().as_raw();
        unsafe {
            let ext = &*raw;
            ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
                ext.playbackRendererRef,
                second as *mut _,
            );
        }
        let host = fixture_host();
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        let bound = owner
            .host_geometries()
            .expect("重复认领必须留下恰好一个绑定，不能整组丢弃");
        assert_eq!(bound.len(), 1, "一个 item 只能归一个 region");
        assert!(
            bound[0].region_key == first || bound[0].region_key == second,
            "保留的必须是其中一个真实认领者"
        );
        document.close();
    }
