//! `render/extension.rs` 的 `region_geometry_match_tests` 测试（从内联模块纯搬到这里，`use super::*` 解析不变）。

    use super::*;
    use crate::ara::AraPlaybackRegion;
    use crate::host::geometry::HostClipGeometry;

    fn candidate(
        item: &str,
        start_sec: f64,
        duration_sec: f64,
        source_start_sec: f64,
        playback_rate: f64,
    ) -> HostClipGeometry {
        HostClipGeometry {
            item_id: item.into(),
            take_id: format!("{item}-take"),
            start_sec,
            source_start_sec,
            duration_sec,
            snap_offset_sec: 0.,
            playback_rate,
            // 测试夹具按正放、无 SECTION 构造；符号与幅值一致。
            play_rate_signed: playback_rate,
            reversed: false,
            section: None,
            preserve_pitch: true,
            channel_mode: 0,
            take_pitch: 0.,
            item_timebase: 0,
            auto_stretch: false,
            muted: false,
            item_gain: 1.,
            group_id: 0,
            take_gain: 1.,
            markers: Vec::new(),
            fade_in_sec: 0.,
            fade_out_sec: 0.,
            fade_in_shape: 0.,
            fade_out_shape: 0.,
            fade_in_dir: 0.,
            fade_out_dir: 0.,
            fade_in_dir_new: 0.,
            fade_out_dir_new: 0.,
            fade_in_dir2_new: 0.,
            fade_out_dir2_new: 0.,
            fade_axes_v2: None,
            auto_fade_in_sec: 0.,
            auto_fade_out_sec: 0.,
            loop_source: false,
            source_file_name: None,
        }
    }

    fn region(
        start_in_playback_time: f64,
        duration_in_playback_time: f64,
        start_in_modification_time: f64,
        duration_in_modification_time: f64,
    ) -> AraPlaybackRegion {
        AraPlaybackRegion {
            start_in_playback_time,
            duration_in_playback_time,
            start_in_modification_time,
            duration_in_modification_time,
            ..Default::default()
        }
    }

    /// 四项全相容才绑定：位置、时长、源内起点、速率。
    #[test]
    fn a_region_binds_to_the_item_matching_all_four_quantities() {
        let regions = [
            candidate("item-a", 0., 2., 0., 1.),
            candidate("item-b", 4., 2., 0., 1.),
        ];
        let bound = unique_geometry_for_region(&region(4., 2., 0., 2.), &regions, None).unwrap();
        assert_eq!(bound.item_id, "item-b");
    }

    /// 同一位置的两条子轨各有一个 item：只比时间窗会歧义，源内起点把两者分开。
    #[test]
    fn the_source_window_disambiguates_items_at_the_same_playback_position() {
        let regions = [
            candidate("trimmed-head", 0., 2., 1., 1.),
            candidate("full", 0., 2., 0., 1.),
        ];
        let bound = unique_geometry_for_region(&region(0., 2., 1., 2.), &regions, None).unwrap();
        assert_eq!(bound.item_id, "trimmed-head");
    }

    /// 速率也是判据：同位置同时长的两个 item，拉伸比不同。
    #[test]
    fn the_playback_rate_disambiguates_stretched_items() {
        let regions = [
            candidate("stretched", 0., 2., 0., 2.),
            candidate("unstretched", 0., 2., 0., 1.),
        ];
        let bound = unique_geometry_for_region(&region(0., 2., 0., 4.), &regions, None).unwrap();
        assert_eq!(bound.item_id, "stretched");
    }

    /// 四项完全相同的两个 item ⇒ 歧义 ⇒ **不绑定**（而不是取第一个）。
    #[test]
    fn ambiguous_candidates_are_refused_rather_than_guessed() {
        let regions = [
            candidate("take-one", 0., 2., 0., 1.),
            candidate("take-two", 0., 2., 0., 1.),
        ];
        assert!(unique_geometry_for_region(&region(0., 2., 0., 2.), &regions, None).is_none());
    }

    /// 歧义时，**上一次绑定的那个**候选可以打破它（稳定性，不是猜测）。
    ///
    /// 【为什么这条必须有】复制粘贴出来的安全副本、同一素材的多个 item 会给出几何
    /// 全等的候选。旧实现把它们**全部**判为歧义、两个都丢 —— 用户看到片段凭空消失。
    /// 保持上次的绑定不会把写回引到新对象上：那次绑定本身就来自一次唯一匹配。
    #[test]
    fn a_previous_binding_disambiguates_otherwise_identical_candidates() {
        let regions = [
            candidate("take-one", 0., 2., 0., 1.),
            candidate("take-two", 0., 2., 0., 1.),
        ];
        let bound = unique_geometry_for_region(&region(0., 2., 0., 2.), &regions, Some("take-two"))
            .expect("the previously bound item must win");
        assert_eq!(bound.item_id, "take-two");
        // 上次绑定的那个已经不在候选里时，仍然不猜。
        assert!(
            unique_geometry_for_region(&region(0., 2., 0., 2.), &regions, Some("gone")).is_none()
        );
    }

    /// 没有候选、或候选都对不上时返回 None，绝不退回"第一个 item"。
    #[test]
    fn a_region_without_a_matching_item_stays_unbound() {
        assert!(unique_geometry_for_region(&region(0., 2., 0., 2.), &[], None).is_none());
        let regions = [candidate("elsewhere", 10., 2., 0., 1.)];
        assert!(unique_geometry_for_region(&region(0., 2., 0., 2.), &regions, None).is_none());
    }

    /// 非有限的播放时长不产生"恰好匹配 NaN"的假绑定。
    #[test]
    fn a_region_with_a_degenerate_duration_never_binds() {
        let regions = [candidate("item", 0., 2., 0., 1.)];
        assert!(unique_geometry_for_region(&region(0., 0., 0., 2.), &regions, None).is_none());
        assert!(
            unique_geometry_for_region(&region(0., f64::NAN, 0., 2.), &regions, None).is_none()
        );
    }
