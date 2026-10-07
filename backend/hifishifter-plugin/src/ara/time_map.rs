//! 已确认秒域的项目↔源时间映射；原生REAPER raw marker必须先验证单位后才能进入这里。
//! 本模块为曲线源坐标迁移提供同一映射，不对PCM做普通重采样，也不假称完整marker渲染。

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TimeAnchor {
    pub project_sec: f64,
    pub source_sec: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct SourceTimeMap {
    points: Vec<TimeAnchor>,
}

impl SourceTimeMap {
    /// 从真实ARA的两个正向时间范围构造仿射映射；单位明确为秒，反向/冻结另有契约。
    pub fn linear(
        project_start: f64,
        project_duration: f64,
        source_start: f64,
        source_duration: f64,
    ) -> Result<Self, String> {
        Self::piecewise(vec![
            TimeAnchor {
                project_sec: project_start,
                source_sec: source_start,
            },
            TimeAnchor {
                project_sec: project_start + project_duration,
                source_sec: source_start + source_duration,
            },
        ])
    }

    /// 接收经过单位确认的严格递增锚点；不排序/钳制损坏数据来伪造有效映射。
    pub fn piecewise(points: Vec<TimeAnchor>) -> Result<Self, String> {
        if !(2..=65_536).contains(&points.len()) {
            return Err("source time map requires 2..65536 anchors".into());
        }
        if points
            .iter()
            .any(|point| !point.project_sec.is_finite() || !point.source_sec.is_finite())
        {
            return Err("non-finite source time anchor".into());
        }
        for pair in points.windows(2) {
            let project_span = pair[1].project_sec - pair[0].project_sec;
            let source_span = pair[1].source_sec - pair[0].source_sec;
            if !project_span.is_finite()
                || !source_span.is_finite()
                || project_span <= 0.0
                || source_span <= 0.0
            {
                return Err("source time map must be strictly forward and finite".into());
            }
        }
        Ok(Self { points })
    }

    /// 闭区间端点可映射，区间外不外推；渲染消费者仍以半开clip区间决定是否供音。
    pub fn project_to_source(&self, project_sec: f64) -> Option<f64> {
        self.interpolate(
            project_sec,
            |point| point.project_sec,
            |point| point.source_sec,
        )
    }
    /// 源坐标反求项目位置；曲线裁切只能继承已有源窗口，不能延展旧末值到新音频。
    pub fn source_to_project(&self, source_sec: f64) -> Option<f64> {
        self.interpolate(
            source_sec,
            |point| point.source_sec,
            |point| point.project_sec,
        )
    }
    /// 以同一源位置把新项目时刻移回旧曲线坐标，不按项目位置/源文件名猜身份。
    pub fn previous_project_time(&self, previous: &Self, project_sec: f64) -> Option<f64> {
        previous.source_to_project(self.project_to_source(project_sec)?)
    }

    /// 两方向共享严格单调的区间查询，保证正反映射使用相同锚点而不是两套倍率公式。
    fn interpolate(
        &self,
        value: f64,
        input: impl Fn(&TimeAnchor) -> f64,
        output: impl Fn(&TimeAnchor) -> f64,
    ) -> Option<f64> {
        if !value.is_finite()
            || value < input(&self.points[0])
            || value > input(self.points.last()?)
        {
            return None;
        }
        let next = self
            .points
            .partition_point(|point| input(point) <= value)
            .clamp(1, self.points.len() - 1);
        let lo = &self.points[next - 1];
        let hi = &self.points[next];
        let fraction = (value - input(lo)) / (input(hi) - input(lo));
        let result = output(lo) + fraction * (output(hi) - output(lo));
        result.is_finite().then_some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(actual: Option<f64>, expected: f64) {
        assert!((actual.unwrap() - expected).abs() < 1e-12);
    }

    /// source0..2秒拉伸到project1..5秒，项目中点仍对应源中点，逆映射同样准确。
    #[test]
    fn forward_stretch_has_source_authority_and_an_exact_inverse() {
        let map = SourceTimeMap::linear(1.0, 4.0, 0.0, 2.0).unwrap();
        for (project, source) in [(1.0, 0.0), (2.0, 0.5), (3.0, 1.0), (4.0, 1.5), (5.0, 2.0)] {
            close(map.project_to_source(project), source);
            close(map.source_to_project(source), project);
        }
        assert_eq!(map.project_to_source(0.99), None);
        assert_eq!(map.project_to_source(5.01), None);
        assert_eq!(map.source_to_project(-0.01), None);
        assert_eq!(map.source_to_project(2.01), None);
    }

    /// 移动/裁切/拉伸/拆分后按源位置继承原曲线，不保留旧的项目帧位置。
    #[test]
    fn movement_crop_and_split_map_to_previous_curve_coordinates() {
        let old = SourceTimeMap::linear(1.0, 2.0, 0.0, 2.0).unwrap();
        let moved = SourceTimeMap::linear(4.0, 2.0, 0.0, 2.0).unwrap();
        close(moved.previous_project_time(&old, 4.5), 1.5);
        let cropped = SourceTimeMap::linear(2.0, 2.0, 1.0, 1.0).unwrap();
        close(cropped.previous_project_time(&old, 2.0), 2.0);
        close(cropped.previous_project_time(&old, 3.0), 2.5);
        close(cropped.previous_project_time(&old, 4.0), 3.0);
        let right = SourceTimeMap::linear(3.0, 1.0, 1.0, 1.0).unwrap();
        close(right.previous_project_time(&old, 3.5), 2.5);
        let extended = SourceTimeMap::linear(0.0, 4.0, 0.0, 4.0).unwrap();
        assert_eq!(
            extended.previous_project_time(&old, 3.0),
            None,
            "新源范围不能继承旧末值"
        );
    }

    /// 分段倍率不是总时长比：前段0.25、后段1.5，内部锚点必须参与映射。
    #[test]
    fn verified_piecewise_anchors_preserve_each_local_rate() {
        let map = SourceTimeMap::piecewise(vec![
            TimeAnchor {
                project_sec: 1.0,
                source_sec: 0.0,
            },
            TimeAnchor {
                project_sec: 3.0,
                source_sec: 0.5,
            },
            TimeAnchor {
                project_sec: 4.0,
                source_sec: 2.0,
            },
        ])
        .unwrap();
        close(map.project_to_source(2.0), 0.25);
        close(map.project_to_source(3.5), 1.25);
        close(map.source_to_project(0.25), 2.0);
        close(map.source_to_project(1.25), 3.5);
        close(map.source_to_project(1.0), 10.0 / 3.0);
        close(map.project_to_source(3.0), 0.5);
    }

    /// 正向域必须严格单调；非法/重复锚点不能被悄悄排序或当反向播放。
    #[test]
    fn malformed_and_reverse_maps_fail_without_clamping_or_sorting() {
        for points in [
            vec![],
            vec![TimeAnchor {
                project_sec: 0.0,
                source_sec: 0.0,
            }],
            vec![
                TimeAnchor {
                    project_sec: 0.0,
                    source_sec: 0.0,
                },
                TimeAnchor {
                    project_sec: 0.0,
                    source_sec: 1.0,
                },
            ],
            vec![
                TimeAnchor {
                    project_sec: 0.0,
                    source_sec: 1.0,
                },
                TimeAnchor {
                    project_sec: 1.0,
                    source_sec: 0.0,
                },
            ],
            vec![
                TimeAnchor {
                    project_sec: 1.0,
                    source_sec: 0.0,
                },
                TimeAnchor {
                    project_sec: 0.0,
                    source_sec: 1.0,
                },
            ],
            vec![
                TimeAnchor {
                    project_sec: 0.0,
                    source_sec: 0.0,
                },
                TimeAnchor {
                    project_sec: f64::NAN,
                    source_sec: 1.0,
                },
            ],
            vec![
                TimeAnchor {
                    project_sec: 0.0,
                    source_sec: 0.0,
                },
                TimeAnchor {
                    project_sec: 1.0,
                    source_sec: f64::INFINITY,
                },
            ],
        ] {
            assert!(SourceTimeMap::piecewise(points).is_err());
        }
        for duration in [0.0, -1.0, f64::INFINITY] {
            assert!(SourceTimeMap::linear(0.0, duration, 0.0, 1.0).is_err());
        }
        let map = SourceTimeMap::linear(0.0, 1.0, 0.0, 1.0).unwrap();
        assert_eq!(map.project_to_source(f64::NAN), None);
        assert_eq!(map.source_to_project(f64::INFINITY), None);
    }
}
