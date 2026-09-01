//! zip 归档写入的共享辅助：保留源文件元数据。
//!
//! 项目的 zip 依赖关闭了 default features（`time` feature 未启用），
//! `FileOptions::default()` 的修改时间会回退为 1980-01-01，导致打包出的
//! 归档丢失原始文件的修改日期。本模块按源文件元数据构造条目选项：
//! - 修改时间（zip 使用 DOS 本地时间，精度 2 秒；超出 1980-2107 范围时
//!   收敛到边界值）；
//! - unix 权限位（仅 unix 平台）。
//! 内存内容的条目使用当前本地时间。

use std::path::Path;

use chrono::{Datelike, Timelike};

/// 为磁盘上的源文件构造条目选项（保留修改时间与 unix 权限位）。
/// 元数据读取失败时回退为当前时间。
pub(crate) fn options_for_source(source: &Path) -> zip::write::FileOptions {
    let options = base_options().last_modified_time(
        std::fs::metadata(source)
            .and_then(|meta| meta.modified())
            .map(datetime_from_system_time)
            .unwrap_or_else(|_| datetime_now()),
    );
    #[cfg(unix)]
    {
        if let Ok(meta) = std::fs::metadata(source) {
            use std::os::unix::fs::PermissionsExt;
            return options.unix_permissions(meta.permissions().mode());
        }
    }
    options
}

/// 为内存内容（如序列化出的 json）构造条目选项，修改时间取当前本地时间。
pub(crate) fn options_now() -> zip::write::FileOptions {
    base_options().last_modified_time(datetime_now())
}

/// 可能为超大文件的条目（如音频媒体）：同 [`options_for_source`]，
/// 并允许 ZIP64（>4GiB 的录音文件不开启此标志会在写入时报错）。
pub(crate) fn options_for_large_source(source: &Path) -> zip::write::FileOptions {
    options_for_source(source).large_file(true)
}

fn base_options() -> zip::write::FileOptions {
    zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated)
}

fn datetime_now() -> zip::DateTime {
    let dt = chrono::Local::now();
    datetime_from_parts(
        dt.year(),
        dt.month() as u8,
        dt.day() as u8,
        dt.hour() as u8,
        dt.minute() as u8,
        dt.second() as u8,
    )
}

fn datetime_from_system_time(t: std::time::SystemTime) -> zip::DateTime {
    let dt: chrono::DateTime<chrono::Local> = t.into();
    datetime_from_parts(
        dt.year(),
        dt.month() as u8,
        dt.day() as u8,
        dt.hour() as u8,
        dt.minute() as u8,
        dt.second() as u8,
    )
}

/// 打包为 MS-DOS 日期/时间位域（与 `zip::DateTime::datepart()` / `timepart()`
/// 的布局互逆），再经 `DateTime::from_msdos` 构造。
/// DOS 时间精度为 2 秒（秒的低 bit 被丢弃）；年份范围 1980-2107，越界收敛。
fn datetime_from_parts(
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
) -> zip::DateTime {
    let year = year.clamp(1980, 2107) as u16;
    let second = second.min(59);
    let datepart = (day as u16) | ((month as u16) << 5) | ((year - 1980) << 9);
    let timepart = ((second / 2) as u16) | ((minute as u16) << 5) | ((hour as u16) << 11);
    zip::DateTime::from_msdos(datepart, timepart)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use std::time::{Duration, SystemTime};

    /// 1980-01-01 00:00:00 的 DOS 位域：datepart = 1 | (1<<5) | (0<<9) = 33，
    /// timepart = 0。
    const DOS_EPOCH_DATEPART: u16 = 33;
    const DOS_EPOCH_TIMEPART: u16 = 0;

    /// 2107-12-31 23:59:58 的 DOS 位域（DOS 上限，秒取 58）：
    /// datepart = 31 | (12<<5) | (127<<9) = 65439，
    /// timepart = 29 | (59<<5) | (23<<11) = 49021。
    const DOS_MAX_DATEPART: u16 = 65439;
    const DOS_MAX_TIMEPART: u16 = 49021;

    fn system_time_from_utc_seconds(secs: i64) -> SystemTime {
        use chrono::TimeZone;
        chrono::Utc
            .timestamp_opt(secs, 0)
            .single()
            .expect("valid timestamp")
            .into()
    }

    #[test]
    fn datetime_clamps_year_to_dos_lower_bound() {
        // 1970-01-02（任何时区的本地年份都是 1970，早于 DOS 下限 1980）
        // → 年份收敛为 1980，月/日/时间分量保留本地值。
        let t: SystemTime = chrono::Utc
            .with_ymd_and_hms(1970, 1, 2, 0, 0, 0)
            .single()
            .expect("valid timestamp")
            .into();
        let dt = datetime_from_system_time(t);
        assert_eq!(dt.datepart() >> 9, 0); // year == 1980
        let local: chrono::DateTime<chrono::Local> = t.into();
        let expected_timepart = ((local.second() as u16 / 2)
            | ((local.minute() as u16) << 5)
            | ((local.hour() as u16) << 11));
        assert_eq!(dt.timepart(), expected_timepart);
    }

    #[test]
    fn datetime_clamps_year_to_dos_upper_bound() {
        // 2108-06-15 12:00Z（任何时区的本地年份都是 2108，晚于 DOS 上限 2107）
        // → 年份收敛为 2107。
        let t: SystemTime = chrono::Utc
            .with_ymd_and_hms(2108, 6, 15, 12, 0, 0)
            .single()
            .expect("valid timestamp")
            .into();
        let dt = datetime_from_system_time(t);
        assert_eq!((dt.datepart() >> 9) + 1980, 2107);
    }

    #[test]
    fn datetime_roundtrips_recent_time_within_two_seconds() {
        // 普通近期时间：DOS 精度 2 秒，回读分量误差不超过 2 秒（本地时区）。
        let mtime = system_time_from_utc_seconds(1_756_600_000);
        let dt = datetime_from_system_time(mtime);
        let local: chrono::DateTime<chrono::Local> = mtime.into();
        let expected_datepart = (local.day() as u16)
            | ((local.month() as u16) << 5)
            | (((local.year() as i64 - 1980) as u16) << 9);
        assert_eq!(dt.datepart(), expected_datepart);
        // 秒按 2 秒量化。
        let quantized_second = (local.second() as u16 / 2) * 2;
        let expected_timepart = (quantized_second / 2)
            | ((local.minute() as u16) << 5)
            | ((local.hour() as u16) << 11);
        assert_eq!(dt.timepart(), expected_timepart);
    }

    #[test]
    fn options_preserve_source_mtime_end_to_end() {
        // 端到端：源文件设置 26 小时前的修改时间 → 打包 → 回读 ZipArchive，
        // 条目的 DOS 时间位域应与源文件 mtime 一致（修复前恒为 1980-01-01）。
        let dir = std::env::temp_dir().join(format!(
            "hifishifter_zip_util_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let source = dir.join("source.bin");
        std::fs::write(&source, b"payload").expect("write source");

        let mtime = SystemTime::now() - Duration::from_secs(26 * 3600);
        {
            let f = std::fs::File::options()
                .write(true)
                .open(&source)
                .expect("open for set_modified");
            f.set_modified(mtime).expect("set_modified");
        }

        let zip_path = dir.join("out.zip");
        {
            let f = std::fs::File::create(&zip_path).expect("create zip");
            let mut zip = zip::ZipWriter::new(f);
            zip.start_file("source.bin", options_for_source(&source))
                .expect("start_file");
            std::io::copy(&mut std::fs::File::open(&source).unwrap(), &mut zip)
                .expect("copy");
            zip.finish().expect("finish");
        }

        let f = std::fs::File::open(&zip_path).expect("open zip");
        let mut archive = zip::ZipArchive::new(f).expect("archive");
        let entry = archive.by_index(0).expect("entry");
        let expected = datetime_from_system_time(mtime);
        assert_eq!(entry.last_modified().datepart(), expected.datepart());
        assert_eq!(entry.last_modified().timepart(), expected.timepart());
        // 修复前的默认值是 1980-01-01（33, 0），不允许再出现。
        assert_ne!(
            (entry.last_modified().datepart(), entry.last_modified().timepart()),
            (DOS_EPOCH_DATEPART, DOS_EPOCH_TIMEPART)
        );

        drop(entry);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
