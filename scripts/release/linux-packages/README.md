# Linux 原生包（deb / rpm）

Linux 主安装路径：用 `nfpm` 把一次构建的产物打成四个包，两种格式各一份。

| 包                   | 内容                                                                                                                                                | 依赖                                                        |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `hifishifter`        | `/usr/bin/hifishifter`、`/usr/lib/HiFiShifter/`（`models.json`、许可、`libSoundTouchDLL.so`、x86_64 的 `libwebgpu_dawn.so`）、desktop / 图标 / MIME | `hifishifter-models (= <manifest version>)`，推荐两个插件包 |
| `hifishifter-models` | `/usr/share/HiFiShifter/models/<manifest version>/` 全量权重                                                                                        | 无；`Architecture: all` / `noarch`                          |
| `hifishifter-vst3`   | `/usr/lib/vst3/HiFiShifter.vst3/`（bundle 内只留 `models.json`）                                                                                    | `hifishifter-models`                                        |
| `hifishifter-clap`   | `/usr/lib/clap/HiFiShifter.clap` + 同级 `HiFiShifter-resources/`                                                                                    | `hifishifter-models`                                        |

## 为什么是 nfpm

一份 YAML 同时产出 `.deb` 与 `.rpm`，包括**在 x86_64 runner 上产 aarch64 的 rpm** —— 用
`rpmbuild` 就得准备目标架构的工具链。这里不编译任何东西：`build.sh` 只把准备好的树映射到 FHS
布局，再交给 nfpm。

## 用法

```bash
scripts/release/linux-packages/build.sh \
  --stage dist/stage-linux-packages \
  --out dist/HiFiShifter_v0.2.0-beta_linux-x86_64-packages \
  --arch amd64 --version 0.2.0-beta
```

`--stage` 的目录结构由 `scripts/build/unix/package.sh` 组装，见 `build.sh` 文件头。常用开关：

- `--only=app,models,vst3,clap`：只出其中几个包（本地改一个格式时用）。
- `--nfpm=PATH`：用指定的 nfpm；否则用 `PATH` 上的，再否则下载脚里固定的版本。

## 版本映射

| 用途                    | 值                                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| 产品版本                | `0.2.0-beta`（文件与界面）                                                                |
| deb `Version`           | `0.2.0~beta`（`~` 让预发布排在正式版之前）                                                |
| rpm `Version`-`Release` | `0.2.0`-`0.1.beta`                                                                        |
| 模型包版本              | `models.json` 的 `version`（内容派生，不手写）                                            |
| 文件名                  | `hifishifter_<deb version>_<arch>.deb` / `hifishifter-<rpm version>-<release>.<arch>.rpm` |

## 模型包为什么是 `all` / `noarch`

`models.json` 由内容派生：它列出每个文件的 `sha256` 与字节数。`build.sh` 在打包前逐件校验
staged 目录，校验通过的树与**任何**另一棵校验通过的树逐字节相同 —— 所以权重不可能随架构变化，
`Architecture: all` 是事实而不是假设。校验失败直接失败，不会打出一个"看起来正常"的包。

## 用户数据

包里没有任何用户数据路径：设置（`~/.config/HiFiShifter`）、本地数据
（`~/.local/share/HiFiShifter`，含 `models-user`）、工程与音频文件都不由包管理器管理，
`apt purge` / `dnf remove` 也不会碰它们。

## 不在本目录范围

GPG 签名、厂商仓库、自动更新、AUR —— 都需要独立的发布流程（密钥、仓库托管、更新元数据），
不在这套打包脚本的范围内。当前发布方式是把 `-packages/` 目录整体作为 Release 附件上传。
