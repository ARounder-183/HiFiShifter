--[[
F-1：量出 REAPER 的淡化轴映射（形状预设 ↔ 连续 curvature/S 两轴）。

【为什么需要这个脚本】官方头文件（`sdk/reaper_plugin_functions.h`）只给了区间标注：
  `C_FADE*SHAPE` / `D_FADE*DIR`        —— v7.80 and earlier
  `D_FADE*DIR_NEW` / `D_FADE*DIR2_NEW` —— v7.81 and later
它**没有**公开 fade 求值函数，也没有说明 7 个形状预设分别对应哪一组
(curvature, S)。因此在 7.81+ 上"用户选预设"这件事，只能靠实测把表量出来。

量出来之后：
- 若新轴能表达全部 7 个预设 → 把表落成常量，插件在新轴宿主上也能摆预设按钮；
- 若不能 → 维持现状（只给连续滑杆），本脚本的结果就是那条结论的证据。

【怎么跑】与其它 probe 脚本同一纪律：**必须指向隔离的临时工程**，
绝不改动用户工程。设置 `HIFISHIFTER_FADE_AXIS_OUT` 指定输出 JSON 路径。

  1. 在 REAPER 里新建一个空工程，放一条轨道、一个 item（任意音频）；
  2. `Actions → ReaScript → Load`，选本文件并运行；
  3. 读输出 JSON。

  注意：命令行**不能**跑这个脚本（REAPER 7.82 没有运行脚本的开关，已实测；
  见 ../README.md「命令行不能跑探针脚本」）。上面的第 2 步就是唯一的入口。

【为什么读回四个轴 + 版本】三件事要一起看：宿主版本决定了哪套轴是权威的；
写预设后旧轴是否被改写、新轴是否被推导，决定了"写预设"这条路在 7.81+ 上到底
能不能用。只读一半会得出错误结论。
]]

local out_path = assert(
  os.getenv("HIFISHIFTER_FADE_AXIS_OUT"),
  "set HIFISHIFTER_FADE_AXIS_OUT to an output JSON path"
)

-- 官方头文件里的 7 个预设（0=linear）。小数变体（1.1 / 5.1）另算一组，
-- 它们是 REAPER 自己也在用的"等功率 / 锐利 S"编码。
local SHAPES = { 0, 1, 2, 3, 4, 5, 6, 1.1, 5.1 }
local CURVATURES = { -1.0, -0.5, 0.0, 0.5, 1.0 }

local AXES = {
  "C_FADEINSHAPE",
  "D_FADEINDIR",
  "D_FADEINDIR_NEW",
  "D_FADEINDIR2_NEW",
  "D_FADEOUTSHAPE",
  "D_FADEOUTDIR",
  "D_FADEOUTDIR_NEW",
  "D_FADEOUTDIR2_NEW",
}

local function find_item()
  for track_index = 0, reaper.CountTracks(0) - 1 do
    local track = reaper.GetTrack(0, track_index)
    if reaper.CountTrackMediaItems(track) > 0 then
      return reaper.GetTrackMediaItem(track, 0)
    end
  end
  return nil
end

local function read_axes(item)
  local row = {}
  for _, name in ipairs(AXES) do
    row[name] = reaper.GetMediaItemInfo_Value(item, name)
  end
  return row
end

local function write(item, name, value)
  reaper.SetMediaItemInfo_Value(item, name, value)
end

local function encode(row)
  local parts = {}
  for _, name in ipairs(AXES) do
    parts[#parts + 1] = string.format('"%s": %.9f', name, row[name])
  end
  return "{" .. table.concat(parts, ", ") .. "}"
end

local item = find_item()
assert(item, "put one item with audio on a track first (this script never creates one)")

reaper.Undo_BeginBlock()

local version = ({ reaper.GetAppVersion() })[1] or "unknown"
local lines = {
  "{",
  string.format('  "appVersion": "%s",', version:gsub('"', '\\"')),
  '  "samples": [',
}

local first = true
for _, shape in ipairs(SHAPES) do
  for _, dir in ipairs(CURVATURES) do
    -- 先归零两套轴，再写预设：否则上一次的读数会污染这一次。
    write(item, "C_FADEINSHAPE", 0)
    write(item, "D_FADEINDIR", 0)
    write(item, "D_FADEINDIR_NEW", 0)
    write(item, "D_FADEINDIR2_NEW", 0)
    write(item, "D_FADEINLEN", 1.0)
    write(item, "D_FADEINLEN_AUTO", 0)
    local before = read_axes(item)

    write(item, "C_FADEINSHAPE", shape)
    write(item, "D_FADEINDIR", dir)
    local after = read_axes(item)

    if not first then lines[#lines + 1] = "," end
    first = false
    lines[#lines + 1] = string.format(
      '    {"shapeWritten": %.4f, "dirWritten": %.4f, "before": %s, "after": %s}',
      shape, dir, encode(before), encode(after)
    )
  end
end

lines[#lines + 1] = "  ]"
lines[#lines + 1] = "}"

-- 收尾：把轴恢复成归零，不留下一堆试验值给用户。
write(item, "C_FADEINSHAPE", 0)
write(item, "D_FADEINDIR", 0)
write(item, "D_FADEINDIR_NEW", 0)
write(item, "D_FADEINDIR2_NEW", 0)

reaper.Undo_EndBlock("HiFiShifter F-1 fade axis capture", -1)

local file = assert(io.open(out_path, "w"))
file:write(table.concat(lines, "\n"))
file:close()

reaper.ShowConsoleMsg(string.format(
  "F-1 capture written to %s\nhost version: %s\nsamples: %d\n",
  out_path, version, #SHAPES * #CURVATURES
))
