--[[
F-1：量出 REAPER 的淡化轴映射（形状预设 ↔ 连续 curvature/S 两轴）。

【为什么需要这个脚本】官方头文件（`sdk/reaper_plugin_functions.h`）只给了区间标注：
  `C_FADE*SHAPE`（int，0..6，0=linear）      —— v7.80 and earlier
  `D_FADE*DIR`  （curvature，-1..1）          —— v7.80 and earlier
  `D_FADE*DIR_NEW`  （curvature，-1..1）      —— v7.81 and later
  `D_FADE*DIR2_NEW` （S 参数，-1..1）         —— v7.81 and later
头文件明说 v7.81+ 由 `DIR_NEW`/`DIR2_NEW` 决定形状，**但没给** 7 个预设各自对应哪一组
(curvature, S)。所以在 7.81+ 上"用户选预设"这件事只能实测：
- 若"写预设 → 新轴被推导出"成立 → 把表落成常量，插件在新轴宿主上也能摆预设按钮；
- 若新轴根本不动 → 预设写入在新宿主上是死路，插件只能给连续滑杆，这就是那条结论的证据。

【为什么每个用例都要新建 item】item 的淡化轴有"最后写入哪套、哪套是权威"的耦合
（实测：写新轴后读 `C_FADEINSHAPE` 会得到 -1）。在同一个 item 上连续试验会让上一次的
写入污染下一次，所以每个用例都从一个**全新 item** 起测。

【轴名的坑】`GetMediaItemInfo_Value` 对**不认识的键返回 0.0，不报错**。所以轴名写错一个
字母，读数会是一串干净的 0，看起来像"宿主不支持"。改轴名时务必和
`reaper_plugin_functions.h` 里的拼写逐字对照（本文件曾把 `C_FADEOUTSHAPE` 写成
`D_FADEOUTSHAPE`，fade-out 那四列因此整列假 0）。

【怎么跑】
  1. 在**空工程**里跑（脚本自建夹具轨，跑完删掉，不留痕迹）；
  2. 设置 `HIFISHIFTER_FADE_AXIS_OUT` 指定输出 JSON 路径；
  3. 入口：REAPER 的 Actions → Show action list → Load… 选本文件 → Run。
     自动化入口见 ../README.md「无头跑探针（__startup.eel）」。
]]

local out_path = assert(
  os.getenv("HIFISHIFTER_FADE_AXIS_OUT"),
  "set HIFISHIFTER_FADE_AXIS_OUT to an output JSON path"
)

-- 头文件里的 7 个预设（0=linear）。小数变体（1.1 / 5.1）另算一组，
-- 它们是 REAPER 自己也在用的"等功率 / 锐利 S"编码。
local SHAPES = { 0, 1, 2, 3, 4, 5, 6, 1.1, 5.1 }
local CURVATURES = { -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0 }
local PAIRS = {
  { 0.0, 0.0 }, { -1.0, 0.0 }, { 1.0, 0.0 },
  { 0.0, -1.0 }, { 0.0, 1.0 }, { -1.0, -1.0 }, { 1.0, 1.0 },
  { -0.5, 0.5 }, { 0.5, -0.5 }, { -1.0, 1.0 }, { 1.0, -1.0 },
}

local AXES = {
  "C_FADEINSHAPE",
  "D_FADEINDIR",
  "D_FADEINDIR_NEW",
  "D_FADEINDIR2_NEW",
  "C_FADEOUTSHAPE",
  "D_FADEOUTDIR",
  "D_FADEOUTDIR_NEW",
  "D_FADEOUTDIR2_NEW",
}

-- 夹具 item 的淡化长度：形状只有在"真的存在一段淡化"时才有意义。
local FIXTURE_FADE_LEN = 0.25

local function read_axes(item)
  local row = {}
  for _, name in ipairs(AXES) do
    row[name] = reaper.GetMediaItemInfo_Value(item, name)
  end
  return row
end

local function encode(row)
  local parts = {}
  for _, name in ipairs(AXES) do
    parts[#parts + 1] = string.format('"%s": %.9f', name, row[name])
  end
  return "{" .. table.concat(parts, ", ") .. "}"
end

local function set(item, name, value)
  reaper.SetMediaItemInfo_Value(item, name, value)
end

-- 用例：名称、说明、写入函数、取值列表、写值标签格式化
local function legacy_shape(item, v)
  set(item, "C_FADEINSHAPE", v)
end
local function legacy_dir(item, v)
  set(item, "D_FADEINDIR", v)
end
local function new_dir(item, v)
  set(item, "D_FADEINDIR_NEW", v)
end
local function new_s(item, v)
  set(item, "D_FADEINDIR2_NEW", v)
end
local function new_pair(item, v)
  set(item, "D_FADEINDIR_NEW", v[1])
  set(item, "D_FADEINDIR2_NEW", v[2])
end
local function legacy_then_new(item, v)
  set(item, "C_FADEINSHAPE", v[1])
  set(item, "D_FADEINDIR_NEW", v[2])
end

-- 数值标签一律用 %.4g：预设里有 1.1 / 5.1 这种小数，`%d` 在 Lua 里会直接报错。
local CASES = {
  { "legacy_shape_only", "只写 C_FADEINSHAPE（v7.80 的预设轴）", legacy_shape, SHAPES,
    function(v) return string.format("%.4g", v) end },
  { "legacy_dir_only", "只写 D_FADEINDIR（v7.80 的曲率轴）", legacy_dir, CURVATURES,
    function(v) return string.format("%.4g", v) end },
  { "new_dir_only", "只写 D_FADEINDIR_NEW（v7.81 的曲率轴）", new_dir, CURVATURES,
    function(v) return string.format("%.4g", v) end },
  { "new_s_only", "只写 D_FADEINDIR2_NEW（v7.81 的 S 轴）", new_s, CURVATURES,
    function(v) return string.format("%.4g", v) end },
  { "new_pair_only", "只写 (D_FADEINDIR_NEW, D_FADEINDIR2_NEW)", new_pair, PAIRS,
    function(v) return string.format("%.4g/%.4g", v[1], v[2]) end },
  { "legacy_shape_then_new_dir", "先写预设再写新曲率（验证最后写入者权威）", legacy_then_new,
    { { 0, 0.5 }, { 3, 0.5 }, { 5, -0.5 } },
    function(v) return string.format("%.4g/%.4g", v[1], v[2]) end },
}

local function measure()
  -- 夹具轨：建在工程末尾，跑完删掉。用 MIDI item（自足，不依赖外部音频文件）。
  reaper.Undo_BeginBlock()
  reaper.InsertTrackAtIndex(reaper.CountTracks(0), true)
  local track = reaper.GetTrack(0, reaper.CountTracks(0) - 1)
  reaper.SetOnlyTrackSelected(track)

  local function fresh_item()
    local item = reaper.CreateNewMIDIItemInProj(track, 0.0, 1.0, false)
    set(item, "D_FADEINLEN", FIXTURE_FADE_LEN)
    set(item, "D_FADEOUTLEN", FIXTURE_FADE_LEN)
    return item
  end

  local version = ({ reaper.GetAppVersion() })[1] or "unknown"
  local lines = {
    "{",
    string.format('  "appVersion": "%s",', version:gsub('"', '\\"')),
    string.format('  "fixtureFadeLen": %.9f,', FIXTURE_FADE_LEN),
  }

  -- 基线：全新 item 只设了淡化长度，没碰任何形状轴。
  local baseline_item = fresh_item()
  lines[#lines + 1] = string.format('  "baseline": %s,', encode(read_axes(baseline_item)))
  reaper.DeleteTrackMediaItem(track, baseline_item)

  lines[#lines + 1] = '  "cases": ['
  local first_case = true
  for _, case in ipairs(CASES) do
    local name, note, apply, values, label = case[1], case[2], case[3], case[4], case[5]
    local samples = {}
    for _, value in ipairs(values) do
      local item = fresh_item()
      local before = read_axes(item)
      apply(item, value)
      local after = read_axes(item)
      samples[#samples + 1] = string.format(
        '      {"written": "%s", "before": %s, "after": %s}',
        label(value), encode(before), encode(after)
      )
      reaper.DeleteTrackMediaItem(track, item)
    end
    if not first_case then lines[#lines + 1] = "," end
    first_case = false
    lines[#lines + 1] = string.format(
      '    {"case": "%s", "note": "%s", "samples": [\n%s\n    ]}',
      name, note, table.concat(samples, ",\n")
    )
  end
  lines[#lines + 1] = "  ]"
  lines[#lines + 1] = "}"

  -- 收尾：删掉夹具轨，工程回到跑之前的样子。
  reaper.DeleteTrack(track)
  reaper.Undo_EndBlock("HiFiShifter F-1 fade axis capture", -1)

  return table.concat(lines, "\n"), version
end

-- 无头跑时没人看控制台，失败必须落进输出文件，否则就是"静默无输出"。
local ok, result, version = pcall(measure)
local payload
if ok then
  payload = result
else
  payload = string.format(
    '{\n  "appVersion": "%s",\n  "error": "%s"\n}',
    tostring(({ reaper.GetAppVersion() })[1] or "unknown"):gsub('"', '\\"'),
    tostring(result):gsub("[\r\n]", " "):gsub('"', '\\"')
  )
end

local file = assert(io.open(out_path, "w"))
file:write(payload)
file:close()

reaper.ShowConsoleMsg(string.format(
  "F-1 capture written to %s\nhost version: %s\nstatus: %s\n",
  out_path, tostring(version or "unknown"), ok and "ok" or "error"
))
