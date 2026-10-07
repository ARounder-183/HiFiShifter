--[[ 中文一次性多短clip复现：仅隔离空工程，先放素材再加ARA插件，不编辑用户工程。 ]]
local scratch=assert(os.getenv('HIFISHIFTER_MANY_CLIPS_DIR'),'private scratch required')
local function norm(path) return path:gsub('/','\\'):gsub('\\+$',''):lower() end
local _,project=reaper.EnumProjects(-1,'')
local startup=assert(io.open(scratch..'\\probe.log','a'))
startup:write('startup resource='..reaper.GetResourcePath()..' project='..project..' tracks='..reaper.CountTracks(0)..'\n');startup:close()
assert(project=='' and reaper.CountTracks(0)==0,'not an empty disposable project')
assert(norm(reaper.GetResourcePath())==norm(scratch..'\\profile'),'not the private profile')
local function log(line)
  local file=assert(io.open(scratch..'\\probe.log','a'));file:write(line..'\n');file:close()
end
local quit
for index=0,20000 do
  local id,name=reaper.kbd_enumerateActions(reaper.SectionFromUniqueID(0),index)
  if id==0 then break end
  if name and (name:find('Quit REAPER',1,true) or name=='File: Quit') then quit=id end
end
assert(quit,'normal quit action not found')
local count=tonumber(os.getenv('HIFISHIFTER_MANY_CLIPS_COUNT')) or 100
local rate=44100
local channels=os.getenv('HIFISHIFTER_MANY_STEREO')=='1' and 2 or 1
local unicode=os.getenv('HIFISHIFTER_MANY_UNICODE')=='1'
local samples={}
for i=0,8820-1 do for channel=1,channels do samples[#samples+1]=string.pack('<f',0.2*math.sin(2*math.pi*(220+channel-1)*i/rate)) end end
local pcm=table.concat(samples)
reaper.InsertTrackAtIndex(0,true)
local track=reaper.GetTrack(0,0);reaper.SetOnlyTrackSelected(track)
reaper.GetSetMediaTrackInfo_String(track,'P_NAME','many short clips - owned reproduction',true)
local empty_folder=os.getenv('HIFISHIFTER_MANY_EMPTY_FOLDER')=='1'
local children={}
if empty_folder then
  reaper.SetMediaTrackInfo_Value(track,'I_FOLDERDEPTH',1)
  for index=1,3 do reaper.InsertTrackAtIndex(index,true);children[index]=reaper.GetTrack(0,index) end
  reaper.SetMediaTrackInfo_Value(children[3],'I_FOLDERDEPTH',-1)
end
reaper.PreventUIRefresh(1)
for index=1,count do
  local path=scratch..'\\'..(unicode and '短素材-' or 'source-')..index..'.wav'
  local file=assert(io.open(path,'wb'))
  file:write('RIFF',string.pack('<I4',36+#pcm),'WAVEfmt ',string.pack('<I4I2I2I4I4I2I2',16,3,channels,rate,rate*channels*4,channels*4,32),'data',string.pack('<I4',#pcm),pcm);file:close()
  if empty_folder then reaper.SetOnlyTrackSelected(children[1+(index-1)%3]) end
  reaper.SetEditCurPos((index-1)*0.25,false,false);reaper.InsertMedia(path,0)
end
reaper.PreventUIRefresh(-1)
reaper.Main_SaveProjectEx(0,scratch..'\\many-clips.RPP',8)
log('before add fx items='..reaper.CountTrackMediaItems(track))
reaper.SetOnlyTrackSelected(track)
local fx=reaper.TrackFX_AddByName(track,'HiFiShifter',false,-1)
log('after add fx='..fx..' items='..reaper.CountTrackMediaItems(track))
assert(fx>=0,'new plugin not discovered')
if empty_folder then
  log('moving children to initialized empty parent')
  reaper.Undo_BeginBlock2(0)
  local moved=0
  for _,child in ipairs(children) do
    while reaper.CountTrackMediaItems(child)>0 do
      assert(reaper.MoveMediaItemToTrack(reaper.GetTrackMediaItem(child,0),track))
      moved=moved+1
      if moved==1 or moved%16==0 then log('moved='..moved) end
    end
  end
  reaper.Undo_EndBlock2(0,'owned move reproduction',-1)
  log('all moved='..moved..' parent items='..reaper.CountTrackMediaItems(track))
end
reaper.Main_SaveProject(0,false)
local finish=reaper.time_precise()+5
local function tick()
  if reaper.time_precise()<finish then reaper.defer(tick);return end
  log('completed normal quit requested');reaper.Main_SaveProject(0,false);reaper.Main_OnCommand(quit,0)
end
reaper.defer(tick)
