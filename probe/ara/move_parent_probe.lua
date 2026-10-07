--[[ 中文一次性实际工程副本复现：只操作已提取的四轨副本，不保存或修改原工程/素材。 ]]
local scratch=assert(os.getenv('HIFISHIFTER_MANY_CLIPS_DIR'))
local _,project=reaper.EnumProjects(-1,'')
local function norm(path) return path:gsub('/','\\'):lower() end
assert(norm(project)==norm(scratch..'\\many-clips.RPP'),'wrong project')
assert(reaper.CountTracks(0)==4,'only parent and three children expected')
local parent=reaper.GetTrack(0,0)
local function log(text) local f=assert(io.open(scratch..'\\probe.log','a'));f:write(text..'\n');f:close() end
local total=0
for i=1,3 do total=total+reaper.CountTrackMediaItems(reaper.GetTrack(0,i)) end
log('before add parent_items='..reaper.CountTrackMediaItems(parent)..' children_items='..total)
assert(reaper.CountTrackMediaItems(parent)==0,'expected empty parent')
local fx=reaper.TrackFX_AddByName(parent,'HiFiShifter',false,-1)
log('after add fx='..fx);assert(fx>=0)
reaper.Undo_BeginBlock2(0)
local moved=0
for i=1,3 do
  local child=reaper.GetTrack(0,i)
  while reaper.CountTrackMediaItems(child)>0 do
    log('before move '..(moved+1))
    assert(reaper.MoveMediaItemToTrack(reaper.GetTrackMediaItem(child,0),parent))
    moved=moved+1
  end
end
reaper.Undo_EndBlock2(0,'owned parent move reproduction',-1)
log('all moved='..moved..' parent_items='..reaper.CountTrackMediaItems(parent))
reaper.Main_SaveProject(0,false)
local quit
for i=0,20000 do local id,name=reaper.kbd_enumerateActions(reaper.SectionFromUniqueID(0),i);if id==0 then break end
  if name and (name:find('Quit REAPER',1,true) or name=='File: Quit') then quit=id end end
assert(quit)
local finish=reaper.time_precise()+5
local function tick() if reaper.time_precise()<finish then reaper.defer(tick);return end
  log('completed normal quit requested');reaper.Main_SaveProject(0,false);reaper.Main_OnCommand(quit,0) end
reaper.defer(tick)
