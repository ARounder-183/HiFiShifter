"""OpenVPI/OpenUtau ONNX voicebank adapter. No training or arbitrary bank code.

The editor sends seconds (already converted through its tempo map). A private
worker reuses one bank; cancellation kills it and idle time releases models.
See docs/diffsinger.md for the supported bank contract and upstream references.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "packages"))
# A dedicated wheel directory avoids mixing CPU and DirectML distributions.
if (Path(__file__).parent / 'gpu-packages' / 'onnxruntime').is_dir():
    sys.path.insert(0, str(Path(__file__).parent / 'gpu-packages'))
import numpy as np
import yaml
import expressions
import compatibility
import runtime
import timing
import pronunciation
import inference
import variance_retake


def read_yaml(path):
    # Large multilingual dictionaries benefit from LibYAML; retain the safe
    # Python loader as a fallback for runtimes without the optional C module.
    data = yaml.load(Path(path).read_text(encoding="utf-8-sig"),
                     Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    if not isinstance(data, dict):
        raise ValueError(f"配置不是 YAML 对象：{path}")
    return data


def resolve(root, relative):
    if not relative:
        raise ValueError(f"模型配置缺少文件路径：{root}")
    return (Path(root) / str(relative)).resolve()


def tokens_for(root, config):
    path = resolve(root, config.get("phonemes", "phonemes.txt"))
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return {p.strip(): i for i, p in enumerate(path.read_text(encoding="utf-8-sig").splitlines())}


class Model:
    def __init__(self, root, execution=None):
        self.root = Path(root)
        self.config = read_yaml(self.root / "dsconfig.yaml")
        self.tokens = tokens_for(self.root, self.config)
        self.languages = (json.loads(resolve(self.root, self.config["languages"]).read_text(encoding="utf-8-sig"))
                          if self.config.get("use_lang_id") else {})
        self.sessions = {}
        self.speaker = ""
        self.expression_curves = None
        self.execution = execution or inference.Context()

    @property
    def dt(self):
        return self.config.get("hop_size", 512) / self.config.get("sample_rate", 44100)

    def session(self, name):
        if name not in self.sessions:
            self.sessions[name] = self.execution.session(resolve(self.root, self.config[name]), self.root.name + "/" + name)
        return self.sessions[name]

    def run(self, name, values, frames=None):
        session = self.session(name)
        values = dict(values)
        execution = getattr(self, 'execution', None)
        settings = execution.options if execution else inference.options()
        sampling = compatibility.sampling_inputs(self.config, session.get_inputs(),
            settings.get(name + '_steps', 20), settings['depth'] if name == 'acoustic' else 1.0)
        values.update(sampling)
        if execution is not None and sampling:
            execution.sampling[name] = sampling
        for item in session.get_inputs():
            if item.name == "spk_embed":
                speakers = self.config.get("speakers", [])
                speaker = self.speaker if self.speaker in speakers else next(iter(speakers), "")
                if not speaker:
                    raise ValueError(f"{self.root}: 模型需要说话人 embedding，但配置没有 speakers")
                emb = np.fromfile(resolve(self.root, speaker + ".emb"), dtype="<f4")
                hidden = self.config.get("hidden_size", 256)
                if emb.size != hidden:
                    raise ValueError(f"说话人 embedding 长度错误：{speaker}")
                if frames is None:
                    raise ValueError("内部错误：缺少 embedding 时间维度")
                values["spk_embed"] = np.broadcast_to(emb, (1, frames, hidden)).copy()
                if self.expression_curves is not None and name in ("acoustic", "pitch", "variance"):
                    weights = expressions.speaker_weights(self.expression_curves, speakers, speaker)
                    if weights.shape[0] != frames:
                        raise ValueError("音色曲线的帧数与模型不匹配")
                    embeddings = [np.fromfile(resolve(self.root, s + ".emb"), dtype="<f4") for s in speakers]
                    if any(e.size != hidden for e in embeddings):
                        raise ValueError("混合音色 embedding 长度错误")
                    values["spk_embed"] = (weights @ np.stack(embeddings))[None].astype(np.float32)
        feed = {}
        types = {"tensor(float)": np.float32, "tensor(int64)": np.int64,
                 "tensor(bool)": np.bool_, "tensor(double)": np.float64}
        for item in session.get_inputs():
            if item.name not in values:
                raise ValueError(f"不支持的 {name} 模型输入：{item.name} ({self.root})")
            if item.type not in types:
                raise ValueError(f"不支持的模型类型：{item.type}")
            feed[item.name] = np.asarray(values[item.name], dtype=types[item.type])
        outputs = session.run(None, feed)
        for value in outputs:
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"{name} 模型输出包含无效数值")
        return dict(zip([o.name for o in session.get_outputs()], outputs))

    def linguistic(self, phones, durations, word_div, word_dur):
        missing = sorted(set(phones) - self.tokens.keys())
        if missing:
            raise ValueError(f"{self.root.name} 不支持音素：{' '.join(missing)}")
        values = dict(tokens=[[self.tokens[p] for p in phones]], ph_dur=[durations],
                      word_div=[word_div], word_dur=[word_dur],
                      languages=[[self.languages.get(p.split('/')[0], 0) if '/' in p else 0 for p in phones]])
        return self.run("linguistic", values)


class Bank:
    def __init__(self, root, language="zh", speaker="", vocoders=None, settings=None):
        self.root = Path(root).resolve()
        self.execution = inference.Context(settings)
        self.main = Model(self.root, self.execution)
        self.language = language
        self.speaker = speaker
        self.vocoders = Path(vocoders) if vocoders else self.root / "vocoders"
        self.models = {}
        self._vocoder_session = None
        self._preflight = None
        self.audio_cache = runtime.Cache()
        self.plan_cache = runtime.Cache(limit=8*1024*1024)
        self.dictionary = {}
        self.vowels = {"SP", "AP"}
        candidates = [self.root / "dsdur" / f"dsdict-{language}.yaml",
                      self.root / "dsdur" / "dsdict.yaml",
                      self.root / f"dsdict-{language}.yaml", self.root / "dsdict.yaml"]
        dictionary = next((p for p in candidates if p.is_file()), None)
        if dictionary:
            data = read_yaml(dictionary)
            self.vowels.update(s["symbol"] for s in data.get("symbols", []) if s.get("type") == "vowel")
            self.dictionary = {str(e["grapheme"]): e["phonemes"] for e in data.get("entries", [])}
        self.main.speaker = speaker
        self.resolver = pronunciation.Resolver(self)

    def model(self, folder):
        if folder not in self.models:
            root = self.root / folder
            if not (root / "dsconfig.yaml").is_file():
                return None
            self.models[folder] = Model(root, self.execution)
            self.models[folder].speaker = self.speaker
            if abs(self.models[folder].dt - self.main.dt) > 1e-9:
                raise ValueError(f"暂不支持帧率与声学模型不同的 {folder} 模型")
        return self.models[folder]

    def phonemes(self, lyric):
        return self.resolver.resolve(lyric)[0]

    def inspect(self):
        cfg = self.main.config
        if self._preflight is None:
            self._preflight = compatibility.preflight(self)
        languages = sorted({p.stem[7:] for folder in (self.root, self.root/'dsdur')
                            for p in folder.glob('dsdict-*.yaml')})
        result = dict(self._preflight, name=self.root.name, languages=languages,
                      speakers=cfg.get('speakers', []), sample_rate=cfg.get('sample_rate', 44100))
        result['expressions'] = expressions.capabilities(self) if result['ok'] else []
        result['inference'] = self.execution.report()
        result['warnings'] = list(result.get('warnings', [])) + self.execution.warnings
        return result

    def vocoder_session(self):
        if self._vocoder_session is None:
            folder, config = self.vocoder_config()
            self._vocoder_session = self.execution.session(resolve(folder, config.get('model', 'model.onnx')), 'vocoder')
        return self._vocoder_session

    def vocoder_config(self, required=True):
        name = self.main.config.get("vocoder", "")
        dirs = [self.root / "dsvocoder"]
        if name:
            dirs += [resolve(self.root, name), self.vocoders / name]
        folder = next((p for p in dirs if (p / "vocoder.yaml").is_file()), None)
        if folder is None:
            if required:
                raise ValueError(f"缺少声码器 {name}。请放入音源的 dsvocoder 或软件 models/diffsinger/vocoders/{name}。")
            return None, {}
        return folder, read_yaml(folder / "vocoder.yaml")

    def expression_curves(self, notes, plan):
        times = plan["origin"] + np.arange(plan["total"])*self.main.dt
        specs = expressions.specifications(self.main.config.get("speakers", []))
        supported = {item["key"] for item in expressions.capabilities(self) if item["supported"]}
        return expressions.sample(notes, times, specs, supported, plan.get("note_onsets"))

    def plan(self, notes):
        key = json.dumps([{k: note.get(k) for k in ('start', 'duration', 'midi', 'lyric', 'resolved_phones', 'resolved_extension')} for note in notes],
                         ensure_ascii=False, separators=(',', ':'))
        cached = self.plan_cache.get(key)
        if cached is not None:
            return cached
        plan = self._plan(notes)
        self.plan_cache.put(key, plan, plan['pitch'].nbytes + plan['voiced'].nbytes
                            + len(key.encode()) + len(plan['phones'])*100 + len(notes)*80)
        return plan

    def _plan(self, notes):
        dt = self.main.dt
        # Absolute boundaries are rounded once; a long MIDI file cannot collect
        # one rounding error per note. Fixed padding is removed after inference.
        head = max(16, round(.5 / dt))
        origin = notes[0]["start"] - head * dt
        words = [dict(phones=["SP"], start=0, end=head, midi=notes[0]["midi"])]
        melody, note_dur, rests = [notes[0]["midi"]], [head], [True]
        last = head
        for ni, note in enumerate(notes):
            start = max(head, round((note["start"] - origin) / dt))
            end = max(start + 1, round((note["start"] + note["duration"] - origin) / dt))
            if start < last:
                raise ValueError("DiffSinger 要求单声部；请将重叠音符移到另一条轨道。")
            if start > last:
                words.append(dict(phones=["SP"], start=last, end=start, midi=note["midi"]))
                melody.append(note["midi"]); note_dur.append(start-last); rests.append(True)
            if note.get('resolved_extension') or (note["lyric"].strip() in ("+", "-") and 'resolved_phones' not in note):
                if len(words) < 2 or words[-1]["phones"] == ["SP"]:
                    raise ValueError("延音符号 + 或 - 必须紧接有歌词的音符。")
                words[-1]["end"] = end
                is_rest = False
            else:
                phones = note.get('resolved_phones') or self.phonemes(note["lyric"])
                words.append(dict(phones=phones, start=start, end=end, midi=note["midi"], owner=ni))
                is_rest = all(p in ("AP", "SP") for p in phones)
            melody.append(note["midi"]); note_dur.append(end-start); rests.append(is_rest)
            last = end
        words.append(dict(phones=["SP"], start=last, end=last+16, midi=notes[-1]["midi"]))
        melody.append(notes[-1]["midi"]); note_dur.append(16); rests.append(True)
        phones = [p for w in words for p in w["phones"]]
        # Duration linguistic groups run from one vowel anchor to the next.
        # An onset consonant belongs to the PRECEDING group's time budget;
        # putting it in its written syllable conditions it on the vowel length.
        anchors = [(0, 0)]
        offset = 0
        for w in words:
            vowel = next((i for i, p in enumerate(w["phones"]) if p in self.vowels), 0)
            anchor = (offset + vowel, w["start"])
            if anchor != anchors[-1]: anchors.append(anchor)
            offset += len(w["phones"])
        anchors.append((len(phones), last+16))
        divisions, word_dur, durations = [], [], []
        for (a, t0), (b, t1) in zip(anchors, anchors[1:]):
            if b <= a or t1 <= t0:
                raise ValueError("音素的元音对齐位置无效")
            divisions.append(b-a); word_dur.append(t1-t0)
            durations.extend(allocate([1 if p in self.vowels else .15 for p in phones[a:b]], t1-t0))
        dur_model = self.model("dsdur")
        if dur_model:
            values = dur_model.linguistic(phones, durations, divisions, word_dur)
            midi_frames = np.repeat(melody, note_dur)
            values["ph_midi"] = [[round(float(midi_frames[t0]))
                for (a, t0), (b, _) in zip(anchors, anchors[1:]) for _ in range(b-a)]]
            predicted = next(iter(dur_model.run("dur", values, len(phones)).values())).reshape(-1)
            if len(predicted) != len(phones):
                raise ValueError("时长模型输出长度与音素数量不一致")
            durations = []
            for (a, t0), (b, t1) in zip(anchors, anchors[1:]):
                weights = np.maximum(predicted[a:b], 0).copy()
                if a == 0 and b > 1:
                    # Keep the predicted leading consonants in frames. SP
                    # consumes the remaining padding, not an extra phoneme's
                    # proportional share. See OpenUtau's starting-consonant rule.
                    leading = np.maximum(1, np.rint(weights[1:])).astype(int).tolist()
                    if sum(leading) < t1-t0:
                        durations.extend([t1-t0-sum(leading), *leading])
                        continue
                    weights[0] = 1
                durations.extend(allocate(weights, t1-t0))
        total = sum(durations)
        if total != sum(note_dur):
            raise ValueError("音素时长和音符时长不一致")
        pitch = np.repeat(np.asarray(melody, dtype=np.float32), note_dur)
        voiced = np.repeat([p not in ("SP", "AP") for p in phones], durations)
        return dict(origin=origin, phones=phones, owners=[w.get("owner", -1) for w in words for p in w["phones"]],
                    ph_dur=durations, word_div=divisions,
                    word_dur=word_dur, note_midi=[melody], note_dur=[note_dur],
                    note_rest=[rests], note_glide=[np.zeros(len(melody), dtype=np.int64)],
                    pitch=pitch, voiced=voiced, total=total)

    def pitch(self, plan, curves=None, existing=None, retake=None):
        model = self.model("dspitch")
        if not model:
            raise ValueError("该音源没有 dspitch 音高预测模型；仍可使用编辑器已有 pitch 合成。")
        n = plan["total"]
        model.expression_curves = curves
        if retake is not None and 'retake' not in {i.name for i in model.session('pitch').get_inputs()}:
            raise ValueError('该音源 pitch 模型没有 retake 输入，不能保留上下文局部重生成；可使用整轨生成')
        values = model.linguistic(plan["phones"], plan["ph_dur"], plan["word_div"], plan["word_dur"])
        values.update({k: plan[k] for k in ("note_midi", "note_dur", "note_rest", "note_glide")})
        values.update(ph_dur=[plan["ph_dur"]], pitch=[plan["pitch"] if existing is None else existing],
                      retake=np.ones((1,n), bool) if retake is None else retake[None], expr=np.ones((1,n), np.float32)
                      if curves is None else curves["DS:PEXP"][None]/100)
        result = next(iter(model.run("pitch", values, n).values())).reshape(-1)
        if len(result) != n or np.any(result < 0) or np.any(result > 127):
            raise ValueError("pitch 模型返回超出 MIDI 范围或长度不符的曲线")
        # Some exported models do not restore the locked frames themselves.
        return result if retake is None else np.where(retake, result, existing)

    def render(self, plan, pitch, curves=None):
        if curves is not None:
            curves = self.parameter_bases(plan, pitch, curves)
        key = (runtime.audio_key(plan, pitch, curves),
               self.execution.options['acoustic_steps'], self.execution.options['variance_steps'],
               self.execution.options['depth'])
        cached = self.audio_cache.get(key)
        if cached is None:
            cached = self._render_raw(plan, pitch, curves)
            self.audio_cache.put(key, cached, cached.nbytes)
        # Never apply dynamics in-place to cached audio: disabling/undoing a
        # curve must return the identical original samples.
        wav = cached.copy()
        if curves is not None:
            db = np.interp(np.arange(len(wav))/self.main.config.get('sample_rate', 44100),
                           np.arange(plan['total'])*self.main.dt, curves['DS:DYN'])/10
            wav *= np.power(10, db/20).astype(np.float32)
        return wav

    def predict_parameters(self, plan, pitch, curves=None, existing=None, retake=None, force_full=False):
        return variance_retake.predict(self, plan, pitch, curves, existing, retake, force_full)

    def parameter_bases(self, plan, pitch, curves):
        curves = dict(curves or {})
        required = [code for code, name in expressions.VARIANCE.items() if self.main.config.get('use_'+name+'_embed')]
        if not required or all('DS:ABS:'+code in curves and np.isfinite(curves['DS:ABS:'+code]).all() for code in required):
            return curves
        _, vc = self.vocoder_config()
        shifted = pitch+curves['DS:SHFC']/100 if vc.get('pitch_controllable', False) and 'DS:SHFC' in curves else pitch
        predicted = self.predict_parameters(plan, shifted, curves or None)
        for code in required:
            name = expressions.VARIANCE[code]
            old = curves.get('DS:ABS:'+code, np.full(plan['total'], np.nan, np.float32))
            curves['DS:ABS:'+code] = np.where(np.isfinite(old), old, np.asarray(predicted[name]).reshape(-1))
        return curves

    def _render_raw(self, plan, pitch, curves=None):
        n = plan["total"]
        cfg = self.main.config
        self.main.expression_curves = curves
        folder, vc = self.vocoder_config()
        f0 = 440*np.exp2((pitch-69)/12)
        acoustic_pitch = pitch if curves is None or not vc.get("pitch_controllable", False) else pitch+curves["DS:SHFC"]/100
        values = dict(tokens=[[self.main.tokens[p] for p in plan["phones"]]],
                      durations=[plan["ph_dur"]], f0=[440*np.exp2((acoustic_pitch-69)/12)],
                      languages=[[self.main.languages.get(p.split('/')[0], 0) if '/' in p else 0 for p in plan["phones"]]],
                      gender=np.zeros((1,n), np.float32), velocity=np.ones((1,n), np.float32))
        if curves is not None:
            values["gender"] = expressions.gender(curves["DS:GENC"], cfg)[None]
            values["velocity"] = np.exp2((curves["DS:VELC"]-100)/100)[None]
        required = [k for k in ("energy", "breathiness", "voicing", "tension") if cfg.get(f"use_{k}_embed")]
        # Avoid re-predicting frozen frames, including after reopening the project.
        needs_prediction = any(curves is None or "DS:ABS:"+code not in curves
            or not np.isfinite(curves["DS:ABS:"+code]).all()
            for code, name in expressions.VARIANCE.items() if name in required)
        predicted = self.predict_parameters(plan, acoustic_pitch, curves) if needs_prediction else {}
        for code, name in expressions.VARIANCE.items():
            if name not in required:
                continue
            base = np.asarray(predicted.get(name+"_pred", predicted.get(name, np.zeros((1,n), np.float32)))).reshape(1,n)
            if curves is not None and "DS:ABS:"+code in curves:
                actual = curves["DS:ABS:"+code][None]
                base = np.where(np.isfinite(actual), actual, base)
            values[name] = expressions.variance_delta(name, base, curves["DS:"+code][None]) if curves is not None else base
        mel = next(iter(self.main.run("acoustic", values, n).values()))
        # Bank-local vocoders take precedence. External packages are selected by
        # name; never substitute a vocoder just because its filename looks right.
        compatibility.validate_vocoder_specs(cfg, vc)
        bases = {"10": math.log(10), "e": 1.0}
        mel = mel * (bases[str(cfg.get("mel_base", "10"))] / bases[str(vc.get("mel_base", "10"))])
        vocoder = self.vocoder_session()
        feed = {"mel": np.asarray(mel, np.float32), "f0": np.asarray([f0], np.float32)}
        missing = [i.name for i in vocoder.get_inputs() if i.name not in feed]
        if missing:
            raise ValueError(f"不支持的声码器输入：{missing}")
        wav = vocoder.run(None, {i.name: feed[i.name] for i in vocoder.get_inputs()})[0].reshape(-1)
        if not np.isfinite(wav).all():
            raise ValueError("声码器返回无效数值")
        return wav


def allocate(weights, length):
    weights = np.maximum(np.asarray(weights, np.float64), .001)
    if length < len(weights):
        raise ValueError("音符过短，无法容纳其音素；请延长音符或减少音素。")
    remaining = length-len(weights)
    boundaries = np.rint(np.r_[0, np.cumsum(weights)/weights.sum()*remaining]).astype(np.int64)
    return (np.diff(boundaries)+1).tolist()


def groups(notes, selected=None, check_overlap=True):
    """Keep monophonic context, but unrelated overlap blocks are scope barriers.

    A selected member of an overlap still fails, even if its partner is muted.
    None means a whole-score operation, which must validate every note.
    """
    result, current = [], []
    ordered = sorted(notes, key=lambda n: n["start"])
    wanted = lambda block: selected is None or any(str(n['id']) in selected for n in block)
    for note in ordered:
        if not math.isfinite(note["start"]) or not math.isfinite(note["duration"]) or note["duration"] <= 0:
            raise ValueError("音符时间必须是有效正时长")
        if note["duration"] > 30:
            raise ValueError("DiffSinger 初版单个音符最长 30 秒，请使用 + 拆分长延音。")
    index = 0
    while index < len(ordered):
        end = index + 1
        occupied_end = ordered[index]['start'] + ordered[index]['duration']
        while check_overlap and end < len(ordered) and ordered[end]['start'] < occupied_end - 1e-6:
            occupied_end = max(occupied_end, ordered[end]['start'] + ordered[end]['duration'])
            end += 1
        if end > index + 1:
            if wanted(ordered[index:end]):
                raise ValueError("DiffSinger 要求单声部；所选音符存在重叠，请将重叠音符移到另一条轨道。")
            if current and wanted(current):
                result.append(current)
            current = []
            index = end
            continue
        note = ordered[index]
        if current and (note.get('context') != current[-1].get('context') or note["start"] - (current[-1]["start"]+current[-1]["duration"]) > .4
                        or note["start"] - current[0]["start"] > 15) and note["lyric"].strip() not in ("+", "-") and not re.fullmatch(r'\+\d+', note["lyric"].strip()) and not note.get('resolved_extension'):
            if wanted(current): result.append(current)
            current = []
        current.append(note)
        index = end
    if current and wanted(current): result.append(current)
    return result


def execute(request, output, bank=None):
    if bank is None:
        bank = Bank(request["voicebank"], request.get("language", "zh"), request.get("speaker", ""), request.get("vocoders"), request.get('inference'))
    bank.execution.update(request.get('inference'))
    bank.parameter_events = []
    if request["operation"] == "inspect":
        return bank.inspect()
    if request["operation"] not in ("pitch", "render", "timing", "phonemize", "parameters"):
        raise ValueError("未知 DiffSinger 操作")
    selected = request.get('retake_ids')
    selected = None if selected is None else set(map(str, selected))
    if selected is not None and (not selected or not selected.issubset({str(n['id']) for n in request['notes']})):
        raise ValueError('局部生成所选音符无效或已不存在')
    scope = ({str(n['id']) for n in request['notes'] if n.get('audible', True)}
             if request['operation'] == 'render' else selected)
    # Scope before lyric conversion too: an unrelated misspelled lyric or
    # unsupported phoneme must not block a valid local edit/render. Retain each
    # selected phrase, including Chinese word and continuation-note context.
    if scope is None:
        prepared, pronunciations = bank.resolver.prepare(request['notes'], request.get('dictionary', ''))
        phrases = None
    else:
        phrases, pronunciations = [], []
        for context in groups(request['notes'], scope, check_overlap=request['operation'] != 'phonemize'):
            prepared, readings = bank.resolver.prepare(context, request.get('dictionary', ''), context_notes=request['notes'])
            phrases.append(prepared)
            pronunciations.extend(readings)
    if request['operation'] == 'phonemize':
        return dict(ok=True, pronunciations=[r for r in pronunciations if selected is None or str(r['id']) in selected])
    inspected = bank.inspect()
    if not inspected['ok']:
        return inspected
    sr = bank.main.config.get("sample_rate", 44100)
    duration = float(request["duration"])
    if not math.isfinite(duration) or not 0 < duration <= 3600:
        raise ValueError("一次请求时长须在 0 到 3600 秒之间")
    audio = np.zeros(math.ceil(duration*sr), np.float32) if request["operation"] == "render" else None
    curves, phonemes, ignored_timings = [], [], []
    parameters = []
    timing_clamped = False
    generated_phrases = 0
    if phrases is None:
        phrases = groups(prepared)
    for phrase, notes in enumerate(phrases):
        if request['operation'] in ('pitch', 'parameters') and selected is not None and not any(str(n['id']) in selected for n in notes):
            continue
        if request["operation"] == "render" and not any(n.get("audible", True) for n in notes):
            continue
        plan, rows, ignored, clamped = timing.apply(bank, notes, bank.plan(notes))
        plan['parameter_context'] = [(n.get('context',''), str(n['id']), n['start'], n['duration']) for n in notes]
        phonemes.extend(dict(row, phrase=phrase) for row in rows)
        ignored_timings.extend(ignored)
        timing_clamped |= clamped
        if request["operation"] == "timing":
            continue
        expression_curves = bank.expression_curves(notes, plan)
        if request["operation"] == "pitch":
            if selected is None:
                pitch = bank.pitch(plan, expression_curves)
            else:
                time = plan['origin'] + np.arange(plan['total'])*bank.main.dt
                existing = plan['pitch'].copy()
                retake = np.zeros(plan['total'], bool)
                for ni, note in enumerate(notes):
                    # Retake follows MIDI selection boundaries. Locked notes,
                    # gaps and padding retain the effective editor pitch.
                    mask = (time >= note['start']) & (time < note['start']+note['duration'])
                    points = np.asarray(note.get('pitch', []), dtype=np.float64)
                    if points.size:
                        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or np.any(np.diff(points[:,0]) < 0):
                            raise ValueError('已有 pitch 曲线含无效点或时间顺序错误')
                        left = -np.inf if ni == 0 else note['start']
                        right = notes[ni+1]['start'] if ni+1 < len(notes) else np.inf
                        lock_mask = (time >= left) & (time < right)
                        existing[lock_mask] = np.interp(time[lock_mask]-note['start'], points[:,0], points[:,1])
                    if str(note['id']) in selected:
                        retake |= mask
                if not retake.any():
                    raise ValueError('所选音符短于一个预测帧，请延长音符')
                pitch = bank.pitch(plan, expression_curves, existing, retake)
            generated_phrases += 1
            for note in notes:
                if selected is not None and str(note['id']) not in selected:
                    continue
                if note.get('resolved_phones', [None])[0] in ('SP', 'AP'):
                    continue
                times = np.arange(0, note["duration"], .005)
                times = np.r_[times, note["duration"]]
                midi = np.interp(note["start"]+times, plan["origin"]+np.arange(len(pitch))*bank.main.dt, pitch)
                curves.append(dict(id=note["id"], points=np.c_[times, midi].tolist()))
        else:
            pitch = plan["pitch"].copy()
            time = plan["origin"]+np.arange(len(pitch))*bank.main.dt
            # User curves, including vibrato and shared PIT, are authoritative.
            for index, note in enumerate(notes):
                points = note.get("pitch", [])
                if points:
                    p = np.asarray(points)
                    left = plan["note_onsets"][index] if index else -np.inf
                    right = plan["note_onsets"][index+1] if index+1 < len(notes) else np.inf
                    mask = (time >= left) & (time < right)
                    pitch[mask] = np.interp(time[mask]-note["start"], p[:,0], p[:,1])
            if request['operation'] == 'parameters':
                _, vc = bank.vocoder_config()
                acoustic_pitch = pitch + expression_curves['DS:SHFC']/100 if vc.get('pitch_controllable', False) else pitch
                if selected is None:
                    predicted = bank.predict_parameters(plan, acoustic_pitch, expression_curves, force_full=True)
                else:
                    supported = {x['key'] for x in expressions.capabilities(bank) if x['supported']}
                    existing = expressions.sample_absolute(notes, time, supported, plan['note_onsets'])
                    mask = variance_retake.selection_mask(notes, plan, time, selected)
                    predicted = bank.predict_parameters(plan, acoustic_pitch, expression_curves, existing=existing, retake=mask)
                if not predicted:
                    raise ValueError('当前音源没有可预测的能量/气声/张力/实声参数；其他控制仍可使用 DF FLAG')
                for ni, note in enumerate(notes):
                    if selected is not None and str(note['id']) not in selected:
                        continue
                    left = plan['note_onsets'][ni]-note['start']
                    right = max(note['duration'], (plan['note_onsets'][ni+1]-note['start']) if ni+1<len(notes) else note['duration'])
                    # Keep the original frame knots. No decimation of synthesis/reference data.
                    local = time-note['start']
                    samples = np.unique(np.r_[left, local[(local>left)&(local<right)], right])
                    row = dict(id=note['id'], curves={})
                    for code, name in expressions.VARIANCE.items():
                        if name in predicted:
                            lo, hi = (-10,10) if code=='TENC' else (-96,0)
                            values = np.clip(np.asarray(predicted[name]).reshape(-1), lo, hi)
                            row['curves'][code] = np.c_[samples, np.interp(note['start']+samples,time,values)].tolist()
                    parameters.append(row)
                generated_phrases += 1
                continue
            supported = {x['key'] for x in expressions.capabilities(bank) if x['supported']}
            expression_curves.update(expressions.sample_absolute(notes, time, supported, plan['note_onsets']))
            expression_curves = bank.parameter_bases(plan, pitch, expression_curves)
            wav = bank.render(plan, pitch, expression_curves)
            for ni, note in enumerate(notes):
                if selected is not None and str(note['id']) not in selected:
                    continue
                left = plan['note_onsets'][ni]-note['start']
                right = max(note['duration'], plan['note_onsets'][ni+1]-note['start'] if ni+1<len(notes) else note['duration'])
                local = time-note['start']
                samples = np.unique(np.r_[left, local[(local>left)&(local<right)], right])
                row = dict(id=note['id'], curves={})
                for code in expressions.VARIANCE:
                    values = expression_curves.get('DS:ABS:'+code)
                    if values is not None and np.isfinite(values).all():
                        lo, hi = (-10,10) if code=='TENC' else (-96,0)
                        row['curves'][code] = np.c_[samples, np.interp(note['start']+samples,time,np.clip(values,lo,hi))].tolist()
                if row['curves']:
                    parameters.append(row)
            start = round(plan["origin"]*sr)
            lo, hi = max(0, start), min(len(audio), start+len(wav))
            if hi > lo: audio[lo:hi] += wav[lo-start:hi-start]
    if audio is not None:
        import soundfile as sf
        sf.write(str(Path(output).with_suffix(".wav")), audio, sr, subtype="FLOAT")
    return dict(ok=True, curves=curves, parameters=parameters, parameter_retake=bank.parameter_events, phonemes=phonemes, pronunciations=pronunciations,
                generated_phrases=generated_phrases, frame_seconds=bank.main.dt,
                ignored_timings=ignored_timings, timing_clamped=timing_clamped,
                sample_rate=sr, inference=bank.execution.report(),
                backend='DiffSinger ONNX (' + bank.execution.report()['backend'] + ')')


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        runtime.serve(Path(sys.argv[2]), int(sys.argv[3]), Bank, execute)
        sys.exit(0)
    output = Path(sys.argv[2])
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
        result = execute(request, output)
    except Exception as error:
        result = dict(ok=False, error=f"{type(error).__name__}: {error}")
    output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    sys.exit(0 if result["ok"] else 1)
