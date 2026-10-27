"""Offline lyric/hint conversion. Every output is checked against this bank.

No voicebank code is executed and no missing sound is silently dropped. Bank
dictionaries take priority over general English G2P; user readings are per-track.
"""
from functools import lru_cache
from pathlib import Path
import re
import sys
import unicodedata

DATA = Path(__file__).parent / 'pronunciation-data'
sys.path.insert(0, str(DATA))
HAN = re.compile(r'^[\u3400-\u9fff]+$')
VOWELS = set('aa ae ah ao aw ay eh er ey ih iy ow oy uh uw ax'.split())
ONSETS = set('bl br ch cl cr dr dw fl fr gl gr kl kr pl pr sk sl sm sn sp st sw tr tw th sh shr spl spr str skr skw thr'.split())


@lru_cache(maxsize=1)
def english_dictionary():
    words = {}
    for line in (DATA / 'cmudict.dict').read_text('utf-8').splitlines():
        parts = line.split('#', 1)[0].split()
        if len(parts) > 1:
            words.setdefault(re.sub(r'\(\d+\)$', '', parts[0]).lower(), parts[1:])
    return words


@lru_cache(maxsize=1)
def english_predictor():
    from english_g2p import EnglishPredictor
    return EnglishPredictor()


def split_language(text, default):
    match = re.match(r'^([a-z]{2,8})/(.+)$', text, re.S)
    return (match[1], match[2].strip()) if match else (default, text)


def dictionary_entries(text, language):
    result = {}
    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            raise ValueError(f'用户词典第 {i+1} 行缺少 =，格式：歌词 = 读音')
        key, value = (x.strip() for x in line.split('=', 1))
        lang, key = split_language(key, language)
        if not key or not value:
            raise ValueError(f'用户词典第 {i+1} 行歌词或读音为空')
        if (lang, key.lower()) in result:
            raise ValueError(f'用户词典重复条目：{lang}/{key}')
        result[lang, key.lower()] = value
    return result


class Resolver:
    def __init__(self, bank):
        self.bank = bank
        self.tables = {bank.language: bank.dictionary}

    def table(self, language):
        if language not in self.tables:
            # Import lazily to avoid a cycle when bridge constructs this class.
            from bridge import read_yaml
            paths = [self.bank.root / 'dsdur' / f'dsdict-{language}.yaml',
                     self.bank.root / f'dsdict-{language}.yaml']
            path = next((p for p in paths if p.is_file()), None)
            data = read_yaml(path) if path else {}
            self.bank.vowels.update(s['symbol'] for s in data.get('symbols', []) if s.get('type') == 'vowel')
            self.tables[language] = {str(e['grapheme']): e['phonemes'] for e in data.get('entries', [])}
        return self.tables[language]

    def token(self, value, language):
        if value in ('SP', 'AP') or '/' in value:
            choices = [value]
        else:
            choices = [language + '/' + value, value]
        token = next((p for p in choices if p in self.bank.main.tokens), None)
        if token is None:
            raise ValueError(f'音源不支持音素 {value}（语言 {language}）；请查看音素表并修改读音')
        # Validate all installed predictors now, instead of failing at playback.
        for folder in ('dsdur', 'dspitch', 'dsvariance'):
            model = self.bank.model(folder)
            if model is not None and token not in model.tokens:
                raise ValueError(f'{folder} 不支持音素 {token}')
        return token

    def explicit(self, text, language):
        phones = [self.token(p, language) for p in text.split()]
        if not phones:
            raise ValueError('显式音素不能为空')
        return phones

    def lookup(self, text, language):
        # Multilingual banks often include every language in each dictionary.
        for table in (self.bank.dictionary,):
            value = table.get(language + '/' + text)
            if value is None and language == self.bank.language:
                value = table.get(text)
            if value is not None:
                return self.explicit(' '.join(value) if isinstance(value, list) else value, language)
        table = self.table(language)
        value = table.get(language + '/' + text, table.get(text))
        return None if value is None else self.explicit(' '.join(value) if isinstance(value, list) else value, language)

    def english(self, word):
        raw = english_dictionary().get(word)
        source = '英文词典'
        if raw is None:
            if not re.fullmatch("[a-z]+(?:'[a-z]+)?", word) or len(word) > 80:
                raise ValueError(f'无法转换英文“{word}”；数字请写成单词，特殊读音请手动覆盖')
            raw, source = english_predictor().predict(word), '英文生词预测（请试听确认）'
        phones = []
        for phone in raw:
            base = re.sub('[012]$', '', phone).lower()
            # OpenUtau/ARPAbet dictionaries may distinguish unstressed schwa.
            candidates = (['ax', 'ah'] if phone == 'AH0' else [base])
            mapped = next((self.token(p, 'en') for p in candidates
                           if 'en/'+p in self.bank.main.tokens or p in self.bank.main.tokens), None)
            if mapped is None:
                raise ValueError(f'英文音素 {phone} 无对应音源标识，请用读音覆盖指定该音源的音素')
            if base in VOWELS:
                self.bank.vowels.add(mapped)
            phones.append(mapped)
        return phones, source

    def resolve(self, text, language=None, user=None, depth=0):
        if depth > 8:
            raise ValueError('用户词典存在循环读音引用')
        language, text = split_language(unicodedata.normalize('NFKC', text).strip(), language or self.bank.language)
        if not text or text.upper() in ('RR', 'R', 'SP'):
            return [self.token('SP', language)], '休止'
        if text == 'AP':
            return [self.token('AP', language)], '换气'
        hint = re.fullmatch(r'(.*?)\[([^\[\]]*)\]', text, re.S)
        if hint:
            # lyric[phones] and [phones] share the same explicit representation.
            self.table(language)
            return self.explicit(hint[2], language), '手动音素'
        if '[' in text or ']' in text:
            raise ValueError(f'读音括号不完整：{text}')
        if user and (language, text.lower()) in user:
            phones, _ = self.resolve(user[language, text.lower()], language, user, depth+1)
            return phones, '用户词典'
        found = self.lookup(text, language) or self.lookup(text.lower(), language)
        if found:
            return found, '音源词典'
        if text in self.bank.main.tokens or language+'/'+text in self.bank.main.tokens:
            return [self.token(text, language)], '单音素'
        if language == 'zh' and HAN.fullmatch(text):
            from pypinyin import lazy_pinyin
            phones = []
            for syllable in lazy_pinyin(text, errors='default'):
                if HAN.search(syllable):
                    raise ValueError(f'汉字无法注音：{syllable}；请填写拼音')
                phones.extend(self.resolve(syllable, 'zh', user, depth+1)[0])
            return phones, '中文词组读音'
        if language == 'zh':
            from pypinyin.contrib.tone_convert import to_normal
            normalized = to_normal(text.lower().replace('u:', 'v').replace('ü', 'v'))
            normalized = re.sub(r'[1-5]', '', normalized).replace('ü', 'v')
            found = self.lookup(normalized, 'zh')
            if found:
                return found, '拼音'
        # Whitespace/apostrophe splits explicit pinyin syllables; English words
        # retain apostrophes. Unprefixed Latin OOVs in Chinese use English G2P.
        pieces = text.split()
        if language == 'zh' and "'" in text and all(self.lookup(p, 'zh') for p in text.split("'")):
            pieces = text.split("'")
        if len(pieces) > 1:
            values = [self.resolve(p, language, user, depth+1) for p in pieces if p]
            return [p for v, _ in values for p in v], ' / '.join(dict.fromkeys(s for _, s in values))
        if language == 'en' or (language == 'zh' and re.fullmatch("[A-Za-z]+(?:'[A-Za-z]+)?", text)):
            if language != 'en':
                found = self.lookup(text.lower(), 'en')
                if found:
                    return found, '英文音源词典'
            return self.english(text.lower())
        raise ValueError(f'词典未找到“{text}”（{language}），请指定读音或 [音素 音素]')

    def syllables(self, phones):
        nuclei = [i for i, p in enumerate(phones) if p in self.bank.vowels and p not in ('SP', 'AP')]
        if len(nuclei) < 2:
            return [phones]
        starts = [0]
        for a, b in zip(nuclei, nuclei[1:]):
            between = [p.split('/')[-1] for p in phones[a+1:b]]
            onset = 0
            for length in range(1, len(between)+1):
                tail = between[-length:]
                if (length == 1 and tail[0] not in ('ng', 'N')) or ''.join(tail) in ONSETS:
                    onset = length
            starts.append(b-onset)
        return [phones[a:b] for a, b in zip(starts, starts[1:]+[len(phones)])]

    def prepare(self, notes, dictionary='', context_notes=None):
        user = dictionary_entries(dictionary, self.bank.language)
        # Validate dictionary entries even when no current note uses them.
        for (lang, text), value in user.items():
            self.resolve(value, lang, user)
        ordered = [dict(n) for n in sorted(notes, key=lambda n: n['start'])]
        readings = {}
        # Adjacent Chinese notes form context for words such as 音乐 and 银行.
        run = []
        def flush():
            if not run:
                return
            from pypinyin import lazy_pinyin
            text = ''.join(n['lyric'] for n in run)
            values = lazy_pinyin(text)
            offset = 0
            for n in run:
                count = len(n['lyric'])
                readings[str(n['id'])] = ' '.join(values[offset:offset+count])
                offset += count
            run.clear()
        # Text context may cross an inference phrase boundary. Read it without
        # validating unrelated phonemes; only the scoped notes below are resolved.
        text_context = ordered if context_notes is None else sorted(context_notes, key=lambda n: n['start'])
        for n in text_context:
            eligible = self.bank.language == 'zh' and HAN.fullmatch(n['lyric']) and not n.get('pronunciation')
            if not eligible or (run and (n['start'] > run[-1]['start']+run[-1]['duration']+.08
                                        or n.get('context') != run[-1].get('context'))):
                flush()
            if eligible:
                run.append(n)
        flush()
        reports = []
        i = 0
        while i < len(ordered):
            n = ordered[i]
            text = n.get('pronunciation', '').strip() or n['lyric'].strip()
            if text in ('+', '-'):
                if i == 0 or abs(n['start']-ordered[i-1]['start']-ordered[i-1]['duration']) > 1e-5 or n.get('context') != ordered[i-1].get('context'):
                    raise ValueError('延音 + 或 - 必须紧接有歌词的音符')
                previous = next((item.get('resolved_phones') for item in reversed(ordered[:i]) if item.get('resolved_phones')), [])
                if not previous or all(p in ('SP', 'AP') for p in previous):
                    raise ValueError('休止或换气后不能使用延音')
                n['resolved_extension'] = True
                reports.append(dict(id=str(n['id']), lyric=n['lyric'], phones=[], source='延音'))
                i += 1
                continue
            if re.fullmatch(r'\+\d+', text):
                raise ValueError(f'{text} 必须紧接多音节词，并按 +2、+3 顺序分配')
            reading = readings.get(str(n['id']), text)
            if (self.bank.language, text.lower()) in user:
                reading = text
            try:
                phones, source = self.resolve(reading, user=user)
            except ValueError as error:
                raise ValueError(f'音符 {n["lyric"]}：{error}') from error
            if reading != text:
                source = '中文词组读音：' + reading
            chunks = self.syllables(phones)
            end = i+1
            while end < len(ordered) and re.fullmatch(r'\+\d+', ordered[end]['lyric'].strip()):
                expected = end-i+1
                item = ordered[end]
                if item.get('pronunciation') or item['lyric'].strip() != '+'+str(expected):
                    raise ValueError('多音节分配请依次使用 +2、+3；分配音符不要另填读音覆盖')
                previous = ordered[end-1]
                if abs(item['start']-previous['start']-previous['duration']) > 1e-5 or item.get('context') != previous.get('context'):
                    raise ValueError('多音节词与 +2、+3 音符必须在同一片段内连续')
                end += 1
            count = end-i
            if count > len(chunks):
                raise ValueError(f'“{n["lyric"]}”只有 {len(chunks)} 个音节；多余音符请使用 + 延音')
            for j in range(count):
                item = ordered[i+j]
                selected = chunks[j] if j < count-1 else [p for c in chunks[j:] for p in c]
                item['resolved_phones'] = selected
                # The planner only interprets bare + or -; keep display lyrics.
                reports.append(dict(id=str(item['id']), lyric=item['lyric'], phones=selected,
                                    source=source + (f' · 第 {j+1} 音节' if count > 1 else '')))
            i = end
        return ordered, reports
