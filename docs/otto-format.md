# Folder-level native `otto.ini`, phase 1

Implementation: [OttoMaterialIO.h](../juce/src/OttoMaterialIO.h), [OttoMaterialIO.cpp](../juce/src/OttoMaterialIO.cpp). This is **not** classic UTAU `oto.ini`. There is no whole-file OTO export or GUI in this module.

## Version 1

```ini
[otto]
version=1
unit=ms

[a01:recordings/sample.wav]
p01=ka;ka_alt|0,600|100,80,20|C:100,V:350,V:600|next:p02
p02=a|600,1000|0,0,0|V:1000|flags:transition

[a02:subfolder/empty.wav]
```

One source-root manifest covers subfolders. Each audio section has a stable annotation-variant audio ID and a root-relative filename. **Different audio IDs may intentionally share the same relative path**; this represents separate annotation registrations over one unchanged physical recording, not duplicate PCM. `listAudioVariants` returns them in document order. A path-only read/write is permitted only when unambiguous; the caller must retain and supply the chosen `audioId` when multiple variants exist. Each pronunciation is exactly one line with its own stable ID, independent of alias, position, or timing. IDs are case-sensitive ASCII letters/digits/underscore/hyphen, 1–128 characters; audio IDs are unique in a document; pronunciation IDs are unique within their audio section. Pronunciation identity is the composite audio ID + pronunciation ID; different audio sections can each contain `p01`. Newly assigned IDs are `a` or `p` followed by a random UUID without hyphens. Existing supplied IDs are preferred.

The four required pronunciation fields are:

1. Primary alias, followed by optional additional aliases separated by `;`.
2. Source start and end, absolute audio milliseconds.
3. Fixed duration, preutterance and overlap, in milliseconds. These are durations, not source boundary coordinates. The C++ row's absolute `alignmentSeconds` is converted to/from `regionStartSeconds + preutteranceSeconds`.
4. Comma-separated role and **absolute audio end** pairs. A segment starts at the preceding end, or the pronunciation start. `-` means no explicit segments. Role codes: `U` unknown, `C` consonant, `V` vowel, `T` transition, `S` silence, `B` breath, `N` noise, `E` ending. Arbitrarily many segments and repeated vowel segments are supported.

Compact segments reconstruct deterministic defaults: ID `segment_N` (one-based), trimmed primary alias (or `-`), row provenance (or `estimated`), row confidence/alignment/overlap, stretchable unless consonant, weight 1. Row role defaults to the first segment role. Metadata overrides these only when original fields differ, so ordinary newly authored segmented pronunciations do not need a large metadata payload.

Optional `|next:<pronunciation-id>` and `|flags:<comma-separated flags>` may follow. Whole-pronunciation identity is exactly **normal syllable**, **transition**, or **breath**, independent of internal C/V roles. Defined identity flags are `normal`, `transition`, `breath`; normal is normally omitted. `normal` is written explicitly only when needed to disambiguate a retained historical internal breath/transition role. At most one identity flag is allowed. `simple-consonant` enables the optional linked C/P constraint and is **off by default**; C-P is derived from the existing timing numbers, not redundantly stored. `vowel-splits` records explicitly created and named manual vowel subdivisions (also off by default). Without that intent, analysis/project-note epochs are not authoring subdivisions: normal derivation gives one C prefix and one elastic V, with P retained as a marker, not another C split. Raw legacy metadata is preserved by pure reads; draft normalization is only published on an explicit Save. Manual names may repeat because stable IDs/ordinal positions carry identity; missing/unassigned names are invalid. There is no artificial count limit; `chang` → `an` / `ng` is a typical two-part use for stretch-ratio weights. Duplicate/unknown/conflicting flags fail decoding. An optional `|meta:<base64>` is always **last**. Optional fields cannot repeat; unknown fields fail decoding.

### Shared pronunciation timing

[PronunciationTiming](../juce/src/PronunciationTiming.h) is the single interpretation of offset, preutterance P, consonant end C and source end used by native/OTO editor views. P and C are absolute source positions internally; the compact tuple retains the classic relative durations. Signed overlap remains a relative offset from source start and is never silently clamped. With simplified consonants enabled, moving P moves C by the same amount, preserving the nonnegative derived C-P distance; editing that distance explicitly moves C. Disabling the constraint leaves the current marker positions intact.

Consonants and all material before P are protected from time stretching. Existing internal vowel subdivisions and weights remain available; only elastic vowel intervals after the protected region absorb duration changes. Impossible contractions reject rather than compressing consonants or dropping audio. Kind is not a replacement for these source zones: the old `vowel`/`consonant` dropdown must not classify a whole syllable.

Adjacent annotation mode uses the exact existing pronunciation start/end as its shared join, constrains the free edge to the nearest occupied boundary or recording edge and records explicit same-audio links. It neither physically cuts the WAV nor creates overlap/gaps at the join. Arbitrary overlapping sample ranges may still exist in imported documents; adjacent mode does not jump across them or infer a different recording.

### Escaping

Filenames and each alias use UTF-8 byte percent escaping, `%HH` (hexadecimal, either case on read; uppercase on write). The writer leaves only ASCII letters/digits and `/._-` literal; everything else is escaped. Thus `|`, `;`, `=`, `:`, `%`, `[`, `]`, backslash, whitespace, line breaks and non-ASCII characters round-trip without acting as delimiters. Examples: `a%3Bb` is one alias `a;b`, `%E3%81%8B` is `か`, `sub/a%5Bb%5D.wav` is `sub/a[b].wav`. Malformed escapes, invalid UTF-8 and escaped NUL fail. Read also accepts literal UTF-8 and some nonstructural punctuation, but reserved structural characters must be escaped. No inline comments are supported. Blank lines and full-line `;` / `#` comments are accepted; comments/formatting are not retained on write.

Decoded paths must be nonempty relative slash-separated paths: no leading slash, drive colon, backslash, empty component, `.` or `..` component, or newline. Calls also require the target audio to be a child of the manifest folder. This is lexical containment, not a sandbox against filesystem junction/symlink redirection. Windows compares audio paths case-insensitively; aliases and IDs remain case-sensitive everywhere.

### Migration extension (not another pronunciation format)

`meta:` contains standard Base64 of compact UTF-8 JSON. It is omitted if the compact line plus C++ defaults can reconstruct all fields. Otherwise the JSON contains only needed nondefault row properties, complete extended segment records when needed, and/or envelope points. This is a lossless migration bridge, not permission to silently discard fields.

Supported row properties are the existing C++ field names: `relativePitchCents`, `melodyneData`, all numeric Melodyne parameters (`melodynePitchCenterCents`, `melodyneOriginalPitchCenterCents`, `melodynePitchDrift`, `melodynePitchModulation`, `melodyneTransitionSeconds`, `melodyneFormantCents`, `melodyneAmplitude`, `melodyneSibilantBalance`, `melodyneAttackSeconds`, `melodyneDecayElongation`; the two pitch-center values are included in this list), `hjmVersion`, `role` (integer enum 0–7), `provenance`, `confidence`, `nativeUnpitched`, `segments`, and `amplitudeEnvelope`. Core timing fields may also appear solely to retain exact original seconds when the milliseconds round trip changes a floating-point bit; a contradiction larger than 1 ns is rejected.

Each full segment object contains **all** fields: `id`, `alias`, `role`, `sourceStartSeconds`, `sourceEndSeconds`, `provenance`, `confidence`, `alignmentSeconds`, `overlapSeconds`, `stretchable`, `stretchWeight`. Source coordinates are absolute audio seconds, matching `SampleRegionSetting.segments` (not note-local coordinates). Segment count, role and end must agree with the compact core. Full metadata preserves intentional source gaps and extended policy without changing the compact grammar.

Each envelope object contains `timeSeconds`, `gainDb`, `linearToNext`, `nativeSeamAnchor`, retaining negative envelope times and seam information. JSON numbers are emitted with 17-digit decimal precision. Metadata types, property names and enum values are checked; unknown properties fail rather than disappear in a later save. Add new C++ model fields to this codec before using it to migrate those fields.

## 同录音注音变体与登记约束（2026-10-22 集成补充）

以下是当前服务契约，不是全部 GUI／实际工程音频已验收的声明；公开版本仍为 0.25。新增同路径不同 ID 使用现有版本 1 的音频节，不引入另一份物理录音或平行注音格式。例如 `[takeA:recording.wav]` 与 `[takeB:recording.wav]` 是两套独立登记，行身份仍是音频 ID＋发音 ID；路径、别名、行序均不是身份。`containsAudio` 只回答存在一套或多套，不能代替选择。

社区 CSV 导入和工程素材登记都必须复用本模块的整文档校验及冲突保护发布。社区数据集没有独立原生 P/C，映射元音核到这两个标记必须显式确认；不根据社区的词划分自动恢复 C+V 音节。工程登记默认拒绝未确认事实；用户确认携带的摆放解释后，保留已有边界、P 和未命名 `-` 标签，不把 F0 帧或原始分析分区赋予“手工 V 切点”的意图。

项目内的原始时钟和已有原始分区（当前真实工程审计为 5513 个时钟结点、209 个分区）是工程携带事实，不需要也不应逐一变成新的 OTTO 音节。`vowel-splits` 只能表示实际手工命名分段。一个物理录音可只复制一次，再由多套注音 ID 引用；跨文件发布、项目收纳与素材根偏好不是一个不可失败的原子操作，28b 已用真实 teio 完成登记前／收纳后／冷重开的独立实际音频验证：31 个完整 WAV、81 套变体，三阶段无回退 CPU ONNX 且解码 PCM 完全一致（[实际输出](<G:/hjs-tc/tmp/teio-reg28/stdout.txt>)）。这不代替尚未接线的 Main 登记 GUI 或全部局部编辑入口验收。细节与剩余工作见 [实现状态](native-material-implementation.md)。

## Validation and authority

- The entire document must parse and validate before a read succeeds or any write is attempted. Missing headers, unknown header keys, unsupported versions/units, duplicate IDs, duplicate aliases within a pronunciation, malformed numbers and unknown extension fields fail. Unknown sections/keys are **not preserved: loading and writing are refused**, rather than destroying their contents.
- Aliases must be nonempty and unique within one pronunciation's alias list (including the primary alias). Different stable pronunciation IDs may use the same alias in the same or different recordings; identity is not a label.
- Source ranges must be positive and nonnegative; all numeric model data must be finite. Fixed duration and preutterance must lie in `[0, region length]`; signed overlap may exceed region length and must only be finite. The model's absolute alignment must lie between source start and end. Confidence is `[0,1]`, stretch weight is nonnegative. Segments are positive, ordered, nonoverlapping and inside their pronunciation. Metadata may retain gaps; complete source coverage is not required. Envelope times are nondecreasing.
- A connection must target another pronunciation in the **same recording**, cannot self-link or form a cycle, and must have `abs(sourceEnd - next.sourceStart) <= 1 ms`. Ordinary reads do not invent links. Explicit adjacent-annotation operations may create exact touching links as part of the same undoable draft transaction.
- No audio decoding is performed; the codec cannot verify a source range against the actual recording duration. Missing audio paths can remain in a document.
- An explicitly empty audio section is authoritative: `containsAudio` and `loadAudio` return true, and `loadAudio` returns an empty vector. No matching section makes `loadAudio` return false with a meaningful error; `containsAudio` instead returns false with an empty error for a valid document. This low-level absence result does **not** permit a material reader to revive legacy OTO/CSV aliases: an existing root OTTO is authoritative for the entire registered folder, including absent recording sections. Malformed manifests make both functions fail with an error. Failed reads leave the caller's row/document output unchanged.
- `findManifest(audio)` returns the nearest ancestor existing `otto.ini`, stopping at the filesystem root. An existing invalid manifest (including a directory at that name) is returned, never bypassed for a farther ancestor.

## API, merge and transactions

`saveAudio(manifest, audio, rows, error, expectedFingerprint, audioId)` replaces only the selected annotation variant's rows and preserves every other variant semantically. Empty `audioId` permits only an unambiguous physical path; a new explicit ID may create a sibling variant without overwriting an existing one. The same optional ID is accepted by `loadAudio` and `replaceAudio`. Empty-row variants also require explicit selection when the path is ambiguous. Supplied stable IDs win; rows without IDs match an existing **primary alias only when unique**. Renamed rows without an ID are new rows with fresh UUIDs: no index matching or positional ID reuse. To rename while keeping references, retain `materialRegionId`. An existing audio ID must remain bound to its original relative path; an explicit new sibling ID does not rename or replace that existing section. Empty rows keep/create the selected audio section.

For independently editable drafts, use the optional `loadAudio(..., error, &fingerprint)` output and pass the returned token to `saveAudio(..., error, fingerprint)`. `"absent"` is the explicit expected token for creating a nonexistent manifest. Tokens are bounded SHA-256 digests of the original bytes with a `v1:` prefix. The four-argument convenience calls maintain a process-local last-observation cache per manifest/audio, capped at 256 entries; this catches intervening external changes but **cannot distinguish overlapping drafts when another reader refreshes the same cache entry**. Explicit tokens are recommended for manager/editor transactions.

For multi-recording transactions, use `loadDocument`, call pure `replaceAudio(document, manifest, audio, rows, error)` for each recording, then `saveDocument` once. `replaceAudio` uses the same identity/merge logic as `saveAudio`, validates the complete candidate, performs no file I/O, and leaves its input document unchanged on failure. You can also edit `Document.audio` directly and validate before saving. Keep the original `Document.fingerprint` and `existed` fields untouched. A default document represents a new, absent manifest; `decode` alone does not confer overwrite permission. `encode`, `decode` and `validate` are available for tests and future manager transactions. Successful `saveDocument` updates its byte fingerprint; failed operations do not claim publication.

Saves use a named cross-process lock derived from the normalized manifest path, exact byte fingerprint recheck under that lock, a same-directory JUCE `TemporaryFile`, UTF-8 **without BOM**, **LF** output, temporary-file reread/validation, a second destination content check, then atomic replacement. Reads accept optional UTF-8 BOM and CRLF. Cooperating writers serialize; noncooperating external editors are detected up to the last recheck (there is an unavoidable check-to-replace window for writers that ignore the lock). Conflict errors require a deliberate reload/reconcile, never silent overwrite. Locks time out after five seconds. No build or runtime verification is claimed by this format document.
