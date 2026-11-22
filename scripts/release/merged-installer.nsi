; HiFiShifter merged installer (NSIS): standalone App + VST3/ARA plugins + shared model library.
;
; [Why a hand-written script instead of Tauri's NSIS template or installerHooks]
;
; - bundle.windows.nsis.installerHooks can only **inject code** into the script Tauri
;   generates; it cannot add a components page, nor a "second install directory" (the VST3
;   directory). The models would still be packed into the App directory by Tauri's payload —
;   exactly the second full copy of the weights this is meant to eliminate.
; - A custom nsis.template means copying out the 700-line template together with all of its
;   implicit !define contracts, then drifting along with tauri-cli versions. Every upstream
;   template upgrade would turn into a manual merge.
;
; So this file is its own script: pages, components, directories, the models section and the
; uninstall section are all in plain view, and the build script passes every path in via /D
; (no relative-path guessing appears anywhere in the script).
;
; [Reinstall over an existing version] If .onInit finds an existing install on this machine,
; it points $INSTDIR back at the old directory and lets the user decide on the options page
; whether to delete the shared model library / caches; after "Install" is clicked, **this
; installer itself** removes the previous version (directories / plugins / uninstall entry /
; shortcuts) and never executes the old uninstall.exe — that exe belongs to the previous
; version, whether it is really silent under /S (language selection dialog, host-busy prompt)
; is not for us to decide, and executing it amounts to popping an extra "HiFiShifter
; uninstall" window in the middle of installation.
;
; [UI strings] Live elsewhere: they all come from $(HFS_*), whose values are extracted from
; the five-language string tables by scripts/release/build-installer-strings.mjs.
; To change wording, edit the string tables.
;
; [Three segments that must be preserved verbatim (all from the old plugin installer, all hard-won lessons)]
;   1. Host-running guard: checked on both install and uninstall (a DAW that has loaded the
;      plugin locks the DLL; overwrite = host crash);
;   2. Directory reconciliation: Vite chunks carry content hashes and File /r only writes,
;      never deletes, so reinstalls grow monotonically;
;   3. ARP registration: without it the product does not appear in Windows "Installed apps" at all.
;
; [Three segments ported from the Tauri template]
;   WebView2 runtime detection / install (embedded bootstrapper), main-app-running detection, shortcuts.

Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "nsDialogs.nsh"
!include "x64.nsh"
!include "Sections.nsh"
!include "StrFunc.nsh"
; The installer needs StrStr (process-name filtering) and StrTrimNewLines (to strip the
; trailing newline when reading a file size back).
; The uninstaller only needs StrStr (string functions in the uninstaller carry the un. prefix).
${StrStr}
${UnStrStr}
${StrTrimNewLines}

; ── Build-time parameters (all passed by scripts/build/build.ps1 via /D) ────
;
; [Why all are mandatory] With one parameter missing, File /r only reports "no files found" —
; one of the hardest build errors to locate. !error stops compilation in its first second and
; says exactly what is missing.
!ifndef APP_DIR
  !error "APP_DIR is required (staged standalone app payload)"
!endif
!ifndef VST3_DIR
  !error "VST3_DIR is required (contents of the assembled HiFiShifter.vst3)"
!endif
!ifndef CLAP_FILE
  !error "CLAP_FILE is required (assembled HiFiShifter.clap module)"
!endif
!ifndef CLAP_RES_DIR
  !error "CLAP_RES_DIR is required (staged HiFiShifter-resources: frontend + models.json)"
!endif
!ifndef MODELS_DIR
  !error "MODELS_DIR is required (full model weights)"
!endif
!ifndef MODELS_VERSION
  !error "MODELS_VERSION is required (content hash from models.json)"
!endif
!ifndef MODELS_VERIFY
  !error "MODELS_VERIFY is required (generated verification include)"
!endif
!ifndef HFS_LICENSE
  !error "HFS_LICENSE is required"
!endif
!ifndef HFS_LICENSE_PAGE
  !error "HFS_LICENSE_PAGE is required (UTF-8 with BOM)"
!endif
!ifndef HFS_LICENSES
  !error "HFS_LICENSES is required"
!endif
!ifndef OUTPUT_FILE
  !error "OUTPUT_FILE is required"
!endif
!ifndef APP_VERSION
  !error "APP_VERSION is required"
!endif
!ifndef WEBVIEW2_BOOTSTRAPPER
  !error "WEBVIEW2_BOOTSTRAPPER is required (MicrosoftEdgeWebview2Setup.exe)"
!endif
!ifndef HFS_ICON
  !error "HFS_ICON is required (HiFiShifter icon.ico from the Tauri app icons)"
!endif

; Component indexes: the custom page functions appear **before** the Section declarations,
; when the ${SEC_*} symbols NSIS auto-generates do not exist yet (tested: it reports
; "unknown variable/constant"), so explicit numbers are still used here. A SectionGroup also
; occupies an index itself, and its closing SectionGroupEnd occupies another one —
; the measured order is 0=group, 1/2/3/4=children, 5=group end, 6=group, 7/8=children.
; [Changing declaration order requires changing these numbers in step] A missed update does
; not fail compilation; it only makes the VST3/CLAP directory pages misjudge whether a
; component is checked — the class of regression that is hardest to notice in the UI.
!define SECGRP_APP 0
!define SEC_APP 1
!define SEC_MODELS 2
!define SEC_SHORTCUTS 3
!define SEC_FILEASSOC 4
; 5 = the first SectionGroupEnd
!define SECGRP_PLUGINS 6
!define SEC_VST3 7
!define SEC_CLAP 8


!define VST3_SUBDIR "HiFiShifter.vst3"
!define CLAP_FILENAME "HiFiShifter.clap"
!define CLAP_RESOURCES "HiFiShifter-resources"
; Stable ProgID for the file associations; extension default values point only at it,
; never directly at the exe path.
!define FILEASSOC_PROGID "HiFiShifter.Project"
!define ARP_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\HiFiShifter"
; Uninstall entry left behind by the old plugin installer: this installer takes over the
; model library and plugin directories and clears it after success.
!define LEGACY_VST3_ARP_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\HiFiShifterVST3"
!define UNINSTALLER "$INSTDIR\uninstall.exe"
; Product GUID of the WebView2 Evergreen Runtime (fixed by Microsoft, language-independent).
!define WEBVIEW2_GUID "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"

; ── Local testing switch (never defined for release builds) ────────────────
;
; [Why this exists] The installer really writes Program Files and HKLM; machines without
; admin rights cannot run it, and a DAW is often open (exactly what the installer refuses).
; Without this switch, the full loop "install → check ARP → reinstall (idempotent) → /repair
; → uninstall" cannot be exercised even once on a test machine, and an installer that has
; never been run is precisely the easiest thing to get wrong.
;
; When enabled: install into the current user's temp directory, no elevation required, ARP
; written to HKCU, host-running checks skipped.
; The rest of the logic (directory reconciliation, uninstall section, registry fields, pages
; and strings) is word-for-word identical to the release build.
; scripts/checks/check-product-consistency.ps1 asserts the default branch is still admin /
; HKLM / Common Files — so nobody mistakes the test branch for the default.
!ifdef HFS_TEST_MODE
  InstallDir "$LOCALAPPDATA\HiFiShifterInstallerTest\App"
  RequestExecutionLevel user
  !define HFS_ARP_ROOT HKCU
  ; Test mode writes HKCU\Software\Classes, so gate checks never pollute real users' HKLM associations.
  !define HFS_CLASSES_ROOT HKCU
  !define HFS_DEFAULT_VST3_DIR "$LOCALAPPDATA\HiFiShifterInstallerTest\VST3"
  !define HFS_DEFAULT_CLAP_DIR "$LOCALAPPDATA\HiFiShifterInstallerTest\CLAP"
  ; [Why test mode switches the model root] The test installer runs as user; writing
  ; %ProgramData% triggers UAC virtualization: File/SetOutPath lands in the VirtualStore while
  ; Rename/FileExists hit the real path — the models stay stuck in <version>.staging and both
  ; install and uninstall misjudge them. Using the current context's $APPDATA (the user's
  ; roaming profile) avoids virtualization, so the gate checks can truly exercise the model
  ; round-trip.
  !define HFS_MODEL_ROOT "$APPDATA\HiFiShifterTest\models"
  !define HFS_MODEL_ROOT_CONTEXT current
  ; [Where CLAP resources go] Per the spec, the public CLAP plugin directory
  ; (Common Files\CLAP) holds only the .clap files themselves; the resource directory cannot
  ; live there (see the CLAP section below). Test mode likewise uses the current user's
  ; directory to avoid UAC virtualization.
  !define HFS_CLAP_RES_DIR "$LOCALAPPDATA\HiFiShifterInstallerTest\CLAPResources"
  !define HFS_CLAP_RES_CONTEXT current
!else
  InstallDir "$PROGRAMFILES64\HiFiShifter"
  RequestExecutionLevel admin
  !define HFS_ARP_ROOT HKLM
  ; The release installer registers file associations per-machine.
  !define HFS_CLASSES_ROOT HKLM
  !define HFS_DEFAULT_VST3_DIR "$COMMONFILES64\VST3"
  !define HFS_DEFAULT_CLAP_DIR "$COMMONFILES64\CLAP"
  ; The release installer is admin; writes to %ProgramData% are not virtualized, and the all
  ; context's $APPDATA is exactly that directory.
  !define HFS_MODEL_ROOT "$APPDATA\HiFiShifter\models"
  !define HFS_MODEL_ROOT_CONTEXT all
  ; Release installer: plugin resources share the machine-wide root with the shared model
  ; library (%ProgramData%\HiFiShifter), so the public CLAP directory ends up containing only
  ; one entry, HiFiShifter.clap.
  !define HFS_CLAP_RES_DIR "$APPDATA\HiFiShifter\plugin-resources"
  !define HFS_CLAP_RES_CONTEXT all
  !define HFS_HOST_GUARD
!endif

; SetCompressor must come before the language declarations: as soon as a language file is
; loaded, data is already being compressed.
OutFile "${OUTPUT_FILE}"
SetCompressor /SOLID lzma
ShowInstDetails show
!define MUI_ABORTWARNING

; [Why the installer uses its own icon] MUI2 defaults to NSIS's built-in modern-install.ico —
; the product-agnostic round logo in the top-right corner. HiFiShifter's icon is maintained
; together with the Tauri App at backend\src-tauri\icons\icon.ico and passed in by the build
; script via /D (like every other path, no relative location is guessed in the script).
; Both installer and uninstaller switch to it, so the wizard is recognizable as this product
; the moment it opens.
!define MUI_ICON "${HFS_ICON}"
!define MUI_UNICON "${HFS_ICON}"

; [Why these strings must be defined before the page macros] MUI bakes the page strings into
; the UI functions at !insertmacro MUI_PAGE_* time; $(HFS_...) is a forward reference to a
; LangString whose value is resolved at runtime, so defining them before the language
; declarations is fine (verified with a minimal script). Previously they were not defined,
; and the welcome, components and directory pages all fell back to MUI's built-in English
; defaults — the strings from the five-language tables were never displayed.
!define MUI_WELCOMEPAGE_TITLE "$(HFS_WELCOME_TITLE)"
!define MUI_WELCOMEPAGE_TEXT "$(HFS_WELCOME_TEXT)"
!define MUI_COMPONENTSPAGE_TEXT_TOP "$(HFS_COMPONENTS_TOP)"
!define MUI_DIRECTORYPAGE_TEXT_TOP "$(HFS_APP_DIR_TEXT)"

; ── Pages ───────────────────────────────────────────────────────────────
; Pages of both installer and uninstaller must come before the language declarations
; (MUI inserts the UI functions when the first language is loaded, and at that moment it
; needs to know whether an uninstaller exists).
!insertmacro MUI_PAGE_WELCOME
; [Why reinstalling gets its own page] When reinstalling the same product, the old version
; must be removed before installing, otherwise directories end up mixing old and new files.
; This page matches the uninstall options page: the shared model library (checked by default)
; and caches/logs (unchecked by default) can be selected separately, and **it is the user's
; one and only confirmation** — no extra "Do you want to uninstall?" dialog. The actual
; removal happens in the INSTFILES page's PRE callback, completed in-process by the installer
; itself (it never runs the old uninstall.exe), so the old version stays intact if the user
; cancels midway.
; When no previous version is detected, this page Aborts in its Create callback and the
; wizard is completely unaffected.
Page custom HfsReinstallPageCreate HfsReinstallPageLeave

; [Why the license page does not show the bundled LICENSE] NSIS's LicenseData reads the file
; into the installer as UTF-16 **at compile time**: with a BOM it decodes as UTF-8, without
; one it decodes using **the ANSI code page of the machine doing the compiling**. LICENSE is
; UTF-8 without BOM, so in installers compiled on Chinese Windows θ turned into the hanzi
; for "stomach" and entire runs of Japanese kana went missing — silently at compile time,
; visible only when the license page opened.
; So build.ps1 writes a separate BOM'd license-page.txt dedicated to this page; HFS_LICENSE
; remains the byte-identical copy from the repository, the one installed onto the user's disk.
!insertmacro MUI_PAGE_LICENSE "${HFS_LICENSE_PAGE}"
!define MUI_PAGE_CUSTOMFUNCTION_SHOW HfsComponentsPageShow
; ── Components page layout: reflow by client size, tall text rows ───
;
; [Why this must change] MUI2's built-in IDD_SELCOM squeezes the tree into the left half
; (147px), truncating CJK component names; an earlier reflow with a hardcoded 434x210
; MoveWindow squeezed "Select the components to install" and "Space required" down to 10px
; high, clipping the text, and jammed the description title onto the same line as space
; required. Fixed pixels also mismatch at 125% DPI, so everything here is computed from the
; **real client area**.
;
; [Why not SMALLDESC / a custom UI resource] SMALLDESC crushes the tree to 5 rows and the
; bottom two components become invisible; patching the dialog template inside the .exe means
; maintaining a binary resource. Control IDs match MUI2's GetDlgItem
; (1006/1022/1023/1032/1042/1043):
;   description(28) → list label(14) → components tree (the bulk of the remaining height) →
;   space required (own row, 14) → description title (own row, 14) → description band (30~48)
; The "Installation type" dropdown (1017) and its label (1021) are unused by this installer
; and hidden to free up the full row width.
; 14px comes from the measured line height of the default 8pt font; 10px is exactly why the
; old version clipped the text.
Function HfsComponentsPageShow
  FindWindow $0 "#32770" "" $HWNDPARENT
  ${If} $0 == 0
    Return
  ${EndIf}
  ; $R0 = content width after leaving 8px on each side; $R1 = usable height after removing
  ; the top/bottom margins.
  System::Call 'user32::GetClientRect(p $0, @r1)'
  System::Call '*$1(i,i,i.r2,i.r3)'
  IntOp $R0 $2 - 16
  IntOp $R1 $3 - 6
  ; Fixed usage: description 28 + label 14 + space required 14 + description title 14 +
  ; five row gaps 10 = 80.
  IntOp $R2 $R1 - 80      ; total height to split between the tree and the description box
  IntOp $R4 $R2 / 5       ; description box takes 20% first
  ${If} $R4 < 30
    StrCpy $R4 30
  ${EndIf}
  ${If} $R4 > 48
    StrCpy $R4 48
  ${EndIf}
  IntOp $R3 $R2 - $R4     ; the rest all goes to the components tree (about 112px / 7 rows at a 450x228 client area)
  ${If} $R3 < 70
    StrCpy $R3 70
    IntOp $R4 $R2 - $R3
  ${EndIf}

  GetDlgItem $1 $0 1006
  System::Call 'user32::MoveWindow(p $1, i 8, i 2, i $R0, i 28, i 1)'
  ; Installation type dropdown and label: this installer has no InstType; hiding them keeps
  ; the label row from being squeezed narrower by them.
  GetDlgItem $1 $0 1017
  System::Call 'user32::ShowWindow(p $1, i 0)'
  GetDlgItem $1 $0 1021
  System::Call 'user32::ShowWindow(p $1, i 0)'
  GetDlgItem $1 $0 1022
  System::Call 'user32::MoveWindow(p $1, i 8, i 32, i $R0, i 14, i 1)'
  GetDlgItem $1 $0 1032
  System::Call 'user32::MoveWindow(p $1, i 8, i 48, i $R0, i $R3, i 1)'

  IntOp $R5 48 + $R3
  IntOp $R5 $R5 + 2
  ; Space required gets its own row, no longer side by side with the "Description" title.
  GetDlgItem $1 $0 1023
  System::Call 'user32::MoveWindow(p $1, i 8, i $R5, i $R0, i 14, i 1)'
  IntOp $R5 $R5 + 16
  GetDlgItem $1 $0 1042
  System::Call 'user32::MoveWindow(p $1, i 8, i $R5, i $R0, i 14, i 1)'
  IntOp $R5 $R5 + 14
  GetDlgItem $1 $0 1043
  System::Call 'user32::MoveWindow(p $1, i 8, i $R5, i $R0, i $R4, i 1)'
FunctionEnd

!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
Page custom HfsVst3Page HfsVst3PageLeave
Page custom HfsClapPage HfsClapPageLeave
!define MUI_PAGE_CUSTOMFUNCTION_PRE HfsInstFilesPagePre
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
; [Why UninstPage custom instead of MUI's components page] The uninstaller only needs two
; checkboxes; reusing the installer-side components page would drag in the whole set of
; section indexes plus the page-layout callback. A custom page with two checkboxes is
; lighter, and the defaults in un.onInit take effect directly on silent uninstall without
; any page; reinstalling over an existing version also passes the installer options page's
; choices through /DeleteModels= /DeleteCache=.
UninstPage custom un.HfsUninstallOptionsCreate un.HfsUninstallOptionsLeave
!insertmacro MUI_UNPAGE_INSTFILES

; The first one is the default language (the fallback when the system language matches none).
!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "SimpChinese"
!insertmacro MUI_LANGUAGE "TradChinese"
!insertmacro MUI_LANGUAGE "Japanese"
!insertmacro MUI_LANGUAGE "Korean"

!include "installer\installer_strings.nsh"

Name "$(HFS_NAME)"

Var HfsVst3Dialog
Var HfsVst3Input
Var HfsVst3Dir
Var HfsClapDialog
Var HfsClapInput
Var HfsClapDir
; Plugin resource directory (recorded at install time, deleted against it at uninstall): the
; new layout keeps it in %ProgramData%, sharing the root with the shared model library.
Var HfsClapResDir
Var HfsIsRepair
Var HfsModelRoot
Var HfsVerifyRoot
Var HfsListFile
; Uninstall options page: state of the two checkboxes (1 = clean up). un.onInit sets "delete
; models" to 1 and "clean caches" to 0; silent uninstall skips the options page, so these two
; values take effect directly.
Var HfsUnOptModels
Var HfsUnOptCache
Var HfsUnChkModels
Var HfsUnChkCache
Var HfsUnOtherInstall
; Reinstall over an existing version: old install directory, model library root recorded by
; the old version, the two cleanup choices, and whether a previous version was really detected.
; [Deliberately not saving the uninstall.exe path] Reinstall never runs the old uninstaller
; (see HfsRemovePreviousInstall); storing the path would only dangle the temptation of a
; "quick ExecWait" — exactly the pattern that pops a second window.
Var HfsOldInstallDir
; [Why the old model root is recorded too] The shared library's location has changed across
; installer versions (in the install directory during the Tauri era, now in %ProgramData%).
; Deleting/keeping only by this version's constant would leave behind a ghost copy of
; the old version's weights.
Var HfsOldModelRoot
Var HfsReinstallRemoveModels
Var HfsReinstallRemoveCache
Var HfsHasPrevious
Var HfsReinstallChkModels
Var HfsReinstallChkCache
Var HfsReinstallDialog

; ── VST3 directory page ─────────────────────────────────────────────────
; [Why a separate page] The App and the plugins install into two completely different places
; (Program Files and Common Files\VST3). A single directory page would leave the other half
; in a place the user never expected.
; When VST3 is unchecked on the components page, this page is skipped entirely.
Function HfsVst3Page
  SectionGetFlags ${SEC_VST3} $0
  IntOp $0 $0 & ${SF_SELECTED}
  ${If} $0 == 0
    Abort
  ${EndIf}
  !insertmacro MUI_HEADER_TEXT "$(HFS_COMPONENT_VST3)" "$(HFS_DIR_TEXT)"
  nsDialogs::Create 1018
  Pop $HfsVst3Dialog
  ${If} $HfsVst3Dialog == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 24u "$(HFS_DIR_TEXT)"
  Pop $0
  ${NSD_CreateDirRequest} 0 26u 100% 12u "$HfsVst3Dir"
  Pop $HfsVst3Input
  nsDialogs::Show
FunctionEnd

Function HfsClapPage
  SectionGetFlags ${SEC_CLAP} $0
  IntOp $0 $0 & ${SF_SELECTED}
  ${If} $0 == 0
    Abort
  ${EndIf}
  !insertmacro MUI_HEADER_TEXT "$(HFS_COMPONENT_CLAP)" "$(HFS_CLAP_DIR_TEXT)"
  nsDialogs::Create 1018
  Pop $HfsClapDialog
  ${If} $HfsClapDialog == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 24u "$(HFS_CLAP_DIR_TEXT)"
  Pop $0
  ${NSD_CreateDirRequest} 0 26u 100% 12u "$HfsClapDir"
  Pop $HfsClapInput
  nsDialogs::Show
FunctionEnd

Function HfsClapPageLeave
  ${NSD_GetText} $HfsClapInput $HfsClapDir
  ${If} $HfsClapDir == ""
    StrCpy $HfsClapDir "${HFS_DEFAULT_CLAP_DIR}"
  ${EndIf}
FunctionEnd

Function HfsVst3PageLeave
  ${NSD_GetText} $HfsVst3Input $HfsVst3Dir
  ${If} $HfsVst3Dir == ""
    StrCpy $HfsVst3Dir "${HFS_DEFAULT_VST3_DIR}"
  ${EndIf}
FunctionEnd

; ── Host-running check (run on both install and uninstall) ─────────────
;
; [Why processes are no longer enumerated by DAW name] This project aims to work with every
; host that supports VST3 / CLAP; hardcoding one DAW's process name tells every other user
; "this warning does not concern you" and also misses DAWs that run plugins in a separate
; host process. Instead the check is **by module name**: whoever has loaded this plugin's DLL
; is the host that must exit first, regardless of what it is called.
;
; [Why Sysnative] NSIS is a 32-bit process; calling tasklist.exe directly gets WOW64-
; redirected to the 32-bit copy in SysWOW64, and 32-bit tasklist cannot list the modules
; loaded by 64-bit DAWs — the check would silently fail. Sysnative is WOW64's virtual alias
; that lets a 32-bit process reach the native System32 (it does not exist on 32-bit Windows,
; hence the file-existence test below before falling back to System32).
;
; [Why /FI instead of piping the output to findstr] nsExec::ExecToStack keeps only the first
; 1023 bytes of output, which truncates as soon as several processes match; /FI makes
; tasklist filter by itself so the output is only the hit line, and it does not depend on the
; localized "not found" text. The three module names cover both plugin formats:
; HiFiShifter.vst3 is the VST3 shell loader, HiFiShifter.clap is the CLAP module, and
; HiFiShifterEngine.dll is the core that both load (some hosts list only the latter).
;
; Result goes into $0: 1 = a host is still loading this plugin.
Function HfsHostRunning
  StrCpy $0 0
  StrCpy $R6 "$WINDIR\Sysnative\tasklist.exe"
  ${IfNot} ${FileExists} "$R6"
    StrCpy $R6 "$SYSDIR\tasklist.exe"
  ${EndIf}
  StrCpy $R7 "HiFiShifter.vst3"
  Call HfsHostModuleProbe
  ${If} $0 == 0
    StrCpy $R7 "HiFiShifter.clap"
    Call HfsHostModuleProbe
  ${EndIf}
  ${If} $0 == 0
    StrCpy $R7 "HiFiShifterEngine.dll"
    Call HfsHostModuleProbe
  ${EndIf}
FunctionEnd

; $R7 = module name, $R6 = tasklist path; on a hit, $0 is set to 1.
Function HfsHostModuleProbe
  nsExec::ExecToStack /TIMEOUT=20000 '"$R6" /FI "MODULES eq $R7" /FO CSV /NH'
  Pop $1
  Pop $2
  ${StrStr} $3 $2 '"'
  ${If} $3 != ""
    StrCpy $0 1
  ${EndIf}
FunctionEnd

; The same check for the uninstaller (string functions there carry the un. prefix, so a
; separate copy is required).
Function un.HfsHostRunning
  StrCpy $0 0
  StrCpy $R6 "$WINDIR\Sysnative\tasklist.exe"
  ${IfNot} ${FileExists} "$R6"
    StrCpy $R6 "$SYSDIR\tasklist.exe"
  ${EndIf}
  StrCpy $R7 "HiFiShifter.vst3"
  Call un.HfsHostModuleProbe
  ${If} $0 == 0
    StrCpy $R7 "HiFiShifter.clap"
    Call un.HfsHostModuleProbe
  ${EndIf}
  ${If} $0 == 0
    StrCpy $R7 "HiFiShifterEngine.dll"
    Call un.HfsHostModuleProbe
  ${EndIf}
FunctionEnd

Function un.HfsHostModuleProbe
  nsExec::ExecToStack /TIMEOUT=20000 '"$R6" /FI "MODULES eq $R7" /FO CSV /NH'
  Pop $1
  Pop $2
  ${UnStrStr} $3 $2 '"'
  ${If} $3 != ""
    StrCpy $0 1
  ${EndIf}
FunctionEnd


; ── File associations: uninstall unbinds only this install ─────────────
;
; Mirrors the installer's same-named function; string functions there carry the un. prefix,
; so a separate copy is required.
; Every deletion first confirms the ProgID / Applications open command points at $INSTDIR,
; so the default associations of other installs or other programs are never removed by
; mistake.
Function un.HfsRemoveFileAssociations
  ReadRegStr $R1 ${HFS_CLASSES_ROOT} "Software\Classes\${FILEASSOC_PROGID}\shell\open\command" ""
  StrCpy $R2 ""
  ${UnStrStr} $R2 $R1 "$INSTDIR\HiFiShifter.exe"
  StrCpy $R3 0
  ${If} $R2 != ""
    StrCpy $R3 1
  ${EndIf}

  ReadRegStr $R0 ${HFS_CLASSES_ROOT} "Software\Classes\.hshp" ""
  ${If} $R0 == "${FILEASSOC_PROGID}"
  ${AndIf} $R3 == 1
    DeleteRegValue ${HFS_CLASSES_ROOT} "Software\Classes\.hshp" ""
  ${EndIf}
  ${If} $R3 == 1
    DeleteRegValue ${HFS_CLASSES_ROOT} "Software\Classes\.hshp\OpenWithProgids" "${FILEASSOC_PROGID}"
  ${EndIf}

  ReadRegStr $R0 ${HFS_CLASSES_ROOT} "Software\Classes\.hsp" ""
  ${If} $R0 == "${FILEASSOC_PROGID}"
  ${AndIf} $R3 == 1
    DeleteRegValue ${HFS_CLASSES_ROOT} "Software\Classes\.hsp" ""
  ${EndIf}
  ${If} $R3 == 1
    DeleteRegValue ${HFS_CLASSES_ROOT} "Software\Classes\.hsp\OpenWithProgids" "${FILEASSOC_PROGID}"
  ${EndIf}

  ${If} $R3 == 1
    DeleteRegKey ${HFS_CLASSES_ROOT} "Software\Classes\${FILEASSOC_PROGID}"
  ${EndIf}

  ReadRegStr $R1 ${HFS_CLASSES_ROOT} "Software\Classes\Applications\HiFiShifter.exe\shell\open\command" ""
  StrCpy $R2 ""
  ${UnStrStr} $R2 $R1 "$INSTDIR\HiFiShifter.exe"
  ${If} $R2 != ""
    DeleteRegKey ${HFS_CLASSES_ROOT} "Software\Classes\Applications\HiFiShifter.exe"
  ${EndIf}

  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
FunctionEnd

; ── Main app running check (ported from the Tauri template) ───────────────
;
; [Why /FI instead of piping tasklist output to findstr] Same as above: only one line is
; taken, with no dependence on localized text.
Function HfsAppRunning
  StrCpy $0 0
  nsExec::ExecToStack /TIMEOUT=10000 '"$SYSDIR\tasklist.exe" /FI "IMAGENAME eq HiFiShifter.exe" /FO CSV /NH'
  Pop $1
  Pop $2
  ${StrStr} $3 $2 '"HiFiShifter.exe"'
  ${If} $3 != ""
    StrCpy $0 1
  ${EndIf}
FunctionEnd

; ── WebView2 runtime (embedBootstrapper branch of the Tauri template) ─────
;
; [Why the installer must handle this itself] On Windows 10 without WebView2, the app opens
; to a blank window right after install — no error dialog at all, and the user only says
; "your software won't open". Windows 11 ships the runtime; Windows 10 may not.
Function HfsEnsureWebView2
  SetRegView 64
  ; Three registry locations: per-machine (WOW6432Node), per-user, and per-machine in the
  ; 64-bit view.
  ReadRegStr $0 HKLM "SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\${WEBVIEW2_GUID}" "pv"
  ${If} $0 != ""
  ${AndIf} $0 != "0.0.0.0"
    Return
  ${EndIf}
  ReadRegStr $0 HKCU "Software\Microsoft\EdgeUpdate\Clients\${WEBVIEW2_GUID}" "pv"
  ${If} $0 != ""
  ${AndIf} $0 != "0.0.0.0"
    Return
  ${EndIf}
  ReadRegStr $0 HKLM "SOFTWARE\Microsoft\EdgeUpdate\Clients\${WEBVIEW2_GUID}" "pv"
  ${If} $0 != ""
  ${AndIf} $0 != "0.0.0.0"
    Return
  ${EndIf}

  DetailPrint "Installing the Microsoft Edge WebView2 runtime..."
  SetOutPath "$PLUGINSDIR"
  File "/oname=$PLUGINSDIR\MicrosoftEdgeWebview2Setup.exe" "${WEBVIEW2_BOOTSTRAPPER}"
  nsExec::ExecToLog '"$PLUGINSDIR\MicrosoftEdgeWebview2Setup.exe" /silent /install'
  Pop $0
  SetOutPath "$INSTDIR"

  ReadRegStr $0 HKLM "SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\${WEBVIEW2_GUID}" "pv"
  ${If} $0 != ""
  ${AndIf} $0 != "0.0.0.0"
    Return
  ${EndIf}
  ReadRegStr $0 HKCU "Software\Microsoft\EdgeUpdate\Clients\${WEBVIEW2_GUID}" "pv"
  ${If} $0 == ""
  ${OrIf} $0 == "0.0.0.0"
    ${IfNot} ${Silent}
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_WEBVIEW2_INSTALL_FAILED)" /SD IDOK
    ${EndIf}
  ${EndIf}
FunctionEnd


; ── File associations (optional component) ─────────────────────────────
;
; [Why a ProgID instead of writing the exe path directly] Extension default values must point
; at a stable ProgID; writing the exe directly makes the “Default apps” and “Open with” UIs
; read different shapes. Windows 10/11 UserChoice is system-protected and the installer
; cannot forge the hash; only the ProgID, the extension default values, OpenWithProgids and
; Applications are registered here. Without a UserChoice the system adopts our ProgID; with
; one already set, this program at least appears in the “Open with” list.
Function HfsWriteFileAssociations
  WriteRegStr ${HFS_CLASSES_ROOT} "Software\Classes\${FILEASSOC_PROGID}" "" "$(HFS_FILEASSOC_TYPE_NAME)"
  WriteRegStr ${HFS_CLASSES_ROOT} "Software\Classes\${FILEASSOC_PROGID}\DefaultIcon" "" '"$INSTDIR\HiFiShifter.exe",0'
  WriteRegStr ${HFS_CLASSES_ROOT} "Software\Classes\${FILEASSOC_PROGID}\shell\open\command" "" '"$INSTDIR\HiFiShifter.exe" "%1"'

  WriteRegStr ${HFS_CLASSES_ROOT} "Software\Classes\Applications\HiFiShifter.exe" "FriendlyAppName" "$(HFS_NAME)"
  WriteRegStr ${HFS_CLASSES_ROOT} "Software\Classes\Applications\HiFiShifter.exe\shell\open\command" "" '"$INSTDIR\HiFiShifter.exe" "%1"'
  WriteRegStr ${HFS_CLASSES_ROOT} "Software\Classes\Applications\HiFiShifter.exe\SupportedTypes" ".hshp" ""
  WriteRegStr ${HFS_CLASSES_ROOT} "Software\Classes\Applications\HiFiShifter.exe\SupportedTypes" ".hsp" ""

  WriteRegStr ${HFS_CLASSES_ROOT} "Software\Classes\.hshp" "" "${FILEASSOC_PROGID}"
  WriteRegStr ${HFS_CLASSES_ROOT} "Software\Classes\.hshp\OpenWithProgids" "${FILEASSOC_PROGID}" ""
  WriteRegStr ${HFS_CLASSES_ROOT} "Software\Classes\.hsp" "" "${FILEASSOC_PROGID}"
  WriteRegStr ${HFS_CLASSES_ROOT} "Software\Classes\.hsp\OpenWithProgids" "${FILEASSOC_PROGID}" ""

  ; Notify Explorer to refresh its icon / "Open with" caches.
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
FunctionEnd

; Reads the ProgID's open command and decides whether it points at **this install**.
; Result goes into $0 (1 = yes).
; Both uninstall and unchecking the component must pass through this: another install may
; still be using the same ProgID.
Function HfsFileAssocCommandIsOurs
  ReadRegStr $R1 ${HFS_CLASSES_ROOT} "Software\Classes\${FILEASSOC_PROGID}\shell\open\command" ""
  StrCpy $R2 ""
  ${StrStr} $R2 $R1 "$INSTDIR\HiFiShifter.exe"
  ${If} $R2 != ""
    StrCpy $0 1
  ${Else}
    StrCpy $0 0
  ${EndIf}
FunctionEnd

; Removes bindings only when the current one truly points at this install, so uninstalling
; never deletes another program's associations by mistake.
Function HfsRemoveFileAssociationsIfOurs
  Call HfsFileAssocCommandIsOurs

  ; The default ProgIDs of .hshp / .hsp are cleared only when the command truly points at
  ; this install; when the default points at another program it is left as is. This ProgID's
  ; value under OpenWithProgids is removed only after confirming this install owns the
  ; ProgID, so another install's “Open with” entry is never deleted.
  ReadRegStr $R0 ${HFS_CLASSES_ROOT} "Software\Classes\.hshp" ""
  ${If} $R0 == "${FILEASSOC_PROGID}"
  ${AndIf} $0 == 1
    DeleteRegValue ${HFS_CLASSES_ROOT} "Software\Classes\.hshp" ""
  ${EndIf}
  ${If} $0 == 1
    DeleteRegValue ${HFS_CLASSES_ROOT} "Software\Classes\.hshp\OpenWithProgids" "${FILEASSOC_PROGID}"
  ${EndIf}

  ReadRegStr $R0 ${HFS_CLASSES_ROOT} "Software\Classes\.hsp" ""
  ${If} $R0 == "${FILEASSOC_PROGID}"
  ${AndIf} $0 == 1
    DeleteRegValue ${HFS_CLASSES_ROOT} "Software\Classes\.hsp" ""
  ${EndIf}
  ${If} $0 == 1
    DeleteRegValue ${HFS_CLASSES_ROOT} "Software\Classes\.hsp\OpenWithProgids" "${FILEASSOC_PROGID}"
  ${EndIf}

  ; The ProgID is deleted only when the command truly points at this install; another install
  ; may still be using the same ProgID.
  ${If} $0 == 1
    DeleteRegKey ${HFS_CLASSES_ROOT} "Software\Classes\${FILEASSOC_PROGID}"
  ${EndIf}

  ; Same for the Applications registration: kept when the command points at another install.
  ReadRegStr $R1 ${HFS_CLASSES_ROOT} "Software\Classes\Applications\HiFiShifter.exe\shell\open\command" ""
  StrCpy $R2 ""
  ${StrStr} $R2 $R1 "$INSTDIR\HiFiShifter.exe"
  ${If} $R2 != ""
    DeleteRegKey ${HFS_CLASSES_ROOT} "Software\Classes\Applications\HiFiShifter.exe"
  ${EndIf}

  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
FunctionEnd

; ── Model library: integrity check (mirrors runtime looks_complete) ────────
;
; [Why validate "the byte size of every file in the manifest"] The runtime's
; model_store::looks_complete judges exactly this way. If the installer used a different rule
; (say, only checking that the directory exists), the result would be "the installer says
; installed, the runtime says unavailable" — all the user sees is "inference unavailable"
; while the model directory looks perfectly normal.
; The expected values are generated by build.ps1 from models.json into MODELS_VERIFY
; (one line per file).
Function HfsCheckModelFile
  ; $R0 = relative path, $R1 = expected byte count, $R2 = 1 (still complete so far)
  StrCpy $R3 "$HfsVerifyRoot\$R0"
  ${IfNot} ${FileExists} "$R3"
    StrCpy $R2 0
    Return
  ${EndIf}
  nsExec::ExecToStack /TIMEOUT=10000 'cmd /c for %I in ("$R3") do @echo %~zI'
  Pop $R4
  Pop $R5
  ${StrTrimNewLines} $R5 $R5
  ${If} $R5 != "$R1"
    StrCpy $R2 0
  ${EndIf}
FunctionEnd

; Validates $HfsVerifyRoot; result goes into $0 (1 = complete).
Function HfsVerifyModels
  StrCpy $R2 1
  Call HfsVerifyModelFiles
  StrCpy $0 $R2
FunctionEnd

!include "${MODELS_VERIFY}"

; ── Model library: GC (keep the current version and one other) ─────────────
;
; [Policy] Keep only **the current version** and **the most recently modified other version**
; (normally the previous one): one extra directory is what covers the real-world "roll back
; to the previous App version" scenario; and since version numbers are content hashes that
; cannot be sorted, the first other version is taken from dir /o-d's modification-time
; ordering. Every other directory is deleted with a DetailPrint.
Function HfsModelsGc
  StrCpy $HfsListFile "$TEMP\hfs_models_gc.txt"
  nsExec::ExecToStack /TIMEOUT=10000 'cmd /c dir /b /ad /o-d "$HfsModelRoot" > "$HfsListFile" 2>nul'
  Pop $0
  Pop $1
  ${IfNot} ${FileExists} "$HfsListFile"
    Return
  ${EndIf}

  StrCpy $R0 ""   ; the kept "other version"
  FileOpen $R1 "$HfsListFile" r
  ${Do}
    FileRead $R1 $R2
    ${If} $R2 == ""
      ${ExitDo}
    ${EndIf}
    ${StrTrimNewLines} $R2 $R2
    ${If} $R2 == ""
      ${Continue}
    ${EndIf}
    ${If} $R2 == "${MODELS_VERSION}"
      ${Continue}
    ${EndIf}
    ${If} $R0 == ""
      StrCpy $R0 $R2
    ${EndIf}
  ${Loop}
  FileClose $R1

  nsExec::ExecToStack /TIMEOUT=10000 'cmd /c dir /b /ad /o-d "$HfsModelRoot" > "$HfsListFile" 2>nul'
  Pop $0
  Pop $1
  FileOpen $R1 "$HfsListFile" r
  ${Do}
    FileRead $R1 $R2
    ${If} $R2 == ""
      ${ExitDo}
    ${EndIf}
    ${StrTrimNewLines} $R2 $R2
    ${If} $R2 == ""
      ${Continue}
    ${EndIf}
    ${If} $R2 == "${MODELS_VERSION}"
      ${Continue}
    ${EndIf}
    ${If} $R2 == "$R0"
      ${Continue}
    ${EndIf}
    DetailPrint "Removing old model library $R2"
    RMDir /r "$HfsModelRoot\$R2"
  ${Loop}
  FileClose $R1
  Delete "$HfsListFile"
FunctionEnd

; ── Reinstall: read back the old install dir, remove it in place ───────
;
; [Why the old uninstall.exe is no longer executed] That exe is a binary built by the
; **previous version**; whether it shows any UI under /S is not up to this script: the
; Tauri-era uninstaller calls !insertmacro MUI_UNGETLANGUAGE in un.onInit, so the language
; selection dialog pops up even in silent mode, and its "host still has the plugin loaded"
; prompt is a MessageBox without /SD that shows under /S as well. So after the user clicks
; "Install", a window titled "HiFiShifter uninstall" suddenly appears on the desktop —
; exactly the thing to eliminate for good. A child process's UI can neither be changed nor
; blocked; the only reliable answer is **not to run it**: whatever uninstall has to do, this
; installer does itself in-process (see HfsRemovePreviousInstall). The reinstall path
; therefore no longer has a second window, nor a second uninstaller process.
Function HfsDetectPreviousInstall
  StrCpy $HfsHasPrevious 0
  StrCpy $HfsOldInstallDir ""
  ; Same defaults as the uninstall options page: delete models=1, clean caches=0.
  ; The two switches let silent/unattended upgrades express a choice; interactive installs
  ; are overridden by the options page.
  StrCpy $HfsReinstallRemoveModels 1
  StrCpy $HfsReinstallRemoveCache 0
  ${StrStr} $0 $CMDLINE "/DeleteModels=0"
  ${If} $0 != ""
    StrCpy $HfsReinstallRemoveModels 0
  ${EndIf}
  ${StrStr} $0 $CMDLINE "/DeleteModels=1"
  ${If} $0 != ""
    StrCpy $HfsReinstallRemoveModels 1
  ${EndIf}
  ${StrStr} $0 $CMDLINE "/DeleteCache=0"
  ${If} $0 != ""
    StrCpy $HfsReinstallRemoveCache 0
  ${EndIf}
  ${StrStr} $0 $CMDLINE "/DeleteCache=1"
  ${If} $0 != ""
    StrCpy $HfsReinstallRemoveCache 1
  ${EndIf}
  ${If} $HfsIsRepair == 0
    ReadRegStr $HfsOldInstallDir ${HFS_ARP_ROOT} "${ARP_KEY}" "InstallLocation"
    ; Where the old version wrote the shared library can only be asked from the record it
    ; left behind; only if unreadable do we fall back to this version's constant.
    ReadRegStr $HfsOldModelRoot ${HFS_ARP_ROOT} "${ARP_KEY}" "ModelRoot"
    ${If} $HfsOldInstallDir != ""
      ; [Why HiFiShifter.exe is the criterion, not the existence of the uninstall entry]
      ; InstallLocation may be a path the user edited by hand, or an empty directory left by a
      ; failed install. "The previous version is still here" only holds if the main executable
      ; really exists in the directory; otherwise treat this as a fresh install.
      ${If} ${FileExists} "$HfsOldInstallDir\HiFiShifter.exe"
        StrCpy $HfsHasPrevious 1
        ; [Why the directory is set before removal] MUI's directory page is created after
        ; .onInit, so changing $INSTDIR here means "try to preselect the old version's path";
        ; the user can still change it to somewhere else on the directory page.
        StrCpy $INSTDIR "$HfsOldInstallDir"
        ; [Why no confirmation dialog here] The "reinstall options" page right after this is
        ; the user's confirmation: the user already chose there whether to delete models and
        ; caches; one more "Do you want to uninstall?" would just add another window.
      ${Else}
        ; Only a leftover registry entry remains: clear it and treat this as a fresh install,
        ; otherwise the user stays stuck on "cannot uninstall the previous version" forever.
        DeleteRegKey ${HFS_ARP_ROOT} "${ARP_KEY}"
        StrCpy $HfsOldInstallDir ""
      ${EndIf}
    ${EndIf}
  ${EndIf}
  ; When the ARP key does not exist, ReadRegStr/DeleteRegKey set the global error flag; the
  ; later Rename/${If} ${Errors} would misread it as a model library publish failure. Clear it
  ; before returning; do not leak it into the install sections.
  ClearErrors
FunctionEnd

; The "uninstall" half of reinstalling: done entirely in this process, no child process started.
;
; The work matches the uninstall section:
;   1. The old App directory (the whole directory belongs to this product — same criterion as
;      the uninstall section's RMDir /r "$INSTDIR");
;   2. The VST3 / CLAP artifacts the old version registered (only this product's own
;      subdirectories are deleted; public plugin directories must not be wiped whole);
;   3. The ARP uninstall entry, the old plugin installer's leftover, Start menu and desktop
;      shortcuts;
;   4. The shared model library and regenerable caches the user checked on the options page
;      (reusing the two existing cleanup routines).
;
; [Why processes are not killed] Force-killing a running App during a silent upgrade loses the
; user's unsaved projects. Instead, "fail explicitly when the main app is still running" —
; consistent with the Tauri installer's behavior: make the user close it first.
; [Why the file is checked before deleting the whole directory] InstallLocation comes from
; this product's own ARP write, but the user may also have edited the registry by hand.
; RMDir /r is allowed only when this product's artifacts really exist in the directory.
Function HfsRemovePreviousInstall
  ${If} $HfsHasPrevious != 1
    Return
  ${EndIf}
  ${If} $HfsOldInstallDir == ""
    Return
  ${EndIf}

  Call HfsAppRunning
  ${If} $0 == 1
    ${IfNot} ${Silent}
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_APP_RUNNING)" /SD IDOK
    ${EndIf}
    SetErrorLevel 3
    Quit
  ${EndIf}

  DetailPrint "$(HFS_REINSTALL_UNINSTALLING)"
  ; First record the plugin directories the old version registered: the user may have pointed
  ; the new version elsewhere, and deleting only the new location would leave the old plugins
  ; in place.
  ReadRegStr $R5 ${HFS_ARP_ROOT} "${ARP_KEY}" "Vst3Dir"
  ReadRegStr $R6 ${HFS_ARP_ROOT} "${ARP_KEY}" "ClapDir"
  ; The plugin resource directory is likewise cleared against the old version's registered
  ; value: the old version kept it next to the CLAP directory, the new one in the machine-wide
  ; shared directory.
  ReadRegStr $R8 ${HFS_ARP_ROOT} "${ARP_KEY}" "ClapResDir"

  ; Separate ${If} checks instead of If + OrIf: LogicLib's FileExists conditions are
  ; jump-based macros that mix badly with OrIf and can generate wrong branches; an explicit
  ; flag is the most robust here.
  StrCpy $R7 0
  ${If} ${FileExists} "$HfsOldInstallDir\HiFiShifter.exe"
    StrCpy $R7 1
  ${EndIf}
  ${If} ${FileExists} "$HfsOldInstallDir\uninstall.exe"
    StrCpy $R7 1
  ${EndIf}
  ${If} $R7 == 1
    RMDir /r "$HfsOldInstallDir"
  ${EndIf}

  ; Delete only this product's own VST3 subdirectory and CLAP module + resource directory:
  ; $R5/$R6 are public plugin directories holding other vendors' plugins; wiping them whole
  ; would amount to destroying other vendors' installs.
  ${If} $R5 != ""
    RMDir /r "$R5\${VST3_SUBDIR}"
  ${EndIf}
  ${If} $R6 != ""
    Delete "$R6\${CLAP_FILENAME}"
    RMDir /r "$R6\${CLAP_RESOURCES}"
  ${EndIf}
  ${If} $R8 != ""
    RMDir /r "$R8"
  ${EndIf}

  DeleteRegKey ${HFS_ARP_ROOT} "${ARP_KEY}"
  DeleteRegKey HKLM "${LEGACY_VST3_ARP_KEY}"
  DeleteRegKey HKCU "${LEGACY_VST3_ARP_KEY}"

  SetShellVarContext all
  RMDir /r "$SMPROGRAMS\HiFiShifter"
  Delete "$DESKTOP\HiFiShifter.lnk"
  SetShellVarContext current

  ; If it cannot be deleted, the exe is still locked (a DAW, or the user opened it by hand):
  ; continuing would only produce a directory mixing old and new files, so prefer failing
  ; explicitly and let the user close whatever holds it first.
  ${If} ${FileExists} "$HfsOldInstallDir\HiFiShifter.exe"
    ${IfNot} ${Silent}
      MessageBox MB_OK|MB_ICONSTOP|MB_DEFBUTTON1 "$(HFS_REINSTALL_FAILED)" /SD IDOK
    ${EndIf}
    SetErrorLevel 2
    Quit
  ${EndIf}

  ; $HfsModelRoot was expanded per release/test mode in .onInit; the models section writes it
  ; again later.
  ${If} $HfsReinstallRemoveModels == 1
    ; The old version may have written the shared library under a different root (version
    ; differences): the same choice must apply to it as well, otherwise a full copy of the
    ; weights is still sitting on disk after "deleted the models".
    Call HfsPurgeLegacyModelRoot
    FindFirst $R3 $R4 "$HfsModelRoot\*"
    StrCmp $R4 "" hfs_prev_models_done
    hfs_prev_models_loop:
    StrCmp $R4 "." hfs_prev_models_next
    StrCmp $R4 ".." hfs_prev_models_next
    ${If} ${FileExists} "$HfsModelRoot\$R4\*"
      RMDir /r "$HfsModelRoot\$R4"
    ${Else}
      Delete "$HfsModelRoot\$R4"
    ${EndIf}
    hfs_prev_models_next:
    ClearErrors
    FindNext $R3 $R4
    IfErrors hfs_prev_models_done
    StrCmp $R4 "" hfs_prev_models_done hfs_prev_models_loop
    hfs_prev_models_done:
    FindClose $R3
    RMDir /r "$HfsModelRoot"
    RMDir "$HfsModelRoot\.."
  ${Else}
    ; The user chose to keep them: if the previous version wrote the library under another
    ; root, merge it into this version's root — otherwise "keep" only leaves a copy the new
    ; version will never read, saving the user neither disk space nor re-unpacking time.
    Call HfsAdoptLegacyModelRoot
  ${EndIf}
  ${If} $HfsReinstallRemoveCache == 1
    Call HfsSweepDisposableDirs
  ${EndIf}
FunctionEnd

; Does the old root really contain this product's model library (a version directory holding
; models.json)?
; [Why check] ModelRoot in the ARP entry is in theory written by this installer, but the user
; may have edited the registry by hand or imported an uninstall entry from somewhere else.
; Without this guard, a corrupted path would get a whole-directory RMDir /r.
; In: $R0 = root directory. Out: $R7 = 1 means it is this product's model library.
Function HfsLibraryHasVersions
  StrCpy $R7 0
  FindFirst $R3 $R4 "$R0\*"
  StrCmp $R4 "" hfs_lib_scan_end
  hfs_lib_scan_loop:
  StrCmp $R4 "." hfs_lib_scan_next
  StrCmp $R4 ".." hfs_lib_scan_next
  ${If} ${FileExists} "$R0\$R4\models.json"
    StrCpy $R7 1
    Goto hfs_lib_scan_end
  ${EndIf}
  hfs_lib_scan_next:
  ClearErrors
  FindNext $R3 $R4
  IfErrors hfs_lib_scan_end
  StrCmp $R4 "" hfs_lib_scan_end hfs_lib_scan_loop
  hfs_lib_scan_end:
  FindClose $R3
FunctionEnd

; When "delete models" is chosen, also clear the model root the previous version registered.
Function HfsPurgeLegacyModelRoot
  ${If} $HfsOldModelRoot == ""
    Return
  ${EndIf}
  ${If} $HfsOldModelRoot == $HfsModelRoot
    Return
  ${EndIf}
  ${IfNot} ${FileExists} "$HfsOldModelRoot\*"
    Return
  ${EndIf}
  StrCpy $R0 "$HfsOldModelRoot"
  Call HfsLibraryHasVersions
  ${If} $R7 != 1
    Return
  ${EndIf}
  DetailPrint "Removing the previous model library: $HfsOldModelRoot"
  RMDir /r "$HfsOldModelRoot"
FunctionEnd

; When "keep models" is chosen, merge the model root the previous version registered into this
; version's root.
;
; [Why move instead of leave] The runtime recognizes only one root (shared_store_dir). A copy
; left in the old root is never read — dead weight on disk; once moved over,
; HfsVerifyModels hits and even the unpacking step is saved.
Function HfsAdoptLegacyModelRoot
  ${If} $HfsOldModelRoot == ""
    Return
  ${EndIf}
  ${If} $HfsOldModelRoot == $HfsModelRoot
    Return
  ${EndIf}
  ${IfNot} ${FileExists} "$HfsOldModelRoot\*"
    Return
  ${EndIf}
  StrCpy $R0 "$HfsOldModelRoot"
  Call HfsLibraryHasVersions
  ${If} $R7 != 1
    Return
  ${EndIf}
  CreateDirectory "$HfsModelRoot"
  FindFirst $R3 $R4 "$HfsOldModelRoot\*"
  StrCmp $R4 "" hfs_adopt_done
  hfs_adopt_loop:
  StrCmp $R4 "." hfs_adopt_next
  StrCmp $R4 ".." hfs_adopt_next
  ${If} ${FileExists} "$HfsOldModelRoot\$R4\models.json"
    ${IfNot} ${FileExists} "$HfsModelRoot\$R4\models.json"
      DetailPrint "Moving the previous model library into place: $R4"
      ClearErrors
      Rename "$HfsOldModelRoot\$R4" "$HfsModelRoot\$R4"
      ${If} ${Errors}
        ; Rename fails across volumes: robocopy /MOVE does the same job, and nsExec guarantees
        ; there is no window.
        nsExec::ExecToStack '"$SYSDIR\robocopy.exe" "$HfsOldModelRoot\$R4" "$HfsModelRoot\$R4" /E /MOVE /NFL /NDL /NJH /NJS /NP'
        Pop $0
        Pop $1
      ${EndIf}
    ${Else}
      ; The new root already has this version: the old copy is a duplicate; deleting it is what
      ; "not storing the weights twice" means.
      RMDir /r "$HfsOldModelRoot\$R4"
    ${EndIf}
  ${Else}
    ; Leftover version-directory remains (*.staging / *.publishing / half-written directories):
    ; not merged in, just cleared.
    RMDir /r "$HfsOldModelRoot\$R4"
  ${EndIf}
  hfs_adopt_next:
  ClearErrors
  FindNext $R3 $R4
  IfErrors hfs_adopt_done
  StrCmp $R4 "" hfs_adopt_done hfs_adopt_loop
  hfs_adopt_done:
  FindClose $R3
  ; Deletion succeeds only when the old root is truly empty; if it fails, other things are
  ; still inside and are kept.
  RMDir "$HfsOldModelRoot"
FunctionEnd

; Silent installs have no pages, so removal happens directly in .onInit; interactive installs
; defer it until after the user clicks "Install".
; $HfsHasPrevious is cleared so the removal does not run twice (once from .onInit, once from
; the INSTFILES PRE callback).
Function HfsRemovePreviousInstallIfAny
  ${If} $HfsHasPrevious == 1
    Call HfsRemovePreviousInstall
    StrCpy $HfsHasPrevious 0
  ${EndIf}
FunctionEnd

; [Why this stays in step with the uninstall options page] Reinstalling over an existing
; version is essentially "uninstall the old version + install the new one", so what the user
; can clean up should be exactly the same: the shared model library (checked by default) and
; caches/logs (unchecked by default). The two strings reuse $(HFS_UNINSTALL_OPT_*) directly
; instead of building a second string table.
Function HfsReinstallPageCreate
  ${If} $HfsHasPrevious != 1
    Abort
  ${EndIf}
  !insertmacro MUI_HEADER_TEXT "$(HFS_REINSTALL_OPTIONS_TITLE)" "$(HFS_REINSTALL_OPTIONS_SUBTITLE)"
  nsDialogs::Create 1018
  Pop $HfsReinstallDialog
  ${If} $HfsReinstallDialog == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 24u "$(HFS_REINSTALL_OPTIONS_TOP)"
  Pop $0
  ${NSD_CreateCheckbox} 0 30u 100% 12u "$(HFS_UNINSTALL_OPT_MODELS)"
  Pop $HfsReinstallChkModels
  ${NSD_CreateLabel} 12u 44u 88% 20u "$(HFS_UNINSTALL_OPT_MODELS_DESC)"
  Pop $0
  ${NSD_CreateCheckbox} 0 72u 100% 12u "$(HFS_UNINSTALL_OPT_CACHE)"
  Pop $HfsReinstallChkCache
  ${NSD_CreateLabel} 12u 86u 88% 20u "$(HFS_UNINSTALL_OPT_CACHE_DESC)"
  Pop $0
  ${If} $HfsReinstallRemoveModels == 1
    ${NSD_Check} $HfsReinstallChkModels
  ${EndIf}
  ${If} $HfsReinstallRemoveCache == 1
    ${NSD_Check} $HfsReinstallChkCache
  ${EndIf}
  nsDialogs::Show
FunctionEnd

Function HfsReinstallPageLeave
  ${NSD_GetState} $HfsReinstallChkModels $HfsReinstallRemoveModels
  ${NSD_GetState} $HfsReinstallChkCache $HfsReinstallRemoveCache
FunctionEnd

; [Why removal happens in INSTFILES's PRE] Reaching this page means the user has clicked
; "Install"; cancelling on any earlier page never touches the old version.
Function HfsInstFilesPagePre
  Call HfsRemovePreviousInstallIfAny
FunctionEnd
; Disposable-data cleanup for the reinstall path: the whitelist matches
; un.HfsSweepDisposableDirs verbatim.
; The uninstaller's copy carries the un. prefix and cannot be Call'ed from the installer, so
; an installer-side copy lives here.
Function HfsSweepDisposableDirs
  SetShellVarContext current
  StrCpy $R0 "$LOCALAPPDATA\HiFiShifter"
  ${IfNot} ${FileExists} "$R0\*"
    Return
  ${EndIf}
  ${If} ${FileExists} "$R0\cache\*"
    RMDir /r "$R0\cache"
  ${EndIf}
  ${If} ${FileExists} "$R0\logs\*"
    RMDir /r "$R0\logs"
  ${EndIf}
  ${If} ${FileExists} "$R0\EBWebView\*"
    RMDir /r "$R0\EBWebView"
  ${EndIf}
  ${If} ${FileExists} "$R0\webview\*"
    RMDir /r "$R0\webview"
  ${EndIf}
  ${If} ${FileExists} "$R0\pcm\*"
    RMDir /r "$R0\pcm"
  ${EndIf}
  ${If} ${FileExists} "$R0\capture\*"
    RMDir /r "$R0\capture"
  ${EndIf}
  ${If} ${FileExists} "$R0\transfer\*"
    RMDir /r "$R0\transfer"
  ${EndIf}
  ${If} ${FileExists} "$R0\links\*"
    RMDir /r "$R0\links"
  ${EndIf}
  ${If} ${FileExists} "$R0\plugin-content-v1\*"
    RMDir /r "$R0\plugin-content-v1"
  ${EndIf}
  ; The root is removed only when empty; if it cannot be deleted, user data remains to keep.
  RMDir "$R0"
FunctionEnd

Function .onInit
  ${IfNot} ${RunningX64}
    ; The one message that cannot be localized: at this point the user's language choice is
    ; not known yet.
    !insertmacro HFS_FATAL_NOT_X64
    Abort
  ${EndIf}
  SetRegView 64
  SetShellVarContext ${HFS_MODEL_ROOT_CONTEXT}
  StrCpy $HfsModelRoot "${HFS_MODEL_ROOT}"
  SetShellVarContext current
  StrCpy $HfsVst3Dir "${HFS_DEFAULT_VST3_DIR}"
  ; [Why both directories are initialized in .onInit] A custom page's Leave callback runs only
  ; when the user actually walks through that page; silent installs (/S) skip all pages.
  ; Previously only VST3 was initialized, leaving the CLAP directory an empty string, so File
  ; landed in the current directory (the VirtualStore) instead of the target directory.
  StrCpy $HfsClapDir "${HFS_DEFAULT_CLAP_DIR}"

  StrCpy $HfsIsRepair 0
  ; /repair: verify and fill in only, no pages shown (files are idempotent overwrites, and the
  ; models section skips on its own).
  ${StrStr} $0 $CMDLINE "/repair"
  ${If} $0 != ""
    StrCpy $HfsIsRepair 1
    SetSilent silent
  ${EndIf}

  ; Language selection: MUI records the choice in the registry and reuses it next time. Must
  ; happen before the reinstall options page — that page's five-language strings resolve via
  ; $(HFS_REINSTALL_*).
  !insertmacro MUI_LANGDLL_DISPLAY

  ; Detect the previous version only after the language is fixed: silent installs have no
  ; pages and remove it on the spot; interactive installs walk the reinstall options page
  ; first (the user's one and only confirmation happens here) and remove it for real at the
  ; INSTFILES page.
  ; Neither path **starts** the old uninstaller, so no second uninstall window can appear
  ; during the actual install.
  Call HfsDetectPreviousInstall
  ${If} ${Silent}
    Call HfsRemovePreviousInstallIfAny
  ${EndIf}

  !ifdef HFS_HOST_GUARD
    Call HfsHostRunning
    ${If} $0 == 1
      ${IfNot} ${Silent}
        MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_HOST_OPEN)" /SD IDOK
      ${EndIf}
      Abort
    ${EndIf}
  !endif
FunctionEnd

Function un.onInit
  ; [Why "delete models" is checked by default] Only this product writes the shared
  ; model library; deleting it by default is the only way the disk does not stay occupied after
  ; "I already uninstalled it". Users who want to keep it (e.g. planning to reinstall the same
  ; version right away) simply uncheck it on the options page. Caches and logs stay kept by
  ; default: the runtime reclaims caches on its own, and logs are the only clue for locating
  ; the last failure — a "clean uninstall" should not destroy the evidence as a side effect.
  ;
  ; Silent uninstalls (/S) skip the options page, so the defaults here apply directly.
  ; Reinstalling over an existing version passes /DeleteModels= and /DeleteCache=, carrying
  ; the user's choices from the **installer's** options page over verbatim (the uninstaller's
  ; own options page is not shown under /S).
  StrCpy $HfsUnOptModels 1
  StrCpy $HfsUnOptCache 0
  ${UnStrStr} $0 $CMDLINE "/DeleteModels=0"
  ${If} $0 != ""
    StrCpy $HfsUnOptModels 0
  ${EndIf}
  ${UnStrStr} $0 $CMDLINE "/DeleteModels=1"
  ${If} $0 != ""
    StrCpy $HfsUnOptModels 1
  ${EndIf}
  ${UnStrStr} $0 $CMDLINE "/DeleteCache=0"
  ${If} $0 != ""
    StrCpy $HfsUnOptCache 0
  ${EndIf}
  ${UnStrStr} $0 $CMDLINE "/DeleteCache=1"
  ${If} $0 != ""
    StrCpy $HfsUnOptCache 1
  ${EndIf}
  SetRegView 64
  ; [Why $INSTDIR cannot be used directly] The uninstaller copies itself to a temporary
  ; directory and runs from there, passing _?= — that value is **the uninstaller's own
  ; directory**, not a trusted source. Reading back the InstallLocation written at install
  ; time is the authoritative value; if unreadable, fall back to the uninstaller's directory.
  ReadRegStr $INSTDIR ${HFS_ARP_ROOT} "${ARP_KEY}" "InstallLocation"
  ${If} $INSTDIR == ""
    StrCpy $INSTDIR "$EXEDIR"
  ${EndIf}
  ReadRegStr $HfsVst3Dir ${HFS_ARP_ROOT} "${ARP_KEY}" "Vst3Dir"
  ${If} $HfsVst3Dir == ""
    StrCpy $HfsVst3Dir "${HFS_DEFAULT_VST3_DIR}"
  ${EndIf}
  ReadRegStr $HfsClapDir ${HFS_ARP_ROOT} "${ARP_KEY}" "ClapDir"
  ${If} $HfsClapDir == ""
    StrCpy $HfsClapDir "${HFS_DEFAULT_CLAP_DIR}"
  ${EndIf}
  ; Same for the plugin resource directory: the value written at install time is the
  ; authoritative one (the old version may have kept it inside the CLAP directory).
  ReadRegStr $HfsClapResDir ${HFS_ARP_ROOT} "${ARP_KEY}" "ClapResDir"
  ${If} $HfsClapResDir == ""
    SetShellVarContext ${HFS_CLAP_RES_CONTEXT}
    StrCpy $HfsClapResDir "${HFS_CLAP_RES_DIR}"
    SetShellVarContext current
  ${EndIf}
  ; [Why the model library root is also read from the uninstall entry] The shared library's
  ; location has changed across installer versions. Deleting by this version's constant alone
  ; means "uninstalled, but the models are still there" for a copy the old version wrote
  ; elsewhere. The value written at install time is authoritative; only when an old version
  ; never wrote this value do we fall back to this version's constant.
  ReadRegStr $HfsModelRoot ${HFS_ARP_ROOT} "${ARP_KEY}" "ModelRoot"
  ${If} $HfsModelRoot == ""
    ; $HfsModelRoot was already read back from the uninstall entry in un.onInit (this version's
    ; constant only as the fallback); it is not recomputed here.
  ${EndIf}

  !ifdef HFS_HOST_GUARD
    Call un.HfsHostRunning
    ${If} $0 == 1
      ${IfNot} ${Silent}
        MessageBox MB_OK|MB_ICONSTOP "$(HFS_UNINSTALL_HOST_OPEN)" /SD IDOK
      ${EndIf}
      Abort
    ${EndIf}
  !endif
FunctionEnd

; ═══ Components ═════════════════════════════════════════════════════════
;
; [Why two groups] Previously the five entries were laid out flat, mixing the App, the shared
; model library, shortcuts and both plugin formats so users could not tell which were the main
; program and which were for DAWs. They are now two expandable trees by purpose: one for the
; main program, one for DAW plugins. Group nodes also occupy a section index (see the
; SECGRP_* numbering notes at the top of the file).
;
; [Putting mandatory items inside groups is safe] Verified by testing: unchecking a whole
; group leaves the SectionIn RO App and shared model library selected and executing as normal;
; only the optional shortcuts get unchecked along with it.
;
; ── Main app ─────────────────────────────────────────────────────────
SectionGroup /e "$(HFS_GROUP_APP)"
Section "$(HFS_COMPONENT_APP)"
  SectionIn RO
  !ifdef HFS_HOST_GUARD
    Call HfsHostRunning
    ${If} $0 == 1
      ${IfNot} ${Silent}
        MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_HOST_OPEN)" /SD IDOK
      ${EndIf}
      Abort
    ${EndIf}
  !endif
  Call HfsAppRunning
  ${If} $0 == 1
    ${IfNot} ${Silent}
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_APP_RUNNING)" /SD IDOK
    ${EndIf}
    Abort
  ${EndIf}

  DetailPrint "$(HFS_INSTALLING)"
  Call HfsEnsureWebView2

  ; ── Directory reconciliation: purge old output first, then write ────
  ;
  ; [Why $INSTDIR\models must be cleared] The old installer (the Tauri template) installed
  ; the weights here; the new layout keeps only models.json here (weights go to the
  ; shared library in ProgramData). Not deleting means quietly keeping a second full copy of
  ; the weights after an upgrade, and the App would still preferentially read those old weights.
  ;
  ; [Why licenses and LICENSE are deleted] Old versions put them in the install root; they now
  ; ship as a whole with the App. Deleting first guarantees that a "document went missing"
  ; does not silently survive via overwrite.
  RMDir /r "$INSTDIR\models"
  RMDir /r "$INSTDIR\licenses"
  Delete "$INSTDIR\LICENSE"

  SetOutPath "$INSTDIR"
  File /r "${APP_DIR}\*"

  ; Full license texts: the App and the plugins each carry a complete copy (the user may
  ; install only one of them).
  SetOutPath "$INSTDIR"
  File "${HFS_LICENSE}"
  SetOutPath "$INSTDIR\licenses"
  File /r "${HFS_LICENSES}\*"

  ; On upgrade the user may have unchecked "file associations": bindings left behind by the
  ; previous version that still point at this install must be removed during this install,
  ; otherwise the user's choice has no effect.
  SectionGetFlags ${SEC_FILEASSOC} $0
  IntOp $0 $0 & ${SF_SELECTED}
  ${If} $0 == 0
    Call HfsRemoveFileAssociationsIfOurs
  ${EndIf}

  ; ── Uninstall info ──────────────────────────────────────────────────
  ;
  ; [Why the \Uninstall\HiFiShifter key is reused] The old App (Tauri's NSIS installer) used
  ; exactly this key. Reusing it makes an upgrade an "adoption" instead of two HiFiShifter
  ; entries appearing in "Installed apps".
  WriteUninstaller "${UNINSTALLER}"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "DisplayName" "$(HFS_NAME)"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "DisplayIcon" "$INSTDIR\HiFiShifter.exe"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "Publisher" "$(HFS_PUBLISHER)"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "Vst3Dir" "$HfsVst3Dir"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "ClapDir" "$HfsClapDir"
  ; [Why the model library root is written into the uninstall entry] The shared library's
  ; location is not a permanent constant: after the directory layout changed, the uninstaller
  ; and the next reinstall must both be able to ask "where did the previous version put its
  ; model library" instead of guessing from this version's constant.
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "ModelRoot" "$HfsModelRoot"
  ; Without _?=: the uninstaller copies itself to a temporary directory by default before
  ; running, which is what lets it delete itself.
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "UninstallString" '"${UNINSTALLER}"'
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "QuietUninstallString" '"${UNINSTALLER}" /S'
  WriteRegDWORD ${HFS_ARP_ROOT} "${ARP_KEY}" "NoModify" 1
  WriteRegDWORD ${HFS_ARP_ROOT} "${ARP_KEY}" "NoRepair" 1
SectionEnd
Section "$(HFS_COMPONENT_MODELS)"
  SectionIn RO

  StrCpy $HfsVerifyRoot "$HfsModelRoot\${MODELS_VERSION}"
  Call HfsVerifyModels
  ${If} $0 == 1
    DetailPrint "$(HFS_COMPONENT_MODELS_SKIP)"
    Goto hfs_models_done
  ${EndIf}

  DetailPrint "$(HFS_COMPONENT_MODELS_INSTALL)"
  ; Write into staging first, then rename as a whole: a mid-way failure never leaves behind a
  ; library that "looks present but misses files", which would make the runtime misjudge it as
  ; usable (the same approach as model_store::publish).
  RMDir /r "$HfsModelRoot\${MODELS_VERSION}.staging"
  CreateDirectory "$HfsModelRoot\${MODELS_VERSION}.staging"
  SetOutPath "$HfsModelRoot\${MODELS_VERSION}.staging"
  File /r "${MODELS_DIR}\*"

  StrCpy $HfsVerifyRoot "$HfsModelRoot\${MODELS_VERSION}.staging"
  Call HfsVerifyModels
  ${If} $0 != 1
    RMDir /r "$HfsModelRoot\${MODELS_VERSION}.staging"
    ${IfNot} ${Silent}
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_MODELS_VERIFY_FAILED)" /SD IDOK
    ${EndIf}
    Abort
  ${EndIf}
  ; [Why SetOutPath must be switched away first] NSIS's SetOutPath calls SetCurrentDirectory;
  ; the File /r above made .staging the current directory, and renaming it directly means
  ; "rename the current working directory" — Windows always fails at that and Rename reports no
  ; error, so the model library would stay in .staging forever: install reports success while
  ; the runtime cannot read the weights (observed as RenameError=1). Switch back to $INSTDIR
  ; before renaming.
  SetOutPath "$INSTDIR"
  RMDir /r "$HfsModelRoot\${MODELS_VERSION}"
  ; RMDir on a nonexistent directory, and the failed ARP queries before it, both set the global
  ; error flag; a successful Rename does not clear it. Let only Rename's own result decide the
  ; ${If} ${Errors} below.
  ClearErrors
  Rename "$HfsModelRoot\${MODELS_VERSION}.staging" "$HfsModelRoot\${MODELS_VERSION}"
  ${If} ${Errors}
    ${IfNot} ${Silent}
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_MODELS_VERIFY_FAILED)" /SD IDOK
    ${EndIf}
    Abort
  ${EndIf}
  hfs_models_done:
  ; Permissions: ProgramData inheritance is incomplete on some machines, so grant Users read
  ; access explicitly. Use the SID (S-1-5-32-545 = Users) instead of the localized group name —
  ; group names do not match on non-English systems.
  nsExec::ExecToLog 'icacls "$HfsModelRoot" /grant *S-1-5-32-545:(OI)(CI)RX /T /Q'
  Call HfsModelsGc

  SetOutPath "$INSTDIR"
  ${If} $HfsIsRepair == 1
    DetailPrint "$(HFS_REPAIR_DONE)"
  ${EndIf}
SectionEnd
Section "$(HFS_COMPONENT_SHORTCUTS)"
  SetShellVarContext all
  CreateDirectory "$SMPROGRAMS\HiFiShifter"
  CreateShortCut "$SMPROGRAMS\HiFiShifter\HiFiShifter.lnk" "$INSTDIR\HiFiShifter.exe"
  CreateShortCut "$DESKTOP\HiFiShifter.lnk" "$INSTDIR\HiFiShifter.exe"
  SetShellVarContext current
SectionEnd
Section "$(HFS_COMPONENT_FILE_ASSOC)"
  DetailPrint "$(HFS_INSTALLING)"
  Call HfsWriteFileAssociations
SectionEnd
SectionGroupEnd

; ── DAW plugins ──────────────────────────────────────────────────────
SectionGroup /e "$(HFS_GROUP_PLUGINS)"
Section "$(HFS_COMPONENT_VST3)"
  !ifdef HFS_HOST_GUARD
    Call HfsHostRunning
    ${If} $0 == 1
      ${IfNot} ${Silent}
        MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_HOST_OPEN)" /SD IDOK
      ${EndIf}
      Abort
    ${EndIf}
  !endif

  DetailPrint "$(HFS_INSTALLING)"

  ; ── Directory reconciliation (verbatim, old plugin installer) ──────
  ;
  ; [Why this must be cleared] Vite stamps every chunk with a content hash, so the filename
  ; changes the moment content does, and File /r only writes, never deletes. Every reinstall
  ; therefore stacks another layer and the directory grows monotonically. These two directories
  ; are **pure build output**; deleting them is idempotent and safe.
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\frontend"
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\x86_64-win"
  ; [Why other platforms' architectures are cleared too] dist\HiFiShifter.vst3 is a
  ; cross-platform shared bundle, and the early packaging logic used Copy-Item * -Recurse,
  ; which installed the x86_64-linux left behind by Linux builds (about 57 MB of .so) onto
  ; Windows as well. File /r only writes and never deletes, so those stale directories survive
  ; every reinstall; explicitly deleting the known non-Windows architecture directories is the
  ; only way already-affected users come out clean after upgrading.
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\x86_64-linux"
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\aarch64-linux"
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\MacOS"
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Frameworks"
  ; Early versions kept the full license texts and change notes in the Resources root; they are
  ; no longer shipped.
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\COPYING.LGPL-2.1"
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\SOUNDTOUCH-CHANGES.md"
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\VST3-BUILD.md"
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\frontend\index.html"
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\frontend\detached.html"
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\frontend\waveform-test.html"

  SetOutPath "$HfsVst3Dir\${VST3_SUBDIR}"
  File /r "${VST3_DIR}\*"
  SetOutPath "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources"
  File "${HFS_LICENSE}"
  SetOutPath "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\licenses"
  File /r "${HFS_LICENSES}\*"

  ; The old plugin installer had its own uninstall entry (\Uninstall\HiFiShifterVST3). This
  ; installer has taken over the plugins' install locations and the model library, so it clears
  ; that entry on success to avoid two entries in "Installed apps".
  DeleteRegKey HKLM "${LEGACY_VST3_ARP_KEY}"
  DeleteRegKey HKCU "${LEGACY_VST3_ARP_KEY}"
SectionEnd
Section "$(HFS_COMPONENT_CLAP)"
  ; [Why the public CLAP directory gets only the .clap file] %COMMONFILES%\CLAP is the
  ; machine-wide shared plugin directory: per the spec only plugin binaries belong there, and
  ; other vendors' plugins live alongside. Putting the frontend and models.json into a
  ; generically named sibling directory shows the user "one plugin plus one folder of unknown
  ; origin" that nobody dares to delete. Resources instead go to
  ; %ProgramData%\HiFiShifter\plugin-resources (the same root as the shared model library), and
  ; the runtime resolves them in "module sibling → machine-wide shared" order (see
  ; resource_root.rs).
  SetOutPath "$HfsClapDir"
  Delete "$HfsClapDir\${CLAP_FILENAME}"
  File "/oname=${CLAP_FILENAME}" "${CLAP_FILE}"
  ; Upgrading from the old layout: clear the resource directory that used to sit in the CLAP
  ; directory. Otherwise both copies coexist and "sibling first" keeps reading the old one —
  ; the new frontend would never take effect.
  RMDir /r "$HfsClapDir\${CLAP_RESOURCES}"
  SetShellVarContext ${HFS_CLAP_RES_CONTEXT}
  SetOutPath "${HFS_CLAP_RES_DIR}"
  File /r "${CLAP_RES_DIR}\*.*"
  SetShellVarContext current
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "ClapDir" "$HfsClapDir"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "ClapResDir" "${HFS_CLAP_RES_DIR}"
SectionEnd
SectionGroupEnd

; ── Component hover descriptions ─────────────────────────────────────
; [Why this block is indispensable] MUI2's components page comes with a description box, but
; its content comes only from the .onMouseOverSection callback; this script previously had no
; MUI_FUNCTION_DESCRIPTION_* at all, so hovering always showed MUI's built-in static hint
; (the "Hover your mouse over a component..." text in the screenshot).
; Every section and every group registers one entry; the five-language strings come from the
; installer_desc_* tables.
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SECGRP_APP} "$(HFS_DESC_GROUP_APP)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_APP} "$(HFS_DESC_APP)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_MODELS} "$(HFS_DESC_MODELS)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_SHORTCUTS} "$(HFS_DESC_SHORTCUTS)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_FILEASSOC} "$(HFS_DESC_FILE_ASSOC)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SECGRP_PLUGINS} "$(HFS_DESC_GROUP_PLUGINS)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_VST3} "$(HFS_DESC_VST3)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_CLAP} "$(HFS_DESC_CLAP)"
!insertmacro MUI_FUNCTION_DESCRIPTION_END


; ── Uninstall options page ─────────────────────────────────────────────
; "Clean shared model library" is checked by default (set to 1 in un.onInit);
; "Clean caches and logs" is unchecked by default:
;   - Clean shared model library (shared by all installs)
;   - Clean caches and logs (waveform/render caches, logs, WebView data, temporary files)
; However the checkboxes are set, settings and unsaved projects are always kept; if another
; HiFiShifter install exists, neither option takes effect (so uninstalling the App never
; sweeps away data the plugins still use).
Function un.HfsUninstallOptionsCreate
  !insertmacro MUI_HEADER_TEXT "$(HFS_UNINSTALL_OPTIONS_TITLE)" "$(HFS_UNINSTALL_OPTIONS_SUBTITLE)"
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 24u "$(HFS_UNINSTALL_OPTIONS_TOP)"
  Pop $0
  ${NSD_CreateCheckbox} 0 30u 100% 12u "$(HFS_UNINSTALL_OPT_MODELS)"
  Pop $HfsUnChkModels
  ${NSD_CreateLabel} 12u 44u 88% 20u "$(HFS_UNINSTALL_OPT_MODELS_DESC)"
  Pop $0
  ${NSD_CreateCheckbox} 0 72u 100% 12u "$(HFS_UNINSTALL_OPT_CACHE)"
  Pop $HfsUnChkCache
  ${NSD_CreateLabel} 12u 86u 88% 20u "$(HFS_UNINSTALL_OPT_CACHE_DESC)"
  Pop $0
  ${If} $HfsUnOptModels == 1
    ${NSD_Check} $HfsUnChkModels
  ${EndIf}
  ${If} $HfsUnOptCache == 1
    ${NSD_Check} $HfsUnChkCache
  ${EndIf}
  nsDialogs::Show
FunctionEnd

Function un.HfsUninstallOptionsLeave
  ${NSD_GetState} $HfsUnChkModels $HfsUnOptModels
  ${NSD_GetState} $HfsUnChkCache $HfsUnOptCache
FunctionEnd

Section "Uninstall"
  ; Removes only associations whose current binding points at this install; other programs'
  ; defaults stay untouched.
  Call un.HfsRemoveFileAssociations
  ; Delete only our own: $HfsVst3Dir is Common Files\VST3, which holds other vendors' plugins too.
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}"
  Delete "$HfsClapDir\${CLAP_FILENAME}"
  ; Delete only this product's resource directory: $HfsClapDir is the public CLAP directory,
  ; which holds other vendors' plugins.
  ; Both layouts must be cleared: the old version kept it next to the module, the new one in
  ; the machine-wide shared directory (recorded in ClapResDir at install time).
  RMDir /r "$HfsClapDir\${CLAP_RESOURCES}"
  RMDir /r "$HfsClapResDir"
  ; The App directory belongs entirely to this product (consistent with the Tauri installer's
  ; behavior).
  RMDir /r "$INSTDIR"

  SetShellVarContext all
  RMDir /r "$SMPROGRAMS\HiFiShifter"
  Delete "$DESKTOP\HiFiShifter.lnk"
  SetShellVarContext current

  DeleteRegKey ${HFS_ARP_ROOT} "${ARP_KEY}"
  ; The old installer's key may still be around (this installer deletes it on success, but the
  ; user may have uninstalled first and cleaned up by hand).
  DeleteRegKey HKLM "${LEGACY_VST3_ARP_KEY}"
  DeleteRegKey HKCU "${LEGACY_VST3_ARP_KEY}"

  ; ── Optional cleanup: only when no other HiFiShifter install exists ────
  ; The App and the still-in-use VST3/CLAP plugins share the model library and cache
  ; directories; uninstalling the App must not sweep away data the plugins still use. The
  ; verdict is stored in a dedicated variable so LogicLib macros cannot clobber $0.
  Call un.HfsOtherInstallExists
  StrCpy $HfsUnOtherInstall $0

  ; ── Model library: removed only when checked and no other install ─────
  ; [Why delete the whole root instead of only ${MODELS_VERSION}] At install time HfsModelsGc
  ; deliberately keeps the current version + the previous one, so any upgraded machine
  ; necessarily has two version directories; a failed install also leaves <version>.staging
  ; behind. Deleting just one version directory would leave the rest making
  ; RMDir "$HfsModelRoot" fail forever — showing up as "uninstalled, but the model files are
  ; still there". The whole root is written by this product and no other program puts anything
  ; here, so deleting the whole directory is safe.
  ${If} $HfsUnOptModels == 1
  ${AndIf} $HfsUnOtherInstall == 0
    SetShellVarContext ${HFS_MODEL_ROOT_CONTEXT}
    StrCpy $HfsModelRoot "${HFS_MODEL_ROOT}"
    SetShellVarContext current
    ; [Why children are deleted one by one before the root] RMDir /r **silently skips**
    ; subdirectories that are locked or have unusual permissions — the whole uninstall reports
    ; success while model files remain on disk. Deleting them individually means one failure
    ; does not affect the others.
    FindFirst $R3 $R4 "$HfsModelRoot\*"
    StrCmp $R4 "" hfs_models_children_done
    hfs_models_children_loop:
    StrCmp $R4 "." hfs_models_children_next
    StrCmp $R4 ".." hfs_models_children_next
    ${If} ${FileExists} "$HfsModelRoot\$R4\*"
      RMDir /r "$HfsModelRoot\$R4"
    ${Else}
      Delete "$HfsModelRoot\$R4"
    ${EndIf}
    hfs_models_children_next:
    ClearErrors
    FindNext $R3 $R4
    IfErrors hfs_models_children_done
    StrCmp $R4 "" hfs_models_children_done hfs_models_children_loop
    hfs_models_children_done:
    FindClose $R3
    RMDir /r "$HfsModelRoot"
    ; Remove the empty shell above the model root (succeeds only when it is truly empty);
    ; applies in both modes.
    RMDir "$HfsModelRoot\.."
  ${EndIf}

  ; ── Caches and logs: only when checked; tiered by regenerability ──────
  ; [Why neither "delete everything" nor "keep everything"] %APPDATA%\HiFiShifter holds
  ; settings, keyboard shortcuts and recent projects — unrecoverable once deleted;
  ; %LOCALAPPDATA%\HiFiShifter holds both gigabytes of regenerable caches/logs/WebView profile
  ; and the user's unsaved project material and plugin notebooks. The whitelist lists only the
  ; subdirectories that app_paths.rs / temp_sweep.rs explicitly define as disposable; everything
  ; else is kept by default.
  ${If} $HfsUnOptCache == 1
  ${AndIf} $HfsUnOtherInstall == 0
    Call un.HfsSweepDisposableDirs
    ; [Why $APPDATA instead of %APPDATA%] NSIS strings do not do cmd-style environment
    ; expansion; %VAR% would appear literally in the uninstall log. The shell folder constants
    ; are $APPDATA / $LOCALAPPDATA.
    DetailPrint "Removed caches, logs and temporary data. Settings and any unsaved work were kept in $APPDATA\HiFiShifter and $LOCALAPPDATA\HiFiShifter."
  ${ElseIf} $HfsUnOtherInstall == 1
    DetailPrint "Another HiFiShifter installation is still present; shared data was kept."
  ${Else}
    DetailPrint "Kept cached data as requested."
  ${EndIf}
SectionEnd

; Deletes the subdirectories of %LOCALAPPDATA%\HiFiShifter that are **explicitly regenerable**.
; This list maps one-to-one to backend/hifishifter-kernel/src/app_paths.rs's cache_dir()/
; log_dir()/local_data_subdir() and to hifishifter-plugin/src/editor/temp_sweep.rs's sweep
; targets:
;   cache / logs — caches and logs (app_paths.rs says outright: "deleting only slows things
;   down, no data is lost")
;   EBWebView / webview — WebView2 user data directories (TempSweep reclaims them too)
;   pcm / capture / transfer / links — PCM staging, capture, transfer and host-link workspace
;   markers
;   plugin-content-v1 — host PCM cache (host_pcm_cache.rs)
; [Deliberately kept] unsaved (unsaved project material), plugin-notebook (plugin notebook
; bodies and attachments), ara-content-v1 / ara-instances (ARA content written by the host),
; and any new directory the source does not list.
Function un.HfsSweepDisposableDirs
  SetShellVarContext current
  StrCpy $R0 "$LOCALAPPDATA\HiFiShifter"
  ${IfNot} ${FileExists} "$R0\*"
    Return
  ${EndIf}
  ; Delete one by one: when a directory is locked or has unusual permissions, RMDir /r silently
  ; skips it, but the other directories are unaffected.
  ${If} ${FileExists} "$R0\cache\*"
    DetailPrint "Removing disposable data: cache"
    RMDir /r "$R0\cache"
  ${EndIf}
  ${If} ${FileExists} "$R0\logs\*"
    DetailPrint "Removing disposable data: logs"
    RMDir /r "$R0\logs"
  ${EndIf}
  ${If} ${FileExists} "$R0\EBWebView\*"
    DetailPrint "Removing disposable data: EBWebView"
    RMDir /r "$R0\EBWebView"
  ${EndIf}
  ${If} ${FileExists} "$R0\webview\*"
    DetailPrint "Removing disposable data: webview"
    RMDir /r "$R0\webview"
  ${EndIf}
  ${If} ${FileExists} "$R0\pcm\*"
    DetailPrint "Removing disposable data: pcm"
    RMDir /r "$R0\pcm"
  ${EndIf}
  ${If} ${FileExists} "$R0\capture\*"
    DetailPrint "Removing disposable data: capture"
    RMDir /r "$R0\capture"
  ${EndIf}
  ${If} ${FileExists} "$R0\transfer\*"
    DetailPrint "Removing disposable data: transfer"
    RMDir /r "$R0\transfer"
  ${EndIf}
  ${If} ${FileExists} "$R0\links\*"
    DetailPrint "Removing disposable data: links"
    RMDir /r "$R0\links"
  ${EndIf}
  ${If} ${FileExists} "$R0\plugin-content-v1\*"
    DetailPrint "Removing disposable data: plugin-content-v1"
    RMDir /r "$R0\plugin-content-v1"
  ${EndIf}
  ; The root is removed only when empty; if it cannot be deleted, user data remains to keep —
  ; RMDir without /r quietly does nothing in that case.
  RMDir "$R0"
FunctionEnd

; Is there another HiFiShifter install? Result goes into $0 (1 = yes).
Function un.HfsOtherInstallExists
  StrCpy $0 0
  StrCpy $R1 "HKLM"
  Call un.HfsScanUninstallRoot
  ${If} $0 == 1
    Return
  ${EndIf}
  StrCpy $R1 "HKCU"
  Call un.HfsScanUninstallRoot
FunctionEnd

Function un.HfsScanUninstallRoot
  StrCpy $R2 0
  ${Do}
    ${If} $R1 == "HKLM"
      EnumRegKey $R3 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall" $R2
    ${Else}
      EnumRegKey $R3 HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall" $R2
    ${EndIf}
    ${If} $R3 == ""
      ${ExitDo}
    ${EndIf}
    IntOp $R2 $R2 + 1
    ; Only "other variant" keys count: this product's own key name (HiFiShifter) is explicitly
    ; excluded.
    ; [Why the self key must be excluded] DeleteRegKey may fail to remove our own key due to
    ; permissions; counting it as "another install" would keep the model library forever —
    ; exactly the "uninstalled, but the models remain" symptom. Only the HiFiShifterVST3 left
    ; by the old plugin installer, and other variants split out in the future, count.
    ${If} $R3 == "HiFiShifter"
      ${Continue}
    ${EndIf}
    StrCpy $R4 $R3 11
    ${If} $R4 == "HiFiShifter"
      StrCpy $0 1
      ${ExitDo}
    ${EndIf}
  ${Loop}
FunctionEnd
