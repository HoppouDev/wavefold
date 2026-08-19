; Built with Inno Setup 6.4+ (needed for WizardStyle=modern's automatic
; dark mode, which follows the Windows theme setting).
;
; Expects, in SourceDir (see below), the files this installer packages:
;   wavefold.exe   - built for #Arch (see ISCC /DArch=... below)
;   icon.ico       - assets/windows/icon.ico
;   LICENSE.md
;   README.md
;
; Build with e.g.:
;   iscc /DAppVersion=1.2.3 /DArch=x64 /DSourceDir=dist\x64 installer.iss
;   iscc /DAppVersion=1.2.3 /DArch=arm64 /DSourceDir=dist\arm64 installer.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Arch
  #define Arch "x64"
#endif
#ifndef SourceDir
  #define SourceDir "dist"
#endif

#define AppName "Wavefold"
#define AppPublisher "HoppouDev"
#define AppURL "https://github.com/HoppouDev/wavefold"

[Setup]
; Fixed GUID - do not change between releases, it's how Windows recognizes
; upgrades vs. a separate install.
AppId={{2B6E6E9A-6C7E-4B8F-9E60-3B1B2F6E9C41}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=wavefold-{#AppVersion}-{#Arch}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\..\assets\windows\icon.ico
UninstallDisplayIcon={app}\icon.ico
LicenseFile=..\..\LICENSE.md
#if Arch == "arm64"
ArchitecturesAllowed=arm64
ArchitecturesInstallIn64BitMode=arm64
#else
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add {#AppName} to PATH (lets you run ""wavefold"" from a terminal)"; Flags: unchecked
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#SourceDir}\wavefold.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\assets\windows\icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\wavefold.exe"; IconFilename: "{app}\icon.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\wavefold.exe"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\wavefold.exe"; Description: "Launch {#AppName}"; Flags: postinstall nowait skipifsilent unchecked

[Code]
const
  EnvironmentKey = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';
  WM_SETTINGCHANGE = $1A;
  SMTO_ABORTIFHUNG = $0002;

function SendMessageTimeoutA(hWnd: LongInt; Msg: LongInt; wParam: LongInt; lParam: PAnsiChar;
  fuFlags: LongInt; uTimeout: LongInt; var lpdwResult: LongInt): LongInt;
  external 'SendMessageTimeoutA@user32.dll stdcall';

procedure EnvAddPath(Path: string);
var
  Paths: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', Paths) then
    Paths := '';
  if Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(Paths) + ';') > 0 then
    exit;
  if (Length(Paths) > 0) and (Paths[Length(Paths)] <> ';') then
    Paths := Paths + ';';
  Paths := Paths + Path;
  if not RegWriteExpandStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', Paths) then
    Log('EnvAddPath: failed to write PATH');
end;

procedure EnvRemovePath(Path: string);
var
  Paths: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', Paths) then
    exit;
  P := Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(Paths) + ';');
  if P = 0 then
    exit;
  Delete(Paths, P - 1, Length(Path) + 1);
  RegWriteExpandStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', Paths);
end;

procedure RefreshEnvironment;
var
  Res: LongInt;
begin
  { Broadcast WM_SETTINGCHANGE so new processes (e.g. a freshly opened
    terminal) see the updated PATH without a reboot or logoff. }
  SendMessageTimeoutA(HWND_BROADCAST, WM_SETTINGCHANGE, 0, 'Environment', SMTO_ABORTIFHUNG, 5000, Res);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
  begin
    EnvAddPath(ExpandConstant('{app}'));
    RefreshEnvironment;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    EnvRemovePath(ExpandConstant('{app}'));
    RefreshEnvironment;
  end;
end;
