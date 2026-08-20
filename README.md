# wavefold

This program applies a VHS-like DCT "distortion" effect to video: it
decodes, runs a per-frame DCT reconstruct pass (GPU via wgpu, or
a CPU fallback), and re-encodes with a chosen codec (x264/x265/vp9/av1,
software or VAAPI hardware).

## Media backend

Decode/encode is behind a `MediaBackend` trait, one implementation per
platform: **GStreamer** everywhere except Windows (no system package
manager there to install it from), **Media Foundation** on Windows (part
of the OS, no separate runtime needed).

### Supported output codecs (GStreamer backend — Linux/macOS)

| Codec       | Software element | VAAPI hardware element | Notes                                                              |
|-------------|-------------------|-------------------------|---------------------------------------------------------------------|
| H.264       | `x264enc`         | `vah264enc`             |                                                                     |
| H.265/HEVC  | `x265enc`         | `vah265enc`             |                                                                     |
| VP9         | `vp9enc`          | — (not available)       | No GPU has VAAPI VP9 encode. Software VP9 must mux to `.mkv`, not `.mp4`. |
| AV1         | `av1enc`          | `vaav1enc`               |                                                                     |

### Supported output codecs (Media Foundation backend — Windows)

Media Foundation auto-negotiates whichever encoder MFT is registered for
the requested codec — there's no fixed element name the way GStreamer has.
`--encoder <codec>-hardware` sets `MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS`
to allow (not force) a hardware MFT; the plain variant forces software.

| Codec       | Notes                                                              |
|-------------|---------------------------------------------------------------------|
| H.264       |                                                                     |
| H.265/HEVC  |                                                                     |
| VP9         | Windows ships an MF *decoder* for VP9, not an encoder — same gap as the GStreamer backend's missing VAAPI VP9 encode. |
| AV1         | Encoder availability depends on what's registered on the system (varies by Windows version/hardware). |

Output container is inferred from the file extension by Media Foundation's
own byte-stream-handler resolution — reliably `.mp4`/`.mov`; unlike the
GStreamer backend there's no built-in non-MP4 fallback, so a `.mkv` output
fails cleanly instead of muxing.

## Build

```bash
cargo build --release
```

You need Rust (stable) and GStreamer's development headers to build, plus
the actual plugins installed at runtime for whichever codecs you use.
Verified in a clean container for each distro below.

### Arch Linux

```bash
sudo pacman -S --needed rust pkgconf clang base-devel \
  gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad \
  gst-plugins-ugly gst-libav gst-plugin-va
```

`gst-plugin-va` (the VAAPI hardware-encode plugin) is packaged separately
from `gst-plugins-bad` on Arch — install it explicitly or `vah264enc`/etc.
won't exist.

### Fedora

```bash
sudo dnf install rust cargo gcc clang pkgconf-pkg-config \
  gstreamer1-devel gstreamer1-plugins-base-devel gstreamer1-plugins-good
```

That's enough to build and to run with `--encoder av1` (the only codec
here with a patent-unencumbered official-repo encoder). For x264/x265/vp9
software encoding and VAAPI hardware encoding, enable
[RPM Fusion](https://rpmfusion.org/) first (Fedora's official repos don't
ship those for patent reasons):

```bash
sudo dnf install \
  https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
  https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install gstreamer1-plugins-bad-freeworld gstreamer1-plugins-ugly \
  gstreamer1-libav gstreamer1-vaapi
```

### Ubuntu / Debian

```bash
sudo apt-get install pkg-config build-essential \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav gstreamer1.0-vaapi
```

Install Rust via [rustup](https://rustup.rs) — Ubuntu's packaged `rustc` is
usually too old.

### openSUSE Tumbleweed

```bash
sudo zypper install rust cargo gcc clang pkg-config \
  gstreamer-devel gstreamer-plugins-base-devel \
  gstreamer-plugins-good gstreamer-plugins-bad gstreamer-plugins-ugly \
  gstreamer-plugins-libav
```

### Alpine Linux

```sh
sudo apk add rust cargo build-base clang pkgconf \
  gstreamer-dev gst-plugins-base-dev \
  gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav
```

Builds fine against musl — no special target needed.

### NixOS

A `flake.nix` is included — verified in a container:

```bash
nix develop --command bash -c 'rustup default stable && cargo build --release'
```

Or without flakes:

```bash
nix-shell -p rustup gcc pkg-config \
  gst_all_1.gstreamer gst_all_1.gst-plugins-base gst_all_1.gst-plugins-good \
  gst_all_1.gst-plugins-bad gst_all_1.gst-plugins-ugly gst_all_1.gst-libav \
  --run 'rustup default stable && cargo build --release'
```

nixpkgs' own `rustc` package can lag behind what this crate's dependencies
need — `rustup` inside the shell sidesteps that. VAAPI (the `va` plugin,
`vah264enc`/etc.) now lives in `gst-plugins-bad` since GStreamer 1.28 —
the older separate `gst-vaapi` package was removed upstream.

### Windows

Uses the Media Foundation backend (see above) — Media Foundation ships
with Windows itself, so unlike Linux there's no separate media framework
to install:

1. Install Rust via [rustup-init](https://rustup.rs), MSVC toolchain
   (`stable-x86_64-pc-windows-msvc`).
2. `cargo build --release`.

Cross-compilation to `x86_64-pc-windows-msvc` (via `cargo xwin`, which
provides the Windows SDK/CRT import libraries) was used to verify this
builds and links correctly during development; a real Windows machine was
not available to verify the built exe runs end-to-end.

### Linux desktop shortcut

A `.desktop` file is in [`packaging/linux`](packaging/linux); the app icon is
[`assets/icon.svg`](assets/icon.svg):

```bash
cp packaging/linux/wavefold.desktop ~/.local/share/applications/
mkdir -p ~/.local/share/icons/hicolor/scalable/apps
cp assets/icon.svg ~/.local/share/icons/hicolor/scalable/apps/wavefold.svg
update-desktop-database ~/.local/share/applications 2>/dev/null
gtk-update-icon-cache ~/.local/share/icons/hicolor 2>/dev/null
```

Requires the `wavefold` binary on `PATH`.

## Run

```bash
wavefold        # GUI (default)
wavefold gui    # same, explicit
wavefold encode <in> <out> [--cutoff F] [--encoder ...] [--backend gpu|cpu]
```

## License

WaveFold is licensed under the terms of GPLv3. Refer to [`LICENSE.md`](LICENSE.md).
