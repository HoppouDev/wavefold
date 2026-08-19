#!/bin/sh
# Shared by Dockerfile.aarch64-unknown-linux-gnu and
# Dockerfile.riscv64gc-unknown-linux-gnu. Installs GStreamer's dev
# packages for a foreign Debian architecture ($1, e.g. "arm64" or
# "riscv64") into a cross-rs base image.
#
# Packages are extracted with `dpkg-deb -x` rather than `apt-get install`
# proper: some of them (transitively, via libglib2.0-dev) pull in a
# foreign-arch python3 whose postinst script tries to *execute* that
# binary to precompile .pyc files - which needs qemu-user binfmt
# emulation for foreign-arch execution, unavailable in some sandboxed
# build hosts (confirmed: fails with "Exec format error" there even
# with binfmt_misc registered on the host). `dpkg-deb -x` only unpacks
# file contents (headers/.so/.pc), never runs maintainer scripts, so
# it sidesteps the problem entirely - fine here since this image is
# only ever used to compile against, never to run the foreign-arch
# packages.
#
# Extracting straight to `/` breaks Ubuntu's usr-merge: some packages'
# archives contain a literal `./lib` directory entry, and unlike a real
# `dpkg -i` install, `dpkg-deb -x` extraction isn't symlink-aware, so it
# replaces the existing `/lib -> usr/lib` symlink with a real directory
# (confirmed: this made every already-installed file "under" /lib vanish,
# breaking the host x86_64 linker's own libc). Extracting to a staging
# root first and merging via a `tar` pipe avoids this - `cp -a` hits the
# exact same clobber (it implies --no-dereference), and even plain
# `tar -x` replaces a destination directory symlink by default unless
# told not to (confirmed both ways); `--keep-directory-symlink` is what
# actually makes it write through the symlink instead.
set -eu

arch="$1"
stage="/tmp/${arch}-root"

dpkg --add-architecture "$arch"
apt-get update
apt-get install --assume-yes --no-install-recommends --download-only \
  "libgstreamer1.0-dev:${arch}" \
  "libgstreamer-plugins-base1.0-dev:${arch}"

mkdir -p "$stage"
for deb in /var/cache/apt/archives/*_"${arch}".deb; do
  dpkg-deb -x "$deb" "$stage"
done
(cd "$stage" && tar -cf - .) | tar -xf - -C / --keep-directory-symlink

rm -rf "$stage" /var/cache/apt/archives/*.deb /var/lib/apt/lists/*
