fn main() {
    // `#[cfg(windows)]` here would reflect the HOST platform, not the
    // cross-compilation target (build.rs always compiles for the host) -
    // embed_resource::compile() itself checks $TARGET at runtime and is a
    // safe no-op (CompilationResult::NotWindows) on non-Windows targets.
    embed_resource::compile("assets/windows/app.rc", embed_resource::NONE)
        .manifest_optional()
        .expect("failed to embed Windows exe icon resource");
}
