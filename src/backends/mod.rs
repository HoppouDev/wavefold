#[cfg(not(windows))]
pub mod gstreamer;

#[cfg(windows)]
pub mod media_foundation;
