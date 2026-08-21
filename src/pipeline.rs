pub use crate::media_backend::PipelineMsg;

use crate::codec::Codec;
use crate::dct_backend::{ComputeBackend, DctAlgorithm};
use crate::media_backend::MediaBackendChoice;
use std::path::Path;
use tokio::sync::mpsc::UnboundedSender as Sender;
use tracing::error;

/// Resolves `compute_backend` into a concrete `DctBackend` once (shared by
/// whichever `MediaBackend` runs the actual encode — the DCT compute
/// implementation is orthogonal to which decode/encode backend is used),
/// then hands off to `media_backend`'s implementation. `dct_algorithm`
/// only affects `ComputeBackend::Gpu` (see `DctAlgorithm`'s doc comment).
pub fn run(
    input: &Path,
    output: &Path,
    cutoff: f32,
    codec: Codec,
    compute_backend: ComputeBackend,
    dct_algorithm: DctAlgorithm,
    media_backend: MediaBackendChoice,
    tx: Sender<PipelineMsg>,
) {
    let _ = tx.send(PipelineMsg::Log(format!("initializing {compute_backend} DCT backend ({dct_algorithm})...")));
    let dct = match compute_backend.build(dct_algorithm) {
        Ok(dct) => dct,
        Err(e) => {
            error!("failed to initialize DCT backend: {e:#}");
            let _ = tx.send(PipelineMsg::Error(format!("{e:#}")));
            return;
        }
    };

    if let Err(e) = media_backend.build().run(input, output, cutoff, codec, dct, tx.clone()) {
        error!("pipeline failed: {e:#}");
        let _ = tx.send(PipelineMsg::Error(format!("{e:#}")));
    }
}
