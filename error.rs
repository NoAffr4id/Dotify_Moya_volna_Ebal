use thiserror::Error;
use crate::ids::{TrackId, UserId};

#[derive(Debug, Error)]
pub enum WaveError {
    #[error("track {0:?} missing from catalog")]
    TrackNotFound(TrackId),
    #[error("catalog is empty")]
    EmptyCatalog,
    #[error("embedding dim mismatch: expected={expected}, actual={actual}")]
    DimMismatch { expected: usize, actual: usize },
    #[error("insufficient candidates: requested={requested} available={available}")]
    InsufficientCandidates { requested: usize, available: usize },
    #[error("session user {session_user:?} != expected {expected:?}")]
    SessionUserMismatch { session_user: UserId, expected: UserId },
    #[error("invalid config: {0}")]
    Config(&'static str),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization: {0}")]
    Serde(#[from] bincode::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("index build: {0}")]
    Index(String),
}

pub type Result<T> = std::result::Result<T, WaveError>;