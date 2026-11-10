use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use crate::adaptive::ArmState;
use crate::bandit::{Arm, BetaState};
use crate::config::PersistenceConfig;
use crate::context::Device;
use crate::error::Result;
use crate::ids::{AlbumId, ArtistId, GenreId, TrackId, UserId};
use crate::profile::UserProfile;
use crate::cooccurrence::CoOccurrence;
use crate::markov::MarkovChain;
use crate::usermap::UserKnn;

#[derive(Serialize, Deserialize)]
pub struct ProfileFile { pub profile: UserProfile }

#[derive(Serialize, Deserialize)]
pub struct CoocFile { pub cooc: CoOccurrence }

#[derive(Serialize, Deserialize)]
pub struct MarkovFile { pub markov: MarkovChain }

#[derive(Serialize, Deserialize)]
pub struct UsermapFile { pub usermap: UserKnn }

#[derive(Serialize, Deserialize)]
pub struct BanditFile { pub state: Vec<(Arm, (f64, f64))>, pub seed: u64 }

#[derive(Serialize, Deserialize)]
pub struct PolicyFile {
    pub tracks:  Vec<(TrackId, ArmState)>,
    pub tau_hours: f32,
    pub shrink:  f32,
}

#[derive(Serialize, Deserialize)]
pub struct GlobalPolicyFile {
    pub artists: Vec<(ArtistId, ArmState)>,
    pub genres:  Vec<(GenreId, ArmState)>,
    pub global:  ArmState,
    pub tau_hours: f32,
    pub shrink:  f32,
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() { fs::create_dir_all(parent)?; }
    let tmp = path.with_extension("tmp");
    { let mut f = fs::File::create(&tmp)?; f.write_all(bytes)?; f.sync_all()?; }
    fs::rename(tmp, path)?;
    Ok(())
}
fn read_bytes(path: &Path) -> Result<Option<Vec<u8>>> {
    if !path.exists() { return Ok(None); }
    Ok(Some(fs::read(path)?))
}

pub struct Store { dir: PathBuf }

impl Store {
    pub fn new(cfg: &PersistenceConfig) -> Result<Self> {
        let dir = PathBuf::from(&cfg.snapshot_dir);
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }
    fn path(&self, user: UserId, name: &str) -> PathBuf {
        self.dir.join(format!("user_{}_{}.bin", user.raw(), name))
    }
    fn global_path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("global_{}.bin", name))
    }

    pub fn save_profile(&self, p: &UserProfile) -> Result<()> {
        let bytes = bincode::serialize(&ProfileFile { profile: p.clone_via_restore() })?;
        atomic_write(&self.path(p.user_id, "profile"), &bytes)
    }
    pub fn load_profile(&self, user: UserId) -> Result<Option<UserProfile>> {
        match read_bytes(&self.path(user, "profile"))? {
            None => Ok(None),
            Some(b) => Ok(Some(bincode::deserialize::<ProfileFile>(&b)?.profile)),
        }
    }
    pub fn load_all_profiles(&self) -> Result<Vec<UserProfile>> {
        let mut out = Vec::new();
        let entries = match fs::read_dir(&self.dir) { Ok(e) => e, Err(_) => return Ok(out) };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("user_") && name.ends_with("_profile.bin") {
                if let Ok(b) = fs::read(entry.path()) {
                    if let Ok(f) = bincode::deserialize::<ProfileFile>(&b) {
                        out.push(f.profile);
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn save_cooc(&self, c: &CoOccurrence) -> Result<()> {
        let bytes = bincode::serialize(&CoocFile { cooc: c.clone_via_restore() })?;
        atomic_write(&self.global_path("cooc"), &bytes)
    }
    pub fn load_cooc(&self) -> Result<Option<CoOccurrence>> {
        match read_bytes(&self.global_path("cooc"))? {
            None => Ok(None),
            Some(b) => Ok(Some(bincode::deserialize::<CoocFile>(&b)?.cooc)),
        }
    }

    pub fn save_markov(&self, m: &MarkovChain) -> Result<()> {
        let bytes = bincode::serialize(&MarkovFile { markov: m.clone_via_restore() })?;
        atomic_write(&self.global_path("markov"), &bytes)
    }
    pub fn load_markov(&self) -> Result<Option<MarkovChain>> {
        match read_bytes(&self.global_path("markov"))? {
            None => Ok(None),
            Some(b) => Ok(Some(bincode::deserialize::<MarkovFile>(&b)?.markov)),
        }
    }

    pub fn save_usermap(&self, u: &UserKnn) -> Result<()> {
        let bytes = bincode::serialize(&UsermapFile { usermap: u.clone_via_restore() })?;
        atomic_write(&self.global_path("usermap"), &bytes)
    }
    pub fn load_usermap(&self) -> Result<Option<UserKnn>> {
        match read_bytes(&self.global_path("usermap"))? {
            None => Ok(None),
            Some(b) => Ok(Some(bincode::deserialize::<UsermapFile>(&b)?.usermap)),
        }
    }

    pub fn save_bandit(&self, state: &[(Arm, (f64, f64))], seed: u64) -> Result<()> {
        let bytes = bincode::serialize(&BanditFile { state: state.to_vec(), seed })?;
        atomic_write(&self.global_path("bandit"), &bytes)
    }
    pub fn load_bandit(&self) -> Result<Option<(FxHashMap<Arm, BetaState>, u64)>> {
        match read_bytes(&self.global_path("bandit"))? {
            None => Ok(None),
            Some(b) => {
                let f: BanditFile = bincode::deserialize(&b)?;
                let map = f.state.into_iter().map(|(k, (a, b))| (k, BetaState { alpha: a, beta: b })).collect();
                Ok(Some((map, f.seed)))
            }
        }
    }

    pub fn save_policy(&self, tracks: &FxHashMap<TrackId, ArmState>, tau: f32, shrink: f32) -> Result<()> {
        let f = PolicyFile {
            tracks: tracks.iter().map(|(k, v)| (*k, *v)).collect(),
            tau_hours: tau, shrink,
        };
        atomic_write(&self.global_path("policy"), &bincode::serialize(&f)?)
    }
    pub fn load_policy(&self) -> Result<Option<PolicyFile>> {
        match read_bytes(&self.global_path("policy"))? {
            None => Ok(None),
            Some(b) => Ok(Some(bincode::deserialize(&b)?)),
        }
    }

    pub fn save_global_policy(&self, artists: &FxHashMap<ArtistId, ArmState>, genres: &FxHashMap<GenreId, ArmState>, global: &ArmState, tau: f32, shrink: f32) -> Result<()> {
        let f = GlobalPolicyFile {
            artists: artists.iter().map(|(k, v)| (*k, *v)).collect(),
            genres:  genres .iter().map(|(k, v)| (*k, *v)).collect(),
            global: *global, tau_hours: tau, shrink,
        };
        atomic_write(&self.global_path("global_policy"), &bincode::serialize(&f)?)
    }
    pub fn load_global_policy(&self) -> Result<Option<GlobalPolicyFile>> {
        match read_bytes(&self.global_path("global_policy"))? {
            None => Ok(None),
            Some(b) => Ok(Some(bincode::deserialize(&b)?)),
        }
    }
}