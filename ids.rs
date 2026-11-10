use serde::{Deserialize, Serialize};

macro_rules! newtype_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub u64);
        impl $name { #[inline] pub const fn raw(self) -> u64 { self.0 } }
    };
}

newtype_id!(TrackId);
newtype_id!(ArtistId);
newtype_id!(AlbumId);
newtype_id!(UserId);
newtype_id!(GenreId);
newtype_id!(LangId);