use color_eyre::Result;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use serde_derive::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectLocation {
    LocalPath(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Object {
    pub locations: Vec<ObjectLocation>,
    pub embedding_id: Option<usize>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ObjectId {
    pub data: [u8; 32],
}

#[derive(Default, Deserialize, Serialize)]
pub struct Database {
    pub by_id: HashMap<ObjectId, Object>,
    pub id_by_loc: HashMap<ObjectLocation, ObjectId>,
    pub clip_embeddings: Vec<Vec<f32>>,
}

pub fn compute_id_file(fh: &mut std::fs::File) -> Result<ObjectId> {
    let mut hw = blake3::Hasher::new();
    hw.update_reader(fh)?;
    Ok(ObjectId {
        data: hw.finalize().into(),
    })
}

impl Database {
    pub fn oid_has_embedding(&self, oid: &ObjectId) -> bool {
        if let Some(obj) = self.by_id.get(oid) {
            obj.embedding_id.is_some()
        } else {
            false
        }
    }
    pub fn loc_has_embedding(&self, loc: &ObjectLocation) -> bool {
        if let Some(oid) = self.id_by_loc.get(loc) {
            return self.oid_has_embedding(oid);
        } else {
            false
        }
    }
    pub fn insert(&mut self, loc: ObjectLocation, id: ObjectId, obj: Object) {
        self.by_id.insert(id, obj);
        self.id_by_loc.insert(loc, id);
    }
    pub fn compact_embeddings(&mut self) {
        let referenced_embeddings: HashSet<_> = self
            .by_id
            .values()
            .flat_map(|obj| obj.embedding_id)
            .collect();
        if referenced_embeddings.len() == self.clip_embeddings.len() {
            // already compact
            return;
        }
        log::info!("Compacting embedding list...");
        let mut new_embeddings = vec![];
        let mut remap = HashMap::new();
        for old_eid in referenced_embeddings.into_iter() {
            let emb = &self.clip_embeddings[old_eid];
            let new_eid = new_embeddings.len();
            new_embeddings.push(emb.to_owned());
            remap.insert(old_eid, new_eid);
        }
        let discarded = self.clip_embeddings.len() - new_embeddings.len();
        log::info!("Discarded {discarded} unreferenced embeddings.");
        self.clip_embeddings = new_embeddings;
        for obj in self.by_id.values_mut() {
            if let Some(old_eid) = obj.embedding_id {
                obj.embedding_id = Some(remap[&old_eid]);
            }
        }
    }
    pub fn persist(&mut self, config: &crate::Config) -> Result<()> {
        self.compact_embeddings();
        let encoded = bincode::serialize(self)?;
        if !config.db_path.exists() {
            std::fs::create_dir(&config.db_path)?
        }
        let temppath = config.db_path.join("db.bc.new");
        let dbpath = config.db_path.join("db.bc");
        let mut outf = std::fs::File::create(&temppath)?;
        let mut rdbslice = &encoded[..];
        std::io::copy(&mut rdbslice, &mut outf)?;
        std::fs::rename(temppath, dbpath)?;
        Ok(())
    }
}
