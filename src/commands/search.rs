use color_eyre::Result;

use ordered_float::NotNan;

use crate::mmindex::{
    database::Database,
    math::{cos_sim, normalize},
};

use std::{
    cmp::Reverse,
    collections::HashMap,
    sync::{Arc, Mutex},
};

pub fn search_db(db: Arc<Mutex<Database>>, config: &crate::Config, query: String) -> Result<()> {
    let mut te = crate::mmindex::embedding::TextEmbedder::new(config)?;

    {
        let embv = te.embed_text(&query)?;
        let embv = crate::mmindex::math::normalize(&embv);

        let db = db.lock().unwrap();
        let eid_to_oid: HashMap<_, _> = db
            .by_id
            .iter()
            .filter_map(|(k, v)| Some((v.embedding_id?, k)))
            .collect();
        let mut scores: Vec<_> = db
            .clip_embeddings
            .iter()
            .enumerate()
            .map(|(dbi, dbv)| {
                let dist = cos_sim(&embv, dbv);
                (dist, dbi)
            })
            .collect();
        scores.sort_by_key(|(d, _)| Reverse(NotNan::new(*d).unwrap()));
        for (dist, eid) in scores.iter().take(5) {
            let oid = eid_to_oid.get(eid).expect("eid-to-oid");
            let obj = db.by_id.get(oid).expect("by-id");
            println!("{dist} {obj:?}")
        }
    }
    Ok(())
}
