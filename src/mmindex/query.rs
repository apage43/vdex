use color_eyre::{eyre::eyre, Result};
use serde_hex::{SerHex, Strict};
use std::sync::Arc;

use serde_derive::{Deserialize, Serialize};

use crate::mmindex::{
    database::{Database, ObjectId},
    embedding::TextEmbedder,
    math::normalize,
};

use color_eyre::eyre::bail;

use std::sync::Mutex;

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq, Hash)]
pub enum EmbeddingExpr {
    LookupByObjectId {
        #[serde(with = "SerHex::<Strict>")]
        object_id: [u8; 32],
    },
    EmbedText(String),
    Mean(Vec<EmbeddingExpr>),
    Normalize(Box<EmbeddingExpr>),
    Negate(Box<EmbeddingExpr>),
    Sum(Vec<EmbeddingExpr>),
}

pub fn compute_embexpr(
    db: Arc<Mutex<Database>>,
    text_embedder: Arc<Mutex<TextEmbedder>>,
    expr: &EmbeddingExpr,
) -> Result<Vec<f32>> {
    match expr {
        EmbeddingExpr::LookupByObjectId { object_id } => {
            let db = db.lock().unwrap();
            let obj = db
                .by_id
                .get(&ObjectId { data: *object_id })
                .ok_or_else(|| eyre!("no oid match"))?;
            let eid = obj.embedding_id.ok_or_else(|| eyre!("no eid for obj"))?;
            let emb = db
                .clip_embeddings
                .get(eid)
                .ok_or_else(|| eyre!("bad eid"))?;
            Ok(emb.to_owned())
        }
        EmbeddingExpr::EmbedText(text) => {
            let mut te = text_embedder.lock().unwrap();
            let emb = te.embed_text(text)?;
            Ok(emb)
        }
        EmbeddingExpr::Mean(exprs) => {
            let embs: Vec<Vec<f32>> = exprs
                .iter()
                .map(|expr| compute_embexpr(db.clone(), text_embedder.clone(), expr))
                .collect::<Result<_, color_eyre::Report>>()?;
            if embs.is_empty() {
                bail!("empty sum ");
            }
            let sum: Vec<f32> = (0..embs[0].len())
                .map(|i| embs.iter().map(|e| e[i]).sum::<f32>())
                .collect();
            Ok(sum.into_iter().map(|n| n / (exprs.len() as f32)).collect())
        }
        EmbeddingExpr::Normalize(expr) => {
            let emb = compute_embexpr(db.clone(), text_embedder.clone(), expr)?;
            Ok(normalize(&emb))
        }
        EmbeddingExpr::Sum(exprs) => {
            let embs: Vec<Vec<f32>> = exprs
                .iter()
                .map(|expr| compute_embexpr(db.clone(), text_embedder.clone(), expr))
                .collect::<Result<_, color_eyre::Report>>()?;
            if embs.is_empty() {
                bail!("empty sum ");
            }
            Ok((0..embs[0].len())
                .map(|i| embs.iter().map(|e| e[i]).sum::<f32>())
                .collect())
        }
        EmbeddingExpr::Negate(expr) => {
            let src = compute_embexpr(db.clone(), text_embedder.clone(), expr)?;
            Ok(src.into_iter().map(|n| -n).collect())
        }
    }
}

#[cfg(test)]
mod test {
    use crate::mmindex::query::EmbeddingExpr;

    #[test]
    fn serialize_expr() -> color_eyre::Result<()> {
        let expr = EmbeddingExpr::Normalize(Box::new(EmbeddingExpr::Sum(vec![
            EmbeddingExpr::EmbedText("king".to_owned()),
            EmbeddingExpr::Negate(Box::new(EmbeddingExpr::EmbedText("man".to_owned()))),
            EmbeddingExpr::EmbedText("woman".to_owned()),
        ])));
        eprintln!("{}", serde_json::to_string(&expr)?);
        Ok(())
    }
}
