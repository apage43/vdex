use color_eyre::{eyre::eyre, Result};
use pest::{
    iterators::{Pair, Pairs},
    pratt_parser::{Assoc, Op, PrattParser},
    Parser,
};
use pest_derive::Parser;
use serde_derive::{Deserialize, Serialize};
use serde_hex::{SerHex, Strict};
use std::sync::Arc;

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

impl EmbeddingExpr {
    fn collapse_sum(self) -> Self {
        match self {
            EmbeddingExpr::Sum(els) => EmbeddingExpr::Sum(
                els.into_iter()
                    .flat_map(|el| match el {
                        EmbeddingExpr::Sum(inner_els) => inner_els.into_iter(),
                        other => vec![other].into_iter(),
                    })
                    .collect(),
            ),
            other => other,
        }
    }
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

#[derive(Parser)]
#[grammar = "mmindex/embexpr.pest"] // relative to src
pub struct EmbeddingExprParser;

fn parse_tree_to_expr(input: Pairs<crate::mmindex::query::Rule>) -> Result<EmbeddingExpr> {
    let pratt = PrattParser::new()
        .op(Op::infix(Rule::add, Assoc::Left) | Op::infix(Rule::sub, Assoc::Left))
        .op(Op::prefix(Rule::neg));
    pratt
        .map_primary(|primary| match primary.as_rule() {
            Rule::par_expr => parse_tree_to_expr(primary.into_inner().next().unwrap().into_inner()), // from "(" ~ expr ~ ")"
            Rule::embed_text => {
                let str = primary
                    .into_inner()
                    .next()
                    .ok_or_else(|| eyre!("unwrapping"))?
                    .into_inner()
                    .next()
                    .ok_or_else(|| eyre!("unwrapping"))?
                    .as_str();
                Ok(EmbeddingExpr::EmbedText(str.to_owned()))
            }
            Rule::lookup => {
                let str = primary
                    .into_inner()
                    .next()
                    .ok_or_else(|| eyre!("unwrapping"))?
                    .into_inner()
                    .next()
                    .ok_or_else(|| eyre!("unwrapping"))?
                    .as_str();
                Ok(EmbeddingExpr::LookupByObjectId {
                    object_id: SerHex::<Strict>::from_hex(str)?,
                })
            }
            Rule::call => {
                let mut parts = primary.into_inner();
                let builtin = parts.next().ok_or_else(|| eyre!("unwrapping"))?.as_str();
                let subexprs: Vec<_> = parts
                    .map(|subexp| parse_tree_to_expr(subexp.into_inner()))
                    .collect::<Result<Vec<_>, color_eyre::Report>>()?;
                match builtin {
                    "normalize" | "N" => {
                        if subexprs.len() != 1 {
                            bail!("normalize() takes exactly one argument")
                        }
                        Ok(EmbeddingExpr::Normalize(Box::new(subexprs[0].clone())))
                    }
                    "mean" | "M" => {
                        if subexprs.is_empty() {
                            bail!("mean() takes at least one argument")
                        }
                        Ok(EmbeddingExpr::Mean(subexprs))
                    }
                    _ => bail!("builtin op {builtin:?} does not exist"),
                }
            }
            r => unimplemented!("{r:?}"),
        })
        .map_prefix(|prefix, p| match prefix.as_rule() {
            Rule::neg => Ok(EmbeddingExpr::Negate(Box::new(p?))),
            _ => unimplemented!(),
        })
        .map_infix(|lhs, op, rhs| match op.as_rule() {
            Rule::add => Ok(EmbeddingExpr::Sum(vec![lhs?, rhs?]).collapse_sum()),
            Rule::sub => Ok(
                EmbeddingExpr::Sum(vec![lhs?, EmbeddingExpr::Negate(Box::new(rhs?))])
                    .collapse_sum(),
            ),
            _ => unimplemented!(),
        })
        .parse(input)
}

pub fn parse_expr(input: &str) -> Result<EmbeddingExpr> {
    let mut pres = EmbeddingExprParser::parse(Rule::emb_expr, input)?;
    parse_tree_to_expr(pres.next().ok_or_else(|| eyre!("no expr"))?.into_inner())
}

#[cfg(test)]
mod test {
    use crate::mmindex::query::EmbeddingExpr;
    use color_eyre::Result;

    #[test]
    fn test_parse_expr() -> Result<()> {
        let pres = super::parse_expr("N(@\"aa12bb34aa12bb34aa12bb34aa12bb34aa12bb34aa12bb34aa12bb34aa12bb34\" + M(\"yacht\" - \"boat\", \"fancy\"))")?;
        eprintln!("{pres:?}");
        Ok(())
    }

    #[test]
    fn serialize_expr() -> Result<()> {
        let expr = EmbeddingExpr::Normalize(Box::new(EmbeddingExpr::Sum(vec![
            EmbeddingExpr::EmbedText("king".to_owned()),
            EmbeddingExpr::Negate(Box::new(EmbeddingExpr::EmbedText("man".to_owned()))),
            EmbeddingExpr::EmbedText("woman".to_owned()),
        ])));
        eprintln!("{}", serde_json::to_string(&expr)?);
        Ok(())
    }
}
