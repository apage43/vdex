use crate::{
    mmindex::{
        database::{Database, ObjectId, ObjectLocation},
        embedding::TextEmbedder,
        math::cos_sim,
    },
    Config,
};
use axum::http::StatusCode;
use color_eyre::{
    eyre::{bail, eyre},
    Result,
};
use ordered_float::NotNan;
use serde_derive::{Deserialize, Serialize};
use serde_hex::{SerHex, Strict};
use std::{
    cmp::Reverse,
    collections::HashMap,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

type QueryCache = lru::LruCache<String, Vec<(f32, usize)>>;
struct AppState {
    te: Arc<Mutex<TextEmbedder>>,
    db: Arc<Mutex<Database>>,
    query_cache: Arc<Mutex<QueryCache>>,
}

#[derive(Deserialize, Debug)]
struct SearchParams {
    query: String,
    limit: Option<usize>,
    skip: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct NNSearchResult {
    distance: f32,
    #[serde(with = "SerHex::<Strict>")]
    object_id: [u8; 32],
    path: url::Url,
}

#[derive(Serialize, Deserialize)]
struct NNSearchResponse {
    items: Vec<NNSearchResult>,
    more: bool,
}

#[axum::debug_handler]
async fn handle_search(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    params: axum::extract::Query<SearchParams>,
) -> Result<axum::Json<NNSearchResponse>, StatusCode> {
    let te = state.te.clone();
    let db = state.db.clone();
    let skip = params.skip.unwrap_or(0);
    let limit = params.limit.unwrap_or(5);

    let scores = {
        let hit = state
            .query_cache
            .lock()
            .unwrap()
            .get(&params.query)
            .cloned();
        if let Some(scores) = hit {
            log::info!("(hit) {:?}", &params);
            Ok::<_, color_eyre::Report>(scores.clone())
        } else {
            log::info!("(miss) {:?}", &params);
            let query = params.query.clone();
            let scores: Vec<(f32, usize)> = tokio::task::spawn_blocking(move || -> Result<_> {
                let embv = if query.starts_with("similar:") {
                    let (_, oid) = query.trim().split_once(':').ok_or_else(|| eyre!("split"))?;
                    if oid.len() < 64 {
                        bail!("bad oid");
                    }
                    let oid: [u8; 32] = SerHex::<serde_hex::Strict>::from_hex(oid)?;
                    let db = db.lock().unwrap();
                    let oid = ObjectId { data: oid };
                    if let Some(obj) = db.by_id.get(&oid) {
                        if let Some(eid) = obj.embedding_id {
                            if let Some(emb) = db.clip_embeddings.get(eid) {
                                //normalize(emb)
                                emb.clone()
                            } else {
                                bail!("no embedding");
                            }
                        } else {
                            bail!("no eid");
                        }
                    } else {
                        bail!("no object");
                    }
                } else {
                    let mut te = te.lock().unwrap();
                    te.embed_text(&query)?
                };

                let db = db.lock().unwrap();
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
                Ok(scores)
            })
            .await
            .map_err(|e| {
                log::warn!("query failed: {e:?}");
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .map_err(|e| {
                log::warn!("bad request: {e:?}");
                StatusCode::BAD_REQUEST
            })?;

            state
                .query_cache
                .lock()
                .unwrap()
                .put(params.query.clone(), scores.clone());
            Ok(scores)
        }
    }
    .map_err(|e| {
        log::warn!("lookup failed: {e:?}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let db = state.db.lock().unwrap();
    let eid_to_oid: HashMap<_, _> = db
        .by_id
        .iter()
        .filter_map(|(k, v)| Some((v.embedding_id?, k)))
        .collect();
    let rv = scores
        .iter()
        .skip(skip)
        .take(limit)
        .map(|(distance, eid)| {
            let object_id = **eid_to_oid.get(eid).ok_or(eyre!("eid/oid"))?;
            let object = db
                .by_id
                .get(&object_id)
                .ok_or(eyre!("oid mismatch"))?
                .clone();
            let ObjectLocation::LocalPath(p) = object.locations[0].clone();
            Ok(NNSearchResult {
                distance: *distance,
                object_id: object_id.data,
                path: url::Url::from_file_path(p).unwrap(),
            })
        })
        .flat_map(|r: Result<_, color_eyre::Report>| match r {
            Ok(r) => Some(r),
            Err(e) => {
                log::warn!("error resolving eid to obj: {e:?}");
                None
            }
        })
        .collect::<Vec<_>>();

    Ok(axum::Json(NNSearchResponse {
        items: rv,
        more: skip + limit < db.by_id.len(),
    }))
}
async fn serve(app: axum::Router, port: u16) -> Result<()> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;
    Ok(())
}
pub fn serve_search(db: Arc<Mutex<Database>>, config: &Config) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    let state = Arc::new(AppState {
        te: Arc::new(Mutex::new(TextEmbedder::new(config)?)),
        db,
        query_cache: Arc::new(Mutex::new(lru::LruCache::new(
            NonZeroUsize::new(5).unwrap(),
        ))),
    });
    let app = axum::Router::new()
        .route("/search_text", axum::routing::get(handle_search))
        .with_state(state);

    rt.block_on(async move { serve(app, 6680).await })?;
    Ok(())
}
