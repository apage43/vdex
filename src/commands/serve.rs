use crate::{
    mmindex::{
        database::{Database, Object, ObjectId, ObjectLocation},
        embedding::TextEmbedder,
        math::cos_sim,
        query::EmbeddingExpr,
    },
    Config,
};
use axum::{
    http::StatusCode,
    response::IntoResponse,
};
use color_eyre::{eyre::eyre, Result};
use ordered_float::NotNan;
use serde_derive::{Deserialize, Serialize};
use serde_hex::{SerHex, Strict};
use std::{
    cmp::Reverse,
    collections::HashMap,
    io::Write,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};
use tower::ServiceExt;

type QueryCache = lru::LruCache<EmbeddingExpr, Vec<(f32, usize)>>;
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
    let query: EmbeddingExpr = crate::mmindex::query::parse_expr(&params.query).map_err(|e| {
        log::warn!("query parsing failed: {e:?}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let scores = {
        let hit = state.query_cache.lock().unwrap().get(&query).cloned();
        if let Some(scores) = hit {
            log::info!("(hit) {:?}", &params);
            Ok::<_, color_eyre::Report>(scores.clone())
        } else {
            log::info!("(miss) {:?}", &params);
            let cquery = query.clone();
            let scores: Vec<(f32, usize)> = tokio::task::spawn_blocking(move || -> Result<_> {
                let embv = crate::mmindex::query::compute_embexpr(db.clone(), te.clone(), &cquery)?;

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

            state.query_cache.lock().unwrap().put(query, scores.clone());
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
async fn handle_serve_image(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::extract::Path(object_id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    if let Some(path) = {
        let db = state.db.lock().unwrap();
        log::info!("lookup oid: {object_id}");
        let oid: ObjectId = ObjectId {
            data: SerHex::<Strict>::from_hex(object_id).map_err(|_| StatusCode::NOT_ACCEPTABLE)?,
        };
        let obj = if let Some(obj) = db.by_id.get(&oid) {
            obj
        } else {
            return Err(StatusCode::NOT_FOUND);
        };
        log::info!("Got obj: {obj:?}");
        obj.locations
            .iter()
            .find_map(|loc| {
                let ObjectLocation::LocalPath(p) = loc;
                if p.exists() {
                    Some(p)
                } else {
                    None
                }
            })
            .cloned()
    } {
        log::info!("found, local path: {path:?}");
        let tfile = tokio::fs::File::open(path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)?;
        let stream = tokio_util::io::ReaderStream::new(tfile);
        Ok(axum::body::Body::from_stream(stream))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}
#[derive(Serialize, Debug)]
pub struct ObjectMetadata {
    #[serde(with = "SerHex::<Strict>")]
    oid: [u8; 32],
}
async fn handle_export_metadata(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Result<axum::Json<Vec<ObjectMetadata>>, StatusCode> {
    let db = state.db.lock().unwrap();
    let by_eid: HashMap<usize, ObjectId> = db
        .by_id
        .iter()
        .filter_map(|(oid, obj)| Some((obj.embedding_id?, *oid)))
        .collect();
    Ok(axum::Json(
        (0..db.clip_embeddings.len())
            .map(|eid| ObjectMetadata {
                oid: by_eid.get(&eid).unwrap().data,
            })
            .collect(),
    ))
}
async fn handle_export_embeddings(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Result<axum::body::Bytes, StatusCode> {
    let db = state.db.lock().unwrap();
    if db.clip_embeddings.is_empty() {
        return Err(StatusCode::NOT_FOUND);
    }
    let emb_dim = db.clip_embeddings.first().unwrap().len();
    let emb_count = db.clip_embeddings.len();
    let hdr_meta_pfx = r#"{'descr':'<f4','fortran_order':False,'shape':"#;
    let hdr_meta = format!("{hdr_meta_pfx}({emb_count},{emb_dim}){}", r"}");
    let hdr_len = 6 + 2 + 2; // 6 byte magic + 2 bytes for version + u16 len
    let padded_len = (hdr_meta.len() + hdr_len).div_ceil(64) * 64;
    let mut response: Vec<u8> = vec![0x93];
    response.extend_from_slice(b"NUMPY");
    response.push(1); // v1.0
    response.push(0);
    response
        .write_all(&((padded_len - hdr_len) as u16).to_le_bytes())
        .unwrap();
    response.write_all(hdr_meta.as_bytes()).unwrap();
    response.resize((padded_len - 1) as usize, b' ');
    response.push(b'\n');
    for emb in db.clip_embeddings.iter() {
        for el in emb {
            response.write_all(&el.to_le_bytes()).unwrap();
        }
    }
    Ok(response.into())
}
async fn serve(app: axum::Router, port: u16) -> Result<()> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    log::info!("Listening on {addr:?}");
    axum::serve(
        tokio::net::TcpListener::bind(addr).await?,
        app.into_make_service(),
    )
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
        .route(
            "/export_embeddings_npy",
            axum::routing::get(handle_export_embeddings),
        )
        .route(
            "/export_metadata",
            axum::routing::get(handle_export_metadata),
        )
        .route("/image_by_oid/:oid", axum::routing::get(handle_serve_image))
        .with_state(state);

    rt.block_on(async move { serve(app, 6680).await })?;
    Ok(())
}
