use axum::http::StatusCode;
use color_eyre::{eyre::eyre, Result};
use image::imageops::FilterType;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use jwalk::WalkDir;
use serde_hex::{SerHex, Strict};

use norman::special::NormEucl;
use ordered_float::NotNan;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use sha2::Digest;
use std::{
    borrow::Cow,
    collections::HashMap,
    io::Write,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokenizers::PaddingParams;

use serde_derive::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CLIPModelInfo {
    image_dim: (u32, u32),
    image_mean: (f32, f32, f32),
    image_std: (f32, f32, f32),
    text_tokenizer_hub_name: Cow<'static, str>,
    text_tokenizer_bos: Option<Cow<'static, str>>,
    text_tokenizer_pad: Cow<'static, str>,
    text_tokenizer_eos: Cow<'static, str>,
    text_input_size: usize,
    arch_name: Cow<'static, str>,
    pretrain_name: Cow<'static, str>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
enum PretrainedCLIP {
    VitL16SiglipWebli,
    VitL14Datacomp,
}

impl PretrainedCLIP {
    fn model_info(&self) -> CLIPModelInfo {
        match self {
            PretrainedCLIP::VitL16SiglipWebli => CLIPModelInfo {
                image_dim: (384, 384),
                image_mean: (0.5, 0.5, 0.5),
                image_std: (0.5, 0.5, 0.5),
                text_tokenizer_hub_name: "timm/ViT-L-16-SigLIP-384".into(),
                text_tokenizer_pad: "</s>".into(),
                text_tokenizer_eos: "</s>".into(),
                text_input_size: 64,
                arch_name: "ViT-L-16-SigLIP-384".into(),
                pretrain_name: "webli".into(),
                text_tokenizer_bos: None,
            },
            PretrainedCLIP::VitL14Datacomp => CLIPModelInfo {
                image_dim: (224, 224),
                image_mean: (0.48145466, 0.4578275, 0.40821073),
                image_std: (0.26862954, 0.261_302_6, 0.275_777_1),
                text_tokenizer_hub_name: "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K".into(),
                text_tokenizer_eos: "<|endoftext|>".into(),
                text_input_size: 77,
                arch_name: "ViT-L-14".into(),
                pretrain_name: "datacomp_xl_s13b_b90k".into(),
                text_tokenizer_bos: Some( "<|startoftext|>".into()),
                text_tokenizer_pad: todo!(),
            },
        }
    }
}

#[derive(Deserialize, Debug)]
struct Collection {
    roots: Vec<PathBuf>,
}

#[derive(Deserialize, Debug)]
struct Config {
    clip_model: PretrainedCLIP,
    collections: HashMap<String, Collection>,
    db_path: PathBuf,
    gpu_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum ObjectLocation {
    LocalPath(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
struct Object {
    locations: Vec<ObjectLocation>,
    embedding_id: Option<usize>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
struct ObjectId {
    data: [u8; 32],
}

#[derive(Default, Deserialize, Serialize)]
struct Database {
    by_id: HashMap<ObjectId, Object>,
    id_by_loc: HashMap<ObjectLocation, ObjectId>,
    clip_embeddings: Vec<Vec<f32>>,
}

#[derive(Default)]
struct HasherWriter {
    pub hasher: sha2::Sha256,
}

impl Write for HasherWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.hasher.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn compute_id(loc: ObjectLocation) -> Result<ObjectId> {
    match loc {
        ObjectLocation::LocalPath(p) => {
            let fh = std::fs::File::open(p)?;
            let mut hw = blake3::Hasher::new();
            hw.update_reader(fh)?;
            Ok(ObjectId {
                data: hw.finalize().into(),
            })
        }
    }
}

fn l2dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).powf(2.0))
        .sum::<f32>()
        .sqrt()
}
fn ip_dist(a: &[f32], b: &[f32]) -> f32 {
    1.0 - a.iter().zip(b.iter()).map(|(av, bv)| av * bv).sum::<f32>()
}

fn normalize(inp: &Vec<f32>) -> Vec<f32> {
    let norm = inp.norm_eucl();
    inp.iter().map(|v| v / norm).collect()
}

impl Database {
    fn has(&self, loc: &ObjectLocation) -> bool {
        self.id_by_loc.contains_key(loc)
    }
    fn insert(&mut self, loc: ObjectLocation, id: ObjectId, obj: Object) {
        self.by_id.insert(id, obj);
        self.id_by_loc.insert(loc, id);
    }
    fn persist(&self, config: &Config) -> Result<()> {
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
fn scan<P: AsRef<Path>>(db: Arc<Mutex<Database>>, root: P, config: &Config) -> Result<()> {
    let mut scanned = Vec::new();

    for entry in WalkDir::new(root) {
        let path = entry?.path();
        let loc = match path.extension().and_then(|s| s.to_str()) {
            Some("jpg") | Some("jpeg") | Some("png") => {
                ObjectLocation::LocalPath(std::fs::canonicalize(path)?)
            }
            _ => continue,
        };
        scanned.push(loc);
    }
    let scanned: Vec<ObjectLocation> = {
        let db = db.lock().expect("lock");
        scanned.into_iter().filter(|i| !db.has(i)).collect()
    };
    eprintln!("Found {} new items.", scanned.len());
    eprintln!("Hashing...");

    scanned
        .into_par_iter()
        .map(|l| -> Result<_> { Ok((l.clone(), compute_id(l)?)) })
        .progress()
        .for_each(|idc| {
            let (p, oid) = idc.expect("hashing failed");
            let obj = Object {
                locations: vec![p.clone()],
                ..Default::default()
            };
            let mut db = db.lock().expect("lock");
            db.insert(p, oid, obj);
        });

    db.lock().unwrap().persist(config)?;

    Ok(())
}

type DynF32Array = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>;

fn preprocess_image<P: AsRef<Path>>(path: P, config: &Config) -> Result<DynF32Array> {
    let minfo = config.clip_model.model_info();
    let img = image::io::Reader::open(&path)?
        .with_guessed_format()?
        .decode()?;
    let img = img.resize_exact(minfo.image_dim.0, minfo.image_dim.1, FilterType::Gaussian);
    let ibuf = img.to_rgb32f();
    let fs = ibuf.as_flat_samples();
    let ndimg = ndarray::ArrayView::from_shape(
        (minfo.image_dim.0 as usize, minfo.image_dim.1 as usize, 3),
        fs.samples,
    )?;
    let nmean = ndarray::array![minfo.image_mean.0, minfo.image_mean.1, minfo.image_mean.2];
    let nstd = ndarray::array![minfo.image_std.0, minfo.image_std.1, minfo.image_std.2];
    let ndimg = ndimg.to_owned() - nmean;
    let ndimg = ndimg / nstd;
    let ndimg = ndimg.permuted_axes([2, 1, 0]);

    Ok(ndimg.into_dyn().into_owned())
}

fn embed_images(
    clip_session: &mut ort::Session,
    images: Vec<DynF32Array>,
) -> Result<Vec<Vec<f32>>> {
    let views: Vec<_> = images.iter().map(|i| i.view()).collect();
    let batch = ndarray::stack(ndarray::Axis(0), &views[..])?;
    let prep = ndarray::CowArray::from(batch);
    let ov = ort::Value::from_array(clip_session.allocator(), &prep)?;
    let result = clip_session.run(vec![ov])?;
    let out: ort::tensor::OrtOwnedTensor<f32, _> = result[0].try_extract()?;
    Ok(out
        .view()
        .rows()
        .into_iter()
        .map(|lane| lane.into_iter().copied().collect())
        .collect())
}

use clap::{CommandFactory, Parser, Subcommand};

#[derive(Parser)]
struct Args {
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Commands>,
}
#[derive(Subcommand)]
enum Commands {
    UpdateDb,
    Serve,
    Search { query: String },
}
fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let config_raw =
        std::fs::read_to_string(args.config.unwrap_or_else(|| PathBuf::from("config.toml")))?;
    let config: Config = toml::from_str(&config_raw)?;
    let get_db = || {
        let db: Database = match std::fs::read(config.db_path.join("db.bc")) {
            Ok(fbsrc) => bincode::deserialize(&fbsrc)?,
            _ => Database::default(),
        };
        let db = Arc::new(Mutex::new(db));
        Ok::<_, color_eyre::Report>(db)
    };
    match args.command {
        None => {
            Args::command().print_help()?;
        }
        Some(Commands::Serve) => {
            serve_search(get_db()?, &config)?;
        }
        Some(Commands::UpdateDb) => {
            update_db(get_db()?, &config)?;
        }
        Some(Commands::Search { query }) => {
            search_db(get_db()?, &config, query)?;
        }
    }
    Ok(())
}

struct TextEmbedder {
    _ortenv: Arc<ort::Environment>,
    session: Option<ort::Session>,
    tokenizer: tokenizers::Tokenizer,
    input_ids: Vec<i64>,
    modelinfo: CLIPModelInfo,
}
impl TextEmbedder {
    fn new(config: &Config) -> Result<Self> {
        let gpu_id = config.gpu_id.unwrap_or(0);
        let modelinfo = config.clip_model.model_info();
        let tpath = PathBuf::from("clip_models").join(format!(
            "{}_{}_text.onnx",
            modelinfo.arch_name, modelinfo.pretrain_name
        ));

        let ortenv = ort::Environment::builder()
            .with_name("clip-text")
            .with_execution_providers([
                ort::ExecutionProvider::CUDA(
                    ort::execution_providers::CUDAExecutionProviderOptions {
                        device_id: gpu_id,
                        ..Default::default()
                    },
                ),
                ort::ExecutionProvider::CPU(Default::default()),
            ])
            .build()?
            .into_arc();
        let session = Some(
            ort::SessionBuilder::new(&ortenv)?
                .with_optimization_level(ort::GraphOptimizationLevel::Level1)?
                .with_model_from_file(tpath)?,
        );
        let tokenizer =
            tokenizers::Tokenizer::from_pretrained(modelinfo.text_tokenizer_hub_name.clone(), None)
                .map_err(|e| eyre!("{e:?}"))?;
        Ok(TextEmbedder {
            _ortenv: ortenv,
            session,
            tokenizer,
            input_ids: vec![0; modelinfo.text_input_size],
            modelinfo,
        })
    }
    fn embed_and_norm(&mut self, inp: &str) -> Result<Vec<f32>> {
        let sot = self.modelinfo.text_tokenizer_bos.as_ref().map(|t| self.tokenizer.token_to_id(t).unwrap());
        let eot = self.tokenizer.token_to_id(&self.modelinfo.text_tokenizer_eos).unwrap();
        let pad = self.tokenizer.token_to_id(&self.modelinfo.text_tokenizer_pad).unwrap();
        let query = self
            .tokenizer
            .encode(inp, false)
            .map_err(|e| eyre!("{e:?}"))?;
        self.input_ids.clear();
        if let Some(sot) = sot { self.input_ids.push(sot as i64); }
        self.input_ids
            .extend(query.get_ids().iter().map(|v| *v as i64));
        self.input_ids.push(eot as i64);
        self.input_ids
            .resize(self.modelinfo.text_input_size, pad as i64);

        eprintln!("Encoded query: {:?}", self.input_ids);
        let ids = ndarray::CowArray::from(&self.input_ids)
            .into_shape((1, self.modelinfo.text_input_size))?
            .into_dyn();
        let session = self.session.as_mut().unwrap();
        let result = session.run(vec![ort::Value::from_array(session.allocator(), &ids)?])?;
        let emb: ort::tensor::OrtOwnedTensor<f32, _> = result[0].try_extract()?;
        let embv: Vec<f32> = emb.view().iter().copied().collect();
        let embv = normalize(&embv);
        Ok(embv)
    }
}
impl Drop for TextEmbedder {
    fn drop(&mut self) {
        // `ort` bug - currently need to make sure the session cannot outlive the environment ourselves
        self.session.take();
    }
}

fn search_db(db: Arc<Mutex<Database>>, config: &Config, query: String) -> Result<()> {
    let mut te = TextEmbedder::new(config)?;

    {
        let embv = te.embed_and_norm(&query)?;
        let embv = normalize(&embv);

        let db = db.lock().unwrap();
        let eid_to_oid: HashMap<_, _> = db
            .by_id
            .iter()
            .filter_map(|(k, v)| Some((v.embedding_id?, k)))
            .collect();
        let mut scores = Vec::new();
        for (dbi, dbv) in db.clip_embeddings.iter().enumerate() {
            let ndbv = normalize(dbv);
            let dist = ip_dist(&embv, &ndbv);
            scores.push((dist, dbi));
        }
        scores.sort_by_key(|(d, _)| NotNan::new(*d).unwrap());
        for (dist, eid) in scores.iter().take(5) {
            let oid = eid_to_oid.get(eid).expect("eid-to-oid");
            let obj = db.by_id.get(oid).expect("by-id");
            eprintln!("{dist} {obj:?}")
        }
    }
    Ok(())
}

struct AppState {
    te: Arc<Mutex<TextEmbedder>>,
    db: Arc<Mutex<Database>>,
    query_cache: Arc<Mutex<lru::LruCache<String, Vec<(f32, usize)>>>>,
}

#[derive(Deserialize)]
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
        let hit = state.query_cache.lock().unwrap().get(&params.query).cloned();
        if let Some(scores) = hit {
            eprintln!("cache hit for {}", &params.query);
            Ok::<_, color_eyre::Report>(scores.clone())
        } else {
            let query = params.query.clone();
            let scores: Vec<(f32, usize)> = tokio::task::spawn_blocking(move || -> Result<_> {
                let mut te = te.lock().unwrap();
                let embv = te.embed_and_norm(&query)?;
                let embv = normalize(&embv);

                let db = db.lock().unwrap();
                let mut scores = Vec::new();
                for (dbi, dbv) in db.clip_embeddings.iter().enumerate() {
                    let ndbv = normalize(dbv);
                    let dist = ip_dist(&embv, &ndbv);
                    scores.push((dist, dbi));
                }
                scores.sort_by_key(|(d, _)| NotNan::new(*d).unwrap());
                Ok(scores)
            })
            .await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            state
                .query_cache
                .lock()
                .unwrap()
                .put(params.query.clone(), scores.clone());
            Ok(scores)
        }
    }
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

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
        .collect::<Result<Vec<_>>>()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

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
fn serve_search(db: Arc<Mutex<Database>>, config: &Config) -> Result<()> {
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

fn update_db(db: Arc<Mutex<Database>>, config: &Config) -> Result<()> {
    eprintln!("Db has {} items.", db.lock().unwrap().by_id.len());
    eprintln!("Scanning...");
    for (_cname, coll) in config.collections.iter() {
        for root in coll.roots.iter() {
            scan(db.clone(), root, config)?;
        }
    }
    let gpu_id = config.gpu_id.unwrap_or(0);
    let minfo = config.clip_model.model_info();
    let vpath = PathBuf::from("clip_models").join(format!(
        "{}_{}_visual.onnx",
        minfo.arch_name, minfo.pretrain_name
    ));
    let ortenv = ort::Environment::builder()
        .with_name("clip")
        .with_execution_providers([
            ort::ExecutionProvider::TensorRT(
                ort::execution_providers::TensorRTExecutionProviderOptions {
                    device_id: gpu_id,
                    fp16_enable: true,
                    engine_cache_enable: true,
                    engine_cache_path: "./engine_cache".to_owned(),
                    ..Default::default()
                },
            ),
            ort::ExecutionProvider::CUDA(ort::execution_providers::CUDAExecutionProviderOptions {
                device_id: gpu_id,
                ..Default::default()
            }),
            ort::ExecutionProvider::CPU(Default::default()),
        ])
        .build()?
        .into_arc();
    let mut clip_visual_session = ort::SessionBuilder::new(&ortenv)?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .with_model_from_file(vpath)?;

    {
        eprintln!("Embedding...");
        let path_and_ids: Vec<(ObjectId, String)> = db
            .lock()
            .unwrap()
            .by_id
            .iter()
            .filter(|(_, obj)| obj.embedding_id.is_none())
            .map(|(id, obj)| {
                (
                    *id,
                    match obj.locations[0].clone() {
                        ObjectLocation::LocalPath(p) => p.to_string_lossy().to_string(),
                    },
                )
            })
            .collect();
        let (tx, rx) = std::sync::mpsc::sync_channel(64);
        std::thread::scope(|s| -> Result<()> {
            s.spawn(move || {
                path_and_ids
                    .par_iter()
                    .progress_with_style(
                        ProgressStyle::default_bar()
                            .template("{wide_bar} {per_sec} {eta} {elapsed}")
                            .unwrap(),
                    )
                    .for_each(|(id, path)| {
                        let r = preprocess_image(path, config);
                        match r {
                            Ok(embedding) => {
                                tx.send(Ok((id.to_owned(), embedding))).unwrap();
                            }
                            Err(e) => {
                                tx.send(Err(e)).unwrap();
                            }
                        }
                    });
            });
            let bs = 48;
            let mut done = false;
            while !done {
                let mut imgbatch = vec![];
                let mut ids = vec![];
                loop {
                    match rx.recv() {
                        Ok(msg) => {
                            let (oid, preprocd) = msg?;
                            ids.push(oid);
                            imgbatch.push(preprocd);
                        }
                        Err(_e) => {
                            done = true;
                            break;
                        }
                    }
                    if imgbatch.len() >= bs {
                        break;
                    }
                }
                if !imgbatch.is_empty() {
                    let mut embeddings = embed_images(&mut clip_visual_session, imgbatch)?;
                    let mut db = db.lock().unwrap();
                    let sidx = db.clip_embeddings.len();
                    db.clip_embeddings.append(&mut embeddings);
                    for (oid, eid) in ids.iter().zip(sidx..) {
                        db.by_id.get_mut(oid).unwrap().embedding_id = Some(eid);
                    }
                }
            }
            //preproc_thread.join().map_err(|e| eyre!("{e:?}"))?;
            Ok(())
        })?;
        db.lock().unwrap().persist(config)?;
    }

    Ok(())
}
