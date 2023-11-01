use axum::http::StatusCode;
use color_eyre::{
    eyre::{bail, eyre},
    Result,
};
use image::imageops::FilterType;

use indicatif::ProgressDrawTarget;
use jwalk::WalkDir;

use serde_hex::{SerHex, Strict};

use ordered_float::NotNan;

use sha2::Digest;
use std::{
    borrow::Cow,
    cmp::Reverse,
    collections::HashMap,
    io::{Seek, Write},
    num::NonZeroUsize,
    path::PathBuf,
    sync::{Arc, Mutex},
};

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

fn compute_id_file(fh: &mut std::fs::File) -> Result<ObjectId> {
    let mut hw = blake3::Hasher::new();
    hw.update_reader(fh)?;
    Ok(ObjectId {
        data: hw.finalize().into(),
    })
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let ip = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f32>();
    let mag_a = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let mag_b = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    ip / (mag_a * mag_b)
}

fn normalize(inp: &Vec<f32>) -> Vec<f32> {
    let norm = inp.iter().map(|v| v.abs().powf(2.0)).sum::<f32>() / (inp.len() as f32);
    inp.iter().map(|v| v / norm).collect()
}

impl Database {
    fn has_embedding(&self, loc: &ObjectLocation) -> bool {
        if let Some(oid) = self.id_by_loc.get(loc) {
            if let Some(obj) = self.by_id.get(oid) {
                obj.embedding_id.is_some()
            } else {
                false
            }
        } else {
            false
        }
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

type DynF32Array = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>;

fn preprocess_image(img: image::DynamicImage, config: &Config) -> Result<DynF32Array> {
    let minfo = &config.model_info;
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

#[derive(Deserialize, Debug)]
struct Config {
    model_info: CLIPModelInfo,
    db_path: PathBuf,
    gpu_id: u32,
}

#[derive(Parser)]
struct Args {
    #[arg(
        short,
        long,
        default_value = "./model_infos/ViT-L-16-SigLIP_webli.toml"
    )]
    model_config: PathBuf,
    #[arg(short, long, default_value = "./db")]
    db_path: PathBuf,
    #[arg(short, long, default_value = "0")]
    gpu_id: u32,
    #[command(subcommand)]
    command: Option<Commands>,
}
#[derive(Subcommand)]
enum Commands {
    UpdateDb {
        #[arg(short, long)]
        media_dirs: Vec<PathBuf>,
        #[arg(short, long, default_value = "8")]
        batch_size: usize,
        #[arg(short, long, default_value = "false")]
        tensorrt: bool,
        #[arg(short, long, default_value = "2")]
        readers: usize,
    },
    Serve,
    Search {
        query: String,
    },
}
fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let model_info: CLIPModelInfo = toml::from_str(&std::fs::read_to_string(args.model_config)?)?;
    let config: Config = Config {
        db_path: args.db_path,
        gpu_id: args.gpu_id,
        model_info,
    };
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
        Some(Commands::UpdateDb {
            media_dirs,
            batch_size,
            tensorrt,
            readers,
        }) => {
            update_db(
                get_db()?,
                &config,
                &media_dirs,
                batch_size,
                tensorrt,
                readers,
            )?;
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
        let gpu_id = config.gpu_id;
        let modelinfo = &config.model_info;
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
            modelinfo: modelinfo.clone(),
        })
    }
    fn embed_text(&mut self, inp: &str) -> Result<Vec<f32>> {
        let sot = self
            .modelinfo
            .text_tokenizer_bos
            .as_ref()
            .map(|t| self.tokenizer.token_to_id(t).unwrap());
        let eot = self
            .tokenizer
            .token_to_id(&self.modelinfo.text_tokenizer_eos)
            .unwrap();
        let pad = self
            .tokenizer
            .token_to_id(&self.modelinfo.text_tokenizer_pad)
            .unwrap();
        let query = self
            .tokenizer
            .encode(inp, false)
            .map_err(|e| eyre!("{e:?}"))?;
        self.input_ids.clear();
        if let Some(sot) = sot {
            self.input_ids.push(sot as i64);
        }
        self.input_ids
            .extend(query.get_ids().iter().map(|v| *v as i64));
        self.input_ids.push(eot as i64);
        self.input_ids
            .resize(self.modelinfo.text_input_size, pad as i64);

        log::info!("Encoded query: {:?}", self.input_ids);
        let ids = ndarray::CowArray::from(&self.input_ids)
            .into_shape((1, self.modelinfo.text_input_size))?
            .into_dyn();
        let session = self.session.as_mut().unwrap();
        let result = session.run(vec![ort::Value::from_array(session.allocator(), &ids)?])?;
        let emb: ort::tensor::OrtOwnedTensor<f32, _> = result[0].try_extract()?;
        let embv: Vec<f32> = emb.view().iter().copied().collect();
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
        let embv = te.embed_text(&query)?;
        let embv = normalize(&embv);

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
                let ndbv = normalize(dbv);
                let dist = cos_sim(&embv, &ndbv);
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

fn update_db(
    db: Arc<Mutex<Database>>,
    config: &Config,
    scan_paths: &[PathBuf],
    batch_size: usize,
    use_tensorrt: bool,
    readers: usize,
) -> Result<()> {
    log::info!("Db has {} items.", db.lock().unwrap().by_id.len());
    let pb = indicatif::ProgressBar::new_spinner().with_style(indicatif::ProgressStyle::default_spinner().template("{spinner} Scanning... {pos} files").unwrap());
    let mut scanned = Vec::new();
    let gpu_id = config.gpu_id;
    let minfo = &config.model_info;
    let vpath = PathBuf::from("clip_models").join(format!(
        "{}_{}_visual.onnx",
        minfo.arch_name, minfo.pretrain_name
    ));
    for root in scan_paths.iter() {
        for entry in WalkDir::new(root) {
            let path = entry?.path();
            let loc = match path.extension().and_then(|s| s.to_str()) {
                Some("jpg") | Some("jpeg") | Some("png") => {
                    ObjectLocation::LocalPath(std::fs::canonicalize(path)?)
                }
                _ => continue,
            };
            scanned.push(loc);
            pb.inc(1);
        }
    }
    let scanned: Vec<ObjectLocation> = {
        let db = db.lock().expect("lock");
        scanned
            .into_iter()
            .filter(|i| !db.has_embedding(i))
            .collect()
    };
    pb.finish();
    log::info!("Found {} unembedded items.", scanned.len());
    log::info!("Processing...");

    let hasher_tasks = readers;
    let preprocess_tasks: usize = std::thread::available_parallelism()?.get();
    let db_inside = db.clone();
    rayon::scope(move |s| {
        let mp = indicatif::MultiProgress::new();
        mp.set_draw_target(ProgressDrawTarget::stderr_with_hz(30));
        let style = indicatif::ProgressStyle::default_bar()
            .template("{prefix} {wide_bar} {per_sec:<9} {eta} {pos:>7}/{len}")
            .unwrap();
        let pb_hash = mp
            .add(indicatif::ProgressBar::new(scanned.len() as u64))
            .with_style(style.clone())
            .with_prefix("   hash");
        let pb_rsz = mp
            .add(indicatif::ProgressBar::new(scanned.len() as u64))
            .with_style(style.clone())
            .with_prefix("prepare");
        let pb_emb = mp
            .add(indicatif::ProgressBar::new(scanned.len() as u64))
            .with_style(style.clone())
            .with_prefix("  embed");
        let db = db_inside;
        let prep_r = {
            let (fp_s, fp_r) = crossbeam_channel::bounded(hasher_tasks * 2);
            let (fpi_s, fpi_r) = crossbeam_channel::bounded(preprocess_tasks * 2);
            let (prep_s, prep_r) = crossbeam_channel::bounded(batch_size * 2);

            s.spawn(move |_| {
                for loc in scanned {
                    let ObjectLocation::LocalPath(ref p) = loc;
                    if let Ok(fp) = std::fs::File::open(p) {
                        fp_s.send((loc, fp)).unwrap();
                    } else {
                        log::warn!("failed to read file: {p:?}");
                    }
                }
            });

            (0..hasher_tasks).for_each(|_| {
                let fp_r = fp_r.clone();
                let fpi_s = fpi_s.clone();
                let db = db.clone();
                let pbh = pb_hash.clone();
                s.spawn(move |_| {
                    while let Ok((loc, mut fp)) = fp_r.recv() {
                        if let Ok(oid) = compute_id_file(&mut fp) {
                            {
                                pbh.inc(1);
                                let mut db = db.lock().unwrap();
                                db.insert(
                                    loc.clone(),
                                    oid,
                                    Object {
                                        locations: vec![loc.clone()],
                                        ..Default::default()
                                    },
                                );
                                fpi_s.send((loc, oid, fp)).unwrap();
                            }
                        } else {
                            pbh.inc(1);

                            log::warn!("error hashing file: {loc:?}");
                        }
                    }
                })
            });

            (0..preprocess_tasks).for_each(|_| {
                let fpi_r = fpi_r.clone();
                let prep_s = prep_s.clone();
                let pb_rsz = pb_rsz.clone();
                s.spawn(move |_| {
                    while let Ok((loc, oid, mut fp)) = fpi_r.recv() {
                        fp.rewind().expect("rewind");
                        if let Ok(img) = image::io::Reader::new(std::io::BufReader::new(fp))
                            .with_guessed_format()
                            .unwrap()
                            .decode()
                        {
                            pb_rsz.inc(1);
                            if let Ok(procd) = preprocess_image(img, config) {
                                prep_s.send((oid, procd)).unwrap();
                            } else {
                                log::warn!("Failed to process image for loc: {loc:?}");
                            }
                        } else {
                            pb_rsz.inc(1);

                            log::warn!("Failed to read image for loc: {loc:?}");
                        }
                    }
                })
            });
            prep_r
        };
        let mut eps = vec![];
        if use_tensorrt {
            eps.push(ort::ExecutionProvider::TensorRT(
                ort::execution_providers::TensorRTExecutionProviderOptions {
                    device_id: gpu_id,
                    fp16_enable: true,
                    engine_cache_enable: true,
                    engine_cache_path: "./engine_cache".to_owned(),
                    ..Default::default()
                },
            ));
        }
        eps.push(ort::ExecutionProvider::CUDA(
            ort::execution_providers::CUDAExecutionProviderOptions {
                device_id: gpu_id,
                ..Default::default()
            },
        ));
        eps.push(ort::ExecutionProvider::CPU(Default::default()));

        let ortenv = ort::Environment::builder()
            .with_name("clip")
            .with_execution_providers(&eps)
            .build()
            .expect("ort")
            .into_arc();
        let mut clip_visual_session = ort::SessionBuilder::new(&ortenv)
            .expect("ort")
            .with_optimization_level(ort::GraphOptimizationLevel::Level2)
            .expect("ort")
            .with_model_from_file(vpath)
            .expect("ort");
        let mut done = false;
        let commit_every = 1000;
        let mut uncommitted = 0;
        while !done {
            let mut batch = vec![];
            let mut ids = vec![];
            loop {
                if let Ok((oid, prepped)) = prep_r.recv() {
                    batch.push(prepped);
                    ids.push(oid);
                    if batch.len() >= batch_size {
                        break;
                    }
                } else {
                    done = true;
                    break;
                }
            }
            if !batch.is_empty() {
                if use_tensorrt && batch.len() != batch_size {
                    // pad the final batch to avoid making TensorRT generate extra kernels
                    batch.resize(batch_size, batch[0].clone());
                }
                if let Ok(mut embeddings) = embed_images(&mut clip_visual_session, batch) {
                    pb_emb.inc(embeddings.len() as u64);
                    embeddings = embeddings.into_iter().map(|e| normalize(&e)).collect();
                    embeddings.truncate(ids.len()); // in case we padded the batch
                    let mut db = db.lock().unwrap();
                    let sidx = db.clip_embeddings.len();
                    db.clip_embeddings.append(&mut embeddings);
                    for (oid, eid) in ids.iter().zip(sidx..) {
                        db.by_id.get_mut(oid).unwrap().embedding_id = Some(eid);
                        uncommitted += 1;
                    }
                    if uncommitted >= commit_every {
                        db.persist(config).expect("commit");
                        uncommitted = 0;
                    }
                } else {
                    log::error!("failed to compute embedding batch, abort!");
                    done = true;
                }
            }
        }
    });
    db.lock().unwrap().persist(config)?;

    Ok(())
}
