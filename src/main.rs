use color_eyre::{eyre::eyre, Result};
use image::imageops::FilterType;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use jwalk::WalkDir;

use norman::special::NormEucl;
use ordered_float::NotNan;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use sha2::Digest;
use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokenizers::PaddingParams;

use serde_derive::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
struct Collection {
    roots: Vec<PathBuf>,
}

#[derive(Deserialize, Debug)]
struct Config {
    collections: HashMap<String, Collection>,
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
    fn persist(&self) -> Result<()> {
        let encoded = bincode::serialize(self)?;
        let mut outf = std::fs::File::create("db.temp.bc")?;
        let mut rdbslice = &encoded[..];
        std::io::copy(&mut rdbslice, &mut outf)?;
        std::fs::rename("db.temp.bc", "db.bc")?;
        Ok(())
    }
}
fn scan<P: AsRef<Path>>(db: Arc<Mutex<Database>>, root: P) -> Result<()> {
    let mut scanned = Vec::new();

    for entry in WalkDir::new(root) {
        let path = entry?.path();
        let loc = match path.extension().and_then(|s| s.to_str()) {
            Some("jpg") | Some("jpeg") | Some("png") => ObjectLocation::LocalPath(path),
            _ => continue,
        };
        scanned.push(loc);
    }
    let scanned: Vec<ObjectLocation> = {
        let db = db.lock().expect("lock");
        scanned.into_iter().filter(|i| !db.has(i)).collect()
    };
    eprintln!("Found {} new items.", scanned.len());

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

    db.lock().unwrap().persist()?;

    Ok(())
}

type DynF32Array = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>;

fn preprocess_image<P: AsRef<Path>>(path: P) -> Result<DynF32Array> {
    let img = image::io::Reader::open(&path)?
        .with_guessed_format()?
        .decode()?;
    let img = img.resize_exact(384, 384, FilterType::Gaussian);
    let ibuf = img.to_rgb32f();
    let fs = ibuf.as_flat_samples();
    let ndimg = ndarray::CowArray::from(ndarray::ArrayView::from_shape((384, 384, 3), fs.samples)?);
    let ndimg = ndimg.permuted_axes([2, 1, 0]);
    let ndimg = ndimg.map(|v| (v - 0.5) / 0.5);
    //eprintln!("pp: {} => {ndimg:?}", path.as_ref().display());
    Ok(ndimg.into_dyn().into_owned())
}

fn embed_images(
    clip_session: &mut ort::Session,
    images: Vec<DynF32Array>,
) -> Result<Vec<Vec<f32>>> {
    let views: Vec<_> = images.iter().map(|i| i.view()).collect();
    //let batch = ndarray::concatenate(ndarray::Axis(0), &views[..])?;
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
    let config_raw = std::fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&config_raw)?;
    let get_db = || {
        let db: Database = match std::fs::read("db.bc") {
            Ok(fbsrc) => bincode::deserialize(&fbsrc)?,
            _ => Database::default(),
        };
        let db = Arc::new(Mutex::new(db));
        Ok::<_, color_eyre::Report>(db)
    };
    let args = Args::parse();
    match args.command {
        None => {
            Args::command().print_help()?;
        }
        Some(Commands::Serve) => {
            //serve_search(get_db()?, &config)?;
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
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
    input_ids: Vec<i64>,
}
impl TextEmbedder {
    fn new(config: &Config, ortenv: Arc<ort::Environment>) -> Result<Self> {
        let session = ort::SessionBuilder::new(&ortenv)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level1)?
            .with_model_from_file("text.onnx")?;
        let tokenizer = tokenizers::Tokenizer::from_pretrained("timm/ViT-L-16-SigLIP-384", None)
            .map_err(|e| eyre!("{e:?}"))?;
        Ok(TextEmbedder {
            session,
            tokenizer,
            input_ids: vec![0; 64],
        })
    }
    fn embed_and_norm(&mut self, inp: &str) -> Result<Vec<f32>> {
        let encd = self
            .tokenizer
            .with_padding(Some(PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(64),
                pad_id: 1,
                ..Default::default()
            }))
            .encode(inp, false)
            .map_err(|e| eyre!("{e:?}"))?;
        //eprintln!("Encoded query: {encd:?}");
        self.input_ids.clear();
        self.input_ids
            .extend(encd.get_ids().iter().map(|v| *v as i64));
        //eprintln!("ids: {ids:?}");
        let ids = ndarray::CowArray::from(&self.input_ids)
            .into_shape((1, 64))?
            .into_dyn();
        let result = self.session.run(vec![ort::Value::from_array(
            self.session.allocator(),
            &ids,
        )?])?;
        let emb: ort::tensor::OrtOwnedTensor<f32, _> = result[0].try_extract()?;
        drop(result);
        let embv: Vec<f32> = emb.view().iter().copied().collect();
        let embv = normalize(&embv);
        Ok(embv)
    }
}

fn search_db(db: Arc<Mutex<Database>>, config: &Config, query: String) -> Result<()> {
    let gpu_id = config.gpu_id.unwrap_or(0);

    let ortenv = ort::Environment::builder()
        .with_name("clip-text")
        .with_execution_providers([
            ort::ExecutionProvider::CUDA(ort::execution_providers::CUDAExecutionProviderOptions {
                device_id: gpu_id,
                ..Default::default()
            }),
            ort::ExecutionProvider::CPU(Default::default()),
        ])
        .build()?
        .into_arc();

    let mut te = TextEmbedder::new(config, ortenv.clone())?;

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
    drop(te);
    Ok(())
}

// struct AppState {
//     te: Arc<Mutex<TextEmbedder>>,
// }

// async fn handle_search(axum::extract::State(state): axum::extract::State<Arc<AppState>>) {}
// async fn serve(app: Router, port: u16) {
//     let addr = SocketAddr::from(([127, 0, 0, 1], port));
//     axum::Server::bind(&addr)
//         .serve(app.into_make_service())
//         .await
//         .unwrap();
// }
// fn serve_search(db: Arc<Mutex<Database>>, config: &Config) -> Result<()> {
//     let rt = tokio::runtime::Runtime::new()?;
//     let state = Arc::new(AppState {
//         te: Arc::new(Mutex::new(TextEmbedder::new(config)?)),
//     });
//     let app = axum::Router::new()
//         .route("/search_text", axum::routing::get(handle_search))
//         .with_state(state);

//     rt.block_on(
//         tokio::spawn(async move {
//             serve(app, 6680).await;
//         }))?;
//     Ok(())
// }

fn update_db(db: Arc<Mutex<Database>>, config: &Config) -> Result<()> {
    eprintln!("Db has {} items.", db.lock().unwrap().by_id.len());
    eprintln!("Scanning...");
    for (_cname, coll) in config.collections.iter() {
        for root in coll.roots.iter() {
            scan(db.clone(), root)?;
        }
    }
    let gpu_id = config.gpu_id.unwrap_or(0);
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
        .with_model_from_file("visual.onnx")?;

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
        let preproc_thread = std::thread::spawn(move || {
            path_and_ids
                .par_iter()
                .progress_with_style(
                    ProgressStyle::default_bar()
                        .template("{wide_bar} {per_sec} {eta} {elapsed}")
                        .unwrap(),
                )
                .for_each(|(id, path)| {
                    let r = preprocess_image(path);
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
        let bs = 32;
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
                //println!("eshape b: {} i {}", embeddings.len(), embeddings[0].len());
                //let mut embeddings = embeddings.iter().map(normalize).collect();
                //eprintln!("emb: {:?}", embeddings);
                let mut db = db.lock().unwrap();
                let sidx = db.clip_embeddings.len();
                db.clip_embeddings.append(&mut embeddings);
                for (oid, eid) in ids.iter().zip(sidx..) {
                    db.by_id.get_mut(oid).unwrap().embedding_id = Some(eid);
                }
            }
        }
        preproc_thread.join().map_err(|e| eyre!("{e:?}"))?;
        db.lock().unwrap().persist()?;
    }

    Ok(())
}