use color_eyre::Result;

use indicatif::ProgressDrawTarget;
use jwalk::WalkDir;

use crate::mmindex::{
    database::{compute_id_file, Database, Object, ObjectLocation},
    embedding::{embed_images, preprocess_image},
};

use std::{
    io::Seek,
    path::PathBuf,
    sync::{Arc, Mutex},
};
pub fn update_db(
    db: Arc<Mutex<Database>>,
    config: &crate::Config,
    scan_paths: &[PathBuf],
    batch_size: usize,
    use_tensorrt: bool,
    readers: usize,
) -> Result<()> {
    log::info!("Db has {} items.", db.lock().unwrap().by_id.len());
    let pb = indicatif::ProgressBar::new_spinner().with_style(
        indicatif::ProgressStyle::default_spinner()
            .template("{spinner} Scanning... {pos} files")
            .unwrap(),
    );
    let mut scanned = Vec::new();
    let gpu_id = config.gpu_id;
    let minfo = &config.model_info;
    let vpath = PathBuf::from("clip_models").join(format!(
        "{}_{}_visual.onnx",
        minfo.arch_name, minfo.pretrain_name
    ));
    for root in scan_paths.iter() {
        let root = std::fs::canonicalize(root)?;
        for entry in WalkDir::new(root) {
            let path = entry?.path();
            let loc = match path.extension().and_then(|s| s.to_str()) {
                Some("jpg") | Some("jpeg") | Some("png") => ObjectLocation::LocalPath(path),
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
            .filter(|i| !db.loc_has_embedding(i))
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
                                {
                                    let mut db = db.lock().unwrap();
                                    // file w/ this hash has been seen before
                                    if db.by_id.contains_key(&oid) {
                                        let obj = db.by_id.get_mut(&oid).unwrap();
                                        // add the new location to the front
                                        obj.locations.insert(0, loc.clone());
                                        // remove any locations that don't exist
                                        obj.locations
                                            .retain(|ObjectLocation::LocalPath(p)| p.exists());
                                        if !db.oid_has_embedding(&oid) {
                                            drop(db);
                                            // embed only if it hasn't been
                                            fpi_s.send((loc.clone(), oid, fp)).unwrap();
                                        }
                                    } else {
                                        db.insert(
                                            loc.clone(),
                                            oid,
                                            Object {
                                                locations: vec![loc.clone()],
                                                ..Default::default()
                                            },
                                        );
                                        drop(db);
                                        fpi_s.send((loc, oid, fp)).unwrap();
                                    }
                                }
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
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)
            .expect("ort")
            .with_memory_pattern(true)
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
                if batch.len() != batch_size {
                    // enable constant batch size optimization
                    batch.resize(batch_size, batch[0].clone());
                }
                if let Ok(mut embeddings) = embed_images(&mut clip_visual_session, batch) {
                    embeddings.truncate(ids.len()); // in case we padded the batch
                    let mut db = db.lock().unwrap();
                    let sidx = db.clip_embeddings.len();
                    db.clip_embeddings.append(&mut embeddings);
                    for (oid, eid) in ids.iter().zip(sidx..) {
                        db.by_id.get_mut(oid).unwrap().embedding_id = Some(eid);
                        uncommitted += 1;
                    }
                    pb_emb.inc(ids.len() as u64);
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
