use color_eyre::{eyre::eyre, Result};
use std::{borrow::Cow, path::PathBuf, sync::Arc};

use image::imageops::FilterType;
use serde_derive::{Deserialize, Serialize};

use crate::Config;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CLIPModelInfo {
    pub image_dim: (u32, u32),
    pub image_mean: (f32, f32, f32),
    pub image_std: (f32, f32, f32),
    pub text_tokenizer_hub_name: Cow<'static, str>,
    pub text_tokenizer_bos: Option<Cow<'static, str>>,
    pub text_tokenizer_pad: Cow<'static, str>,
    pub text_tokenizer_eos: Cow<'static, str>,
    pub text_input_size: usize,
    pub arch_name: Cow<'static, str>,
    pub pretrain_name: Cow<'static, str>,
}

pub type DynF32Array = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>;

pub fn preprocess_image(img: image::DynamicImage, config: &Config) -> Result<DynF32Array> {
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

pub fn embed_images(
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
pub struct TextEmbedder {
    _ortenv: Arc<ort::Environment>,
    session: Option<ort::Session>,
    tokenizer: tokenizers::Tokenizer,
    input_ids: Vec<i64>,
    modelinfo: CLIPModelInfo,
}
impl TextEmbedder {
    pub fn new(config: &Config) -> Result<Self> {
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
    pub fn embed_text(&mut self, inp: &str) -> Result<Vec<f32>> {
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
