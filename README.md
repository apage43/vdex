# CLIP indexer + server

```bash
# index
cargo run --release -- update-db -m /path/to/images
# run search server
cargo run --release -- serve
```

one endpoint, `/search_text` with query string parameters `skip`, `limit` (for paging, optional, default `limit` is 5), and `query`

`query` is an expression that computes an embedding that results will be returned in order of closeness to (by cosine similarity), the syntax allows expressions:
 * `"quoted strings"`, which are embedded with the model's text encoder
 * `@"<blake3 hash of image>"` which looks up the embedding for an image in the database
 * addition and subtraction (`"winter" - "snow"`, `@"ab12cd34" - "person"`)
 *  `mean(<expr>, ...)`
 *  `normalize(<expr>)`

[rudimentary search ui](https://ml.plausiblyreliable.com/vd-expr.mp4) at https://github.com/apage43/vdex-ui-electron

model files must be present in `clip_models`

## Supported models 
* ViT-L-16-SigLIP_webli
  * vision - https://ml.plausiblyreliable.com/clip_models/ViT-L-16-SigLIP-384_webli_visual.onnx
  * text - https://ml.plausiblyreliable.com/clip_models/ViT-L-16-SigLIP-384_webli_text.onnx
