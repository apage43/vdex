pub fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let ip = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f32>();
    let mag_a = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let mag_b = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    ip / (mag_a * mag_b)
}

pub fn normalize(inp: &[f32]) -> Vec<f32> {
    let norm = inp.iter().map(|v| v * v).sum::<f32>().sqrt();
    inp.iter().map(|v| v / norm).collect()
}
