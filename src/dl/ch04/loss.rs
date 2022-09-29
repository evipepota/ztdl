use ndarray::{Array1, Array2};

#[allow(dead_code)]
fn sum_squared_error(y: &Array1<f64>, t: &Array1<f64>) -> f64 {
    0.5 * ((y - t) * (y - t)).sum()
}

#[allow(dead_code)]
fn cross_entoropy_error(y: &Array1<f64>, t: &Array1<f64>) -> f64 {
    let delta = 1e-7;
    let logy = y.map(|&v| (v + delta).log(1.0f64.exp()));
    -((t * logy).sum())
}

#[allow(dead_code)]
fn cross_entoropy_error_batch(y: &Array2<f64>, t: &Array2<f64>) -> f64 {
    let delta = 1e-7;
    let batch_size = y.nrows() as f64;
    let logy = y.map(|&v| (v + delta).log(1.0f64.exp()));
    -((t * logy).sum()) / batch_size
}
