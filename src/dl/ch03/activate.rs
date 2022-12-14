use ndarray::{Array1, Array2};

#[allow(dead_code)]
pub fn step_f(x: &Array1<f64>) -> Array1<f64> {
    x.iter().map(|&v| if v <= 0.0 { 0.0 } else { 1.0 }).collect()
}

#[allow(dead_code)]
pub fn sigmoid_f(x: &Array1<f64>) -> Array1<f64> {
    x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

#[allow(dead_code)]
pub fn sigmoid_batch_f(x: &Array2<f64>) -> Array2<f64> {
    x.map(|v| 1.0 / (1.0 + (-(*v)).exp()))
}

#[allow(dead_code)]
pub fn relu_f(x: &Array1<f64>) -> Array1<f64> {
    x.iter().map(|&v| f64::max(v, 0.0)).collect()
}

#[allow(dead_code)]
pub fn softmax_f(x: &Array1<f64>) -> Array1<f64> {
    let mut c = 0.0;
    for i in x {
        c = f64::max(c, *i);
    }
    let mut sum = 0.0;
    for i in x {
        sum += (i - c).exp();
    }
    x.iter().map(|&v| (v - c).exp() / sum).collect()
}

#[allow(dead_code)]
pub fn softmax_batch_f(x: &Array2<f64>) -> Array2<f64> {
    // muzusugi te kitanai code ni naattimatta... ze...
    let mut res = x.clone();
    for i in 0..x.nrows() {
        let b = x.row(i);
        let mut c = 0.0;
        for i in b {
            c = f64::max(c, *i);
        }
        let mut sum = 0.0;
        for i in b {
            sum += (i - c).exp();
        }
        for j in 0..x.ncols() {
            res[[i, j]] = (x[[i, j]] - c).exp() / sum;
        }
    }
    res
}
