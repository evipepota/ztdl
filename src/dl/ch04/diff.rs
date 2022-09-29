use ndarray::{Array, Array1};

#[allow(dead_code)]
pub fn numerical_diff(f: fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-4;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

#[allow(dead_code)]
pub fn numerical_gradient(f: fn(&Array1<f64>) -> f64, x: &Array1<f64>) -> Array1<f64> {
    let h = 1e-4;
    let mut grad = Array::<f64, _>::zeros(x.raw_dim());
    let mut x2 = x.clone();
    for i in 0..x2.len() {
        let tmp = x2[i];
        x2[i] = tmp + h;
        let f1 = f(&x2);
        x2[i] = tmp - h;
        let f2 = f(&x2);
        grad[i] = (f1 - f2) / (2.0 * h);
        x2[i] = tmp;
    }
    grad
}

#[allow(dead_code)]
pub fn gradient_descent(f: fn(&Array1<f64>) -> f64, init_x: &Array1<f64>, lr: f64, step_num: usize) -> Array1<f64> {
    let mut x = init_x.clone();
    for _i in 0..step_num {
        let grad = numerical_gradient(f, &x);
        x = x - lr * grad;
    }
    x
}
