use super::activate::sigmoid_batch_f;
use super::activate::sigmoid_f;
use super::activate::softmax_batch_f;
use super::activate::softmax_f;
use ndarray::Array1;
use ndarray::Array2;
use std::collections::HashMap;

type Wb = (HashMap<String, Array1<f64>>, HashMap<String, Array2<f64>>);

#[allow(dead_code)]
pub fn init_net() -> Wb {
    let mut map_b: HashMap<String, Array1<f64>> = HashMap::new();
    let mut map_w: HashMap<String, Array2<f64>> = HashMap::new();
    map_w.insert("w1".to_string(), ndarray::arr2(&[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]));
    map_w.insert("w2".to_string(), ndarray::arr2(&[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]));
    map_w.insert("w3".to_string(), ndarray::arr2(&[[0.1, 0.3], [0.2, 0.4]]));
    map_b.insert("b1".to_string(), ndarray::arr1(&[0.1, 0.2, 0.3]));
    map_b.insert("b2".to_string(), ndarray::arr1(&[0.1, 0.2]));
    map_b.insert("b3".to_string(), ndarray::arr1(&[0.1, 0.2]));
    (map_b, map_w)
}

#[allow(dead_code)]
pub fn forward(net: Wb, x: Array1<f64>) -> Array1<f64> {
    let a1 = x.dot(&net.1["w1"]) + &net.0["b1"];
    let z1 = sigmoid_f(&a1);
    let a2 = z1.dot(&net.1["w2"]) + &net.0["b2"];
    let z2 = sigmoid_f(&a2);
    z2.dot(&net.1["w3"]) + &net.0["b3"]
    //softmax_f(&a3)
}

#[allow(dead_code)]
pub fn predict(net: Wb, x: Array1<f64>) -> Array1<f64> {
    let a1 = x.dot(&net.1["w1"]) + &net.0["b1"];
    let z1 = sigmoid_f(&a1);
    let a2 = z1.dot(&net.1["w2"]) + &net.0["b2"];
    let z2 = sigmoid_f(&a2);
    let a3 = z2.dot(&net.1["w3"]) + &net.0["b3"];
    softmax_f(&a3)
}

#[allow(dead_code)]
pub fn predict_batch(net: Wb, x: Array2<f64>) -> Array2<f64> {
    let a1 = x.dot(&net.1["w1"]) + &net.0["b1"];
    let z1 = sigmoid_batch_f(&a1);
    let a2 = z1.dot(&net.1["w2"]) + &net.0["b2"];
    let z2 = sigmoid_batch_f(&a2);
    let a3 = z2.dot(&net.1["w3"]) + &net.0["b3"];
    softmax_batch_f(&a3)
}
