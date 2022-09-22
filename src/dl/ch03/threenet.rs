use std::collections::HashMap;

use ndarray::Array1;
use ndarray::Array2;

use super::activate::sigmoid_f;

type Wb = (HashMap<String, Array1<f64>>, HashMap<String, Array2<f64>>);

#[allow(dead_code)]
pub fn init_net() -> Wb {
    let mut map_b: HashMap<String, Array1<f64>> = HashMap::new();
    let mut map_w: HashMap<String, Array2<f64>> = HashMap::new();
    map_w.insert(
        "w1".to_string(),
        ndarray::arr2(&[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
    );
    map_w.insert(
        "w2".to_string(),
        ndarray::arr2(&[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
    );
    map_w.insert("w3".to_string(), ndarray::arr2(&[[0.1, 0.3], [0.2, 0.4]]));
    map_b.insert("b1".to_string(), ndarray::arr1(&[0.1, 0.2, 0.3]));
    map_b.insert("b2".to_string(), ndarray::arr1(&[0.1, 0.2]));
    map_b.insert("b3".to_string(), ndarray::arr1(&[0.1, 0.2]));
    (map_b, map_w)
}

#[allow(dead_code)]
pub fn forward(net: Wb, x: Array1<f64>) -> Array1<f64> {
    let a1 = x.dot(net.1.get(&"w1".to_string()).unwrap()) + net.0.get(&"b1".to_string()).unwrap();
    let z1 = sigmoid_f(&a1);
    let a2 = z1.dot(net.1.get(&"w2".to_string()).unwrap()) + net.0.get(&"b2".to_string()).unwrap();
    let z2 = sigmoid_f(&a2);
    let z3 = z2.dot(net.1.get(&"w3".to_string()).unwrap()) + net.0.get(&"b3".to_string()).unwrap();
    z3
}
