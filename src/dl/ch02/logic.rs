use ndarray::Array1;

#[allow(dead_code)]
pub fn and(x1: f32, x2: f32) -> bool {
    let w1 = 0.5;
    let w2 = 0.5;
    let theta = 0.7;
    let tmp = x1 * w1 + x2 * w2;
    tmp > theta
}

#[allow(dead_code)]
pub fn and2(x1: f32, x2: f32) -> bool {
    let x: Array1<f32> = ndarray::arr1(&[x1, x2]);
    let w: Array1<f32> = ndarray::arr1(&[0.5, 0.5]);
    let b = -0.7;
    let tmp = (w * x).sum() + b;
    tmp > 0.0
}

#[cfg(test)]
pub mod tests {
    use crate::dl::ch02::logic::{and, and2};
    #[test]
    pub fn test_and() {
        assert!(!and(0.0, 0.0));
        assert!(!and(1.0, 0.0));
        assert!(!and(0.0, 1.0));
        assert!(and(1.0, 1.0));
    }
    #[test]
    pub fn test_and2() {
        assert!(!and2(0.0, 0.0));
        assert!(!and2(1.0, 0.0));
        assert!(!and2(0.0, 1.0));
        assert!(and2(1.0, 1.0));
    }
}
