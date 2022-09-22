#[cfg(test)]
pub mod tests {
    use crate::dl::ch02::logic::{and, and2};
    use crate::dl::ch03::activate::{relu_f, sigmoid_f, softmax_f, step_f};
    use crate::dl::ch03::threenet::{forward, init_net};
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
    #[test]
    pub fn test_step_f() {
        let x = ndarray::arr1(&[-1.0, 1.0, 2.0]);
        let x2 = step_f(&x);
        assert_eq!(x2, ndarray::array![0.0, 1.0, 1.0]);
    }
    #[test]
    pub fn test_sigmoid_f() {
        let x = ndarray::arr1(&[-1.0, 1.0, 2.0]);
        let x2 = sigmoid_f(&x);
        assert!((x2[0] - 0.26894142).abs() < 1e-6);
        assert!((x2[1] - 0.73105858).abs() < 1e-6);
        assert!((x2[2] - 0.88079708).abs() < 1e-6);
    }
    #[test]
    pub fn test_relu_f() {
        let x = ndarray::arr1(&[-1.0, 1.0, 2.0]);
        let x2 = relu_f(&x);
        assert_eq!(x2, ndarray::array![0.0, 1.0, 2.0]);
    }
    #[test]
    pub fn test_softmax_f() {
        let x = ndarray::arr1(&[1010.0, 1000.0, 990.0]);
        let x2 = softmax_f(&x);
        assert!((x2[0] - 9.99954600e-01).abs() < 1e-6);
        assert!((x2[1] - 4.53978686e-05).abs() < 1e-11);
        assert!((x2[2] - 2.06106005e-09).abs() < 1e-15);
    }
    #[test]
    pub fn threenet_test() {
        let x = ndarray::arr1(&[1.0, 0.5]);
        let net = init_net();
        let y = forward(net, x);
        assert!((y[0] - 0.31682708).abs() < 1e-6);
        assert!((y[1] - 0.69627909).abs() < 1e-6);
    }
}
