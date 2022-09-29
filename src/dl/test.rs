#[cfg(test)]
pub mod tests {
    use ndarray::Array1;

    use crate::dl::ch02::logic::{and, and2};
    use crate::dl::ch03::activate::{relu_f, sigmoid_f, softmax_batch_f, softmax_f, step_f};
    use crate::dl::ch03::threenet::{forward, init_net};
    use crate::dl::ch04::diff::{gradient_descent, numerical_gradient};
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
    pub fn test_softmax_batch_f() {
        let x = ndarray::arr2(&[[1010.0, 1000.0, 990.0], [1010.0, 1000.0, 990.0], [1010.0, 1000.0, 990.0]]);
        let x2 = softmax_batch_f(&x);
        assert!((x2[[0, 0]] - 9.99954600e-01).abs() < 1e-6);
        assert!((x2[[0, 1]] - 4.53978686e-05).abs() < 1e-11);
        assert!((x2[[0, 2]] - 2.06106005e-09).abs() < 1e-15);
        assert!((x2[[1, 0]] - 9.99954600e-01).abs() < 1e-6);
        assert!((x2[[1, 1]] - 4.53978686e-05).abs() < 1e-11);
        assert!((x2[[1, 2]] - 2.06106005e-09).abs() < 1e-15);
        assert!((x2[[2, 0]] - 9.99954600e-01).abs() < 1e-6);
        assert!((x2[[2, 1]] - 4.53978686e-05).abs() < 1e-11);
        assert!((x2[[2, 2]] - 2.06106005e-09).abs() < 1e-15);
    }
    #[test]
    pub fn threenet_test() {
        let x = ndarray::arr1(&[1.0, 0.5]);
        let net = init_net();
        let y = forward(net, x);
        assert!((y[0] - 0.31682708).abs() < 1e-6);
        assert!((y[1] - 0.69627909).abs() < 1e-6);
    }
    pub fn function2(x: &Array1<f64>) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }
    #[test]
    pub fn numerical_gradient_test() {
        let test = numerical_gradient(function2, &ndarray::arr1(&[3.0, 4.0]));
        assert!((test[0] - 6.0).abs() < 1e-6);
        assert!((test[1] - 8.0).abs() < 1e-6);
    }
    #[test]
    pub fn gradient_descent_test() {
        let test = gradient_descent(function2, &ndarray::arr1(&[-3.0, 4.0]), 0.1, 100);
        assert!((test[0] - (-6.11110793e-10)).abs() < 1e-14);
        assert!((test[1] - (8.14814391e-10)).abs() < 1e-14);
    }
}
