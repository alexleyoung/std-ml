pub trait Loss {
    fn loss(&self, p: &[f64], y: &[f64]) -> f64;
    fn gradient(&self, p: &[f64], y: &[f64]) -> Vec<f64>;
}

pub struct MSE {}

impl Loss for MSE {
    // Calculate the mean squared error
    fn loss(&self, p: &[f64], y: &[f64]) -> f64 {
        let n = p.len() as f64;
        assert!(
            n as usize == y.len(),
            "Prediction size should match truth size"
        );

        let mut loss = 0.0;
        p.iter()
            .zip(y.iter())
            .for_each(|(&p, &y)| loss += (p - y).powf(2.0));
        loss / n
    }

    // calculate dL/dp
    fn gradient(&self, p: &[f64], y: &[f64]) -> Vec<f64> {
        let n = p.len() as f64;
        assert!(
            n as usize == y.len(),
            "Prediction size should match truth size"
        );

        p.iter()
            .zip(y.iter())
            .map(|(&p, &y)| 2.0 / n * (p - y))
            .collect()
    }
}

pub struct CrossEntropy {}

impl Loss for CrossEntropy {
    fn loss(&self, p: &[f64], y: &[f64]) -> f64 {
        assert!(
            p.len() == y.len(),
            "Prediction size should match truth size"
        );

        let max_val = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = p.iter().map(|&x| (x - max_val).exp()).sum();

        -(p.iter()
            .zip(y.iter())
            .map(|(&x, &y)| {
                let px = (x - max_val).exp() / exp_sum;
                y * px.ln()
            })
            .sum::<f64>())
    }

    fn gradient(&self, p: &[f64], y: &[f64]) -> Vec<f64> {
        assert!(
            p.len() == y.len(),
            "Prediction size should match truth size"
        );

        let max_val = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = p.iter().map(|&x| (x - max_val).exp()).sum();

        p.iter()
            .zip(y.iter())
            .map(|(&x, &y)| {
                let px = (x - max_val).exp() / exp_sum;
                px - y
            })
            .collect()
    }
}
