pub trait Loss {
    fn loss(&self, p: &[f64], y: &[f64]) -> f64;
    fn gradient(&self, p: &[f64], y: &[f64]) -> Vec<f64>;
}

pub struct MSE {}

impl Loss for MSE {
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
        todo!()
    }

    fn gradient(&self, p: &[f64], y: &[f64]) -> Vec<f64> {
        todo!()
    }
}
