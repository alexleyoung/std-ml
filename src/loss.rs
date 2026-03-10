pub trait Loss {
    fn loss(&self, yhat: &[f64], y: &[f64]) -> f64;
    fn gradient(&self, yhat: &[f64], y: &[f64]) -> Vec<f64>;
}

pub struct MSE {}

impl Loss for MSE {
    fn loss(&self, yhat: &[f64], y: &[f64]) -> f64 {
        todo!()
    }

    fn gradient(&self, yhat: &[f64], y: &[f64]) -> Vec<f64> {
        todo!()
    }
}

pub struct CrossEntropy {}

impl Loss for CrossEntropy {
    fn loss(&self, yhat: &[f64], y: &[f64]) -> f64 {
        todo!()
    }

    fn gradient(&self, yhat: &[f64], y: &[f64]) -> Vec<f64> {
        todo!()
    }
}
