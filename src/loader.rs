// IDX DataLoader
pub struct DataLoader {
    data: Vec<f64>,
    targets: Vec<f64>,
    batch_size: usize,
    batch_idx: usize,
}

impl DataLoader {
    pub fn new(raw_data: &str, target_data: &str) -> Self {
        todo!()
    }
}

impl Iterator for DataLoader {
    type Item = (Vec<f64>, Vec<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

fn parse_idx(path) -> Vec<f64> {
    todo!()
}
