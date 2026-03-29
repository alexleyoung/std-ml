use std::fs;

pub struct IDXDataLoader {
    data: Vec<f64>,
    data_size: usize,
    targets: Vec<f64>,
    target_size: usize,
    batch_size: usize,
}

impl IDXDataLoader {
    pub fn new(raw_data: &str, target_data: &str, batch_size: usize) -> Self {
        let (data, data_dim_sizes) = parse_idx(raw_data);
        let (targets, target_dim_sizes) = parse_idx(target_data);
        IDXDataLoader {
            data: data,
            data_size: data_dim_sizes.iter().fold(1, |a, x| a * x),
            targets,
            target_size: target_dim_sizes.iter().fold(1, |a, x| a * x),
            batch_size,
        }
    }
}

impl IDXDataLoader {
    pub fn iter(&self) -> DataLoaderIter<'_> {
        DataLoaderIter {
            data: &self.data,
            data_size: self.data_size,
            targets: &self.targets,
            target_size: self.target_size,
            batch_size: self.batch_size,
            current_data: 0,
            current_target: 0,
        }
    }
}

pub struct DataLoaderIter<'a> {
    data: &'a [f64],
    data_size: usize,
    targets: &'a [f64],
    target_size: usize,
    batch_size: usize,
    current_data: usize,
    current_target: usize,
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = (&'a [f64], &'a [f64]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_data >= self.data.len() || self.current_target >= self.targets.len() {
            return None;
        }

        let data_end = (self.current_data + self.data_size * self.batch_size).min(self.data.len());
        let target_end =
            (self.current_target + self.target_size * self.batch_size).min(self.targets.len());

        self.current_data = data_end;
        self.current_target = target_end;
        Some((
            &self.data[self.current_data..data_end],
            &self.targets[self.current_target..target_end],
        ))
    }
}

fn parse_idx(path: &str) -> (Vec<f64>, Vec<usize>) {
    let bytes = fs::read(path).unwrap();
    // parse magic number (first 4 bytes)
    // byte 1,2 are zero
    // byte 3 is type
    // byte 4 is # dimensions
    let data_type = match bytes[2] {
        0x08 => IDXDataType::UnsignedByte,
        0x09 => IDXDataType::SignedByte,
        0x0B => IDXDataType::Short,
        0x0C => IDXDataType::Int,
        0x0D => IDXDataType::Float,
        0x0E => IDXDataType::Double,
        _ => panic!("Invalid IDX data type"),
    };
    let data_size = match data_type {
        IDXDataType::UnsignedByte | IDXDataType::SignedByte => 1,
        IDXDataType::Short => 2,
        IDXDataType::Int | IDXDataType::Float => 4,
        IDXDataType::Double => 8,
    };
    let num_dims = bytes[3] as usize;

    // parse dimension sizes
    let mut offset: usize = 4;
    let mut dim_sizes = vec![0 as usize; num_dims];
    for i in 0..num_dims {
        dim_sizes[i] = u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;
    }

    let mut data: Vec<f64> = Vec::with_capacity(bytes.len() / data_size);
    while offset < bytes.len() {
        let val: f64 = match data_type {
            IDXDataType::UnsignedByte => u8::from_be_bytes([bytes[offset]]) as f64,
            IDXDataType::SignedByte => i8::from_be_bytes([bytes[offset]]) as f64,
            IDXDataType::Short => i16::from_be_bytes([bytes[offset], bytes[offset + 1]]) as f64,
            IDXDataType::Int => i32::from_be_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as f64,
            IDXDataType::Float => f32::from_be_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as f64,
            IDXDataType::Double => f64::from_be_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]),
        };
        data.push(val);
        offset += data_size;
    }

    (data, dim_sizes)
}

#[derive(Clone, Copy)]
enum IDXDataType {
    UnsignedByte,
    SignedByte,
    Short,
    Int,
    Float,
    Double,
}
