dataset := "mnist"
lr := "0.01"
epochs := "5"
lambda := "0.0001"
batch_size := "32"

run:
    cargo run --release -- \
        --dataset {{dataset}} \
        --lr {{lr}} \
        --epochs {{epochs}} \
        --lambda {{lambda}} \
        --batch-size {{batch_size}}

build:
    cargo build --release

test:
    cargo test
