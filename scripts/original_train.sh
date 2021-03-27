task=$1
model=$2
train_data_path=$3
validation_data_path=$4
seed=${SEED:-1}
gpu=${GPU:-0}
output_dir=${OUTPUT_DIR:-work}

allennlp train \
     --include-package debias_finetuning \
     -f \
     -s ${output_dir}/${task}/method=original,model=${model}/seed-${seed} \
     configs/${task}/${model}.jsonnet \
        -o '{
           "train_data_path": '${train_data_path}'
           "validation_data_path": '${validation_data_path}'
           "trainer":{"cuda_device":'${GPU}'},
           "random_seed":'${seed}',
           "numpy_seed":'${seed}',
           "pytorch_seed":'${seed}',
          }'