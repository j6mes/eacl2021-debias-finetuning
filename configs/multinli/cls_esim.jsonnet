{
  "dataset_reader": {
    "type": "debias_finetuning_classic",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "tokenizer": {
      "type":"spacy",
    },
    "frontend_reader": "snli",
  },
  "train_data_path": "resources/multinli_1.0/multinli_1.0_train.jsonl",
  "validation_data_path": "resources/multinli_1.0/multinli_1.0_dev_matched.jsonl",
  "model": {
    "type": "esim",
    "dropout": 0.5,
    "text_field_embedder": {
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
                "embedding_dim": 300
            }
        }
    },
    "encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 300,
        "num_layers": 1,
        "bidirectional": true
    },
    "matrix_attention": {
        "type": "dot_product"
    },
    "projection_feedforward": {
        "input_dim": 2400,
        "hidden_dims": 300,
        "num_layers": 1,
        "activations": "relu"
    },
    "inference_encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 300,
        "num_layers": 1,
        "bidirectional": true
    },
    "output_feedforward": {
        "input_dim": 2400,
        "num_layers": 1,
        "hidden_dims": 300,
        "activations": "relu",
        "dropout": 0.5
    },
    "output_logit": {
        "input_dim": 300,
        "num_layers": 1,
        "hidden_dims": 3,
        "activations": "linear"
    },
    "initializer": {"regexes":
        [
        [".*linear_layers.*weight", {"type": "xavier_uniform"}],
        [".*linear_layers.*bias", {"type": "zero"}],
        [".*weight_ih.*", {"type": "xavier_uniform"}],
        [".*weight_hh.*", {"type": "orthogonal"}],
        [".*bias_ih.*", {"type": "zero"}],
        [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
    ]
    }
   },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 64,
            "sorting_keys": ["hypothesis","premise"],
        }
    },

  "trainer": {
    "num_epochs": 75,
    "checkpointer": {
        "num_serialized_models_to_keep": 2,
    },
    "patience":10,
    "cuda_device": 0,
    "grad_norm": 10.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 0.0004
    },
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 0
    }

  }
}