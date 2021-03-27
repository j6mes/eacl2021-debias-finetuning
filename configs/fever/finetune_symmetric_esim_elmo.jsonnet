{
  "dataset_reader": {
    "type": "debias_finetuning_classic",
    "token_indexers": {
      "elmo": {
        "type": "elmo_characters"
     }
    },
    "tokenizer": {
      "type":"spacy",
    },
    "frontend_reader": "symmetric"
  },
  "validation_dataset_reader": {
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
    "frontend_reader": "fever",
      "frontend_args": {
        "database": "resources/wikipedia/fever.db",
        "format_evidence":false
      }
  },
  "train_data_path": "resources/symmetric/fever_symmetric_dev.jsonl",
  "test_data_path": "resources/fever/negative_sampled_evidence/dev.ns.pages.p1.jsonl",
  "model": {
    "type": "esim",
    "dropout": 0.5,
    "text_field_embedder": {
        "token_embedders": {
        "elmo":{
            "type": "elmo_token_embedder",
            "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0.0
          }
        }
    },
    "encoder": {
        "type": "lstm",
        "input_size": 1024,
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