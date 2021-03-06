local transformer_model = "roberta-base";
local transformer_dim = 768;
local cls_is_last_token = false;

{
  "dataset_reader": {
    "type": "debias_finetuning_transformer",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 256
      }
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model
    },
    "frontend_reader": "symmetric",
  },
  "validation_dataset_reader": {
    "type": "debias_finetuning_transformer",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 256
      }
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model
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
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 256
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
       "cls_is_last_token": cls_is_last_token
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 1,
      "hidden_dims": transformer_dim,
      "activations": "tanh"
    },
    "dropout": 0.1
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 8
    }
  },
  "trainer": {
    "num_epochs": 4,
    "cuda_device" : 0,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    }
  }
}