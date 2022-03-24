{
  "train_data_path": "../data/conll2003/eng.train",
  "validation_data_path": "../data/conll2003/eng.testa",
  "test_data_path": "../data/conll2003/eng.testb",
  "evaluate_on_test": true,
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "pos",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 1
      }
    }
  },
  "model": {
    "type": "crf_tagger",
    "verbose_metrics": true,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 50
        },
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
            "embedding_dim": 25,
            "vocab_namespace": "token_characters"
          },
          "encoder": {
            "type": "gru",
            "input_size": 25,
            "hidden_size": 80,
            "num_layers": 2,
            "dropout": 0.0,
            "bidirectional": true
          }
        }
      }
    },
    "encoder": {
      "type": "gru",
      "input_size": 210,
      "hidden_size": 300,
      "num_layers": 2,
      "dropout": 0.0,
      "bidirectional": true
    },
    "regularizer": {
      "regexes": [
        [
          "transitions$",
          {
            "type": "l2",
            "alpha": 0.01
          }
        ]
      ]
    }
  },
  "data_loader": {
    "batch_size": 32
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 5,
    // "cuda_device": -1, 
    "callbacks": [
      {
        "type": "wandb",
        // "project": "WeightedCRF",
        // "entity": "eraldoluis",
        // "watch_model": false,
        // "summary_interval": 1,
        // "should_log_parameter_statistics": false,
        // "should_log_learning_rate": false, 
        // "tags": ["weight_misc"]
      }
    ]
  }
}