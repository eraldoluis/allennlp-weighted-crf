import json
import shutil
import sys

from datetime import datetime
from dateutil import tz

from allennlp.commands import main

config_file = "./training_config/conll2003_simple_ner.jsonnet"

# Use overrides to train on CPU.
overrides = json.dumps({
    "train_data_path": "/data/eraldo/allennlp/data/conll2003_simple/eng_simple.train",
    "validation_data_path": "/data/eraldo/allennlp/data/conll2003_simple/eng_simple.testa",
    "test_data_path": "/data/eraldo/allennlp/data/conll2003_simple/eng_simple.testb",
    # "trainer.cuda_device": 0, 
    "trainer.num_epochs": 25,
    "model.label_weights": {"MISC": 2}, 
    # "model.weight_strategy": "emission_transition", 
    "model.weight_strategy": "lannoy", 
    "trainer.validation_metric": "+macro-fscore", 
    # "trainer.patience": 3, 
    "data_loader.batch_size": 1024, 
    # "model.verbose_metrics": True, 
    "random_seed": None, 
    "numpy_seed": None, 
    "pytorch_seed": None, 
})

datetime_fmt = '%Y_%m_%d-%H_%M_%S_%f_%z'
serialization_dir = f'train_out/{datetime.now(tz.tzlocal()).strftime(datetime_fmt)}'

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
# shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "allennlp_models.tagging.models",
    "-o", overrides,
]

main()
