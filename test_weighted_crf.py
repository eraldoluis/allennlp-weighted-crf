import json
import shutil
import sys

from datetime import datetime
from dateutil import tz

from allennlp.commands import main

config_file = "./training_config/conll2003_simple_ner.json"

# Use overrides to train on CPU.
overrides = json.dumps(
    {
        # "train_data_path": "../allennlp-models/test_fixtures/tagging/conll2003.txt",
        # "validation_data_path": "../allennlp-models/test_fixtures/tagging/conll2003.txt",
        "train_data_path": "../data/conll2003_simple/eng_simple.testa",
        "validation_data_path": "../data/conll2003_simple/eng_simple.testa",
        # "trainer.cuda_device": -1, 
        # "trainer.validation_metric": "+f1-measure-overall", 
        # "model.verbose_metrics": True
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
