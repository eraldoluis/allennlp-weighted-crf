import json
import shutil
import sys
import os

from allennlp.commands import main

# train, validation and test files
os.environ["NER_TRAIN_DATA_PATH"]="/data/eraldo/conll2003/eng.testa"
os.environ["NER_TEST_A_PATH"]="/data/eraldo/conll2003/eng.testa"
os.environ["NER_TEST_B_PATH"]="/data/eraldo/conll2003/eng.testb"

config_file = "training_config/ner.jsonnet"

overrides = json.dumps(
    {
        "trainer": {
            "cuda_device": 0,
            # "cuda_device": -1,
            "num_epochs": 15,
            "patience": 3,
        },
        # "data_loader": {
        #     "batch_size": 2
        # },
    })

serialization_dir = "log_train_ner"

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
