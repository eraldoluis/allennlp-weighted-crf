import json
import shutil
import sys

from allennlp.commands import main

config_file = "training_config/ner_new.json"

# Use overrides to train on CPU.
overrides = json.dumps(
    {
        "train_data_path": "~/lia/src/allennlpdev/allennlp-models/test_fixtures/tagging/conll2003.txt",
        "validation_data_path": "~/lia/src/allennlpdev/allennlp-models/test_fixtures/tagging/conll2003.txt",
        "trainer": {
            # "cuda_device": -1, 
            "validation_metric": "+f1-measure-overall"
        },
        "model": {"verbose_metrics": True}
    })

serialization_dir = "train_out"

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
