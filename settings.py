import os

DATASET_DIR     = os.path.join("dataset")
DESCRIPTORS_DIR = os.path.join("descriptors")

if not os.path.exists(DESCRIPTORS_DIR):
    os.mkdir(DESCRIPTORS_DIR)

DEBUG           = True
ELITE_NUMBER    = 12