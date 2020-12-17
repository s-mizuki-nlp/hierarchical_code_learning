#!/bin/bash

DIR_JUPYTER_NOTEBOOK="/home/sakae/jupyter/notebook/sync_on_cloud/hierarchical_code_learning"
WORK_DIR="/home/sakae/hierarchical_code_learning/"

python ./batch_evaluate_hyponymy_detection.py \
--evaluation_notebook="${DIR_JUPYTER_NOTEBOOK}/eval_checkpoint_hyponymy_detection_tasks.ipynb" \
--output_summary="${WORK_DIR}/experiment_results/summary_hyponymy_detection_tasks_version_28-72.txt" \
--output_notebook="${DIR_JUPYTER_NOTEBOOK}/evaluation_results/eval_version_{version_no}.ipynb" \
--evaluation_arguments='{"hyponymy_predictor_type":"entailment_probability", "cross_validation":True, "embedding_name":"fastText-wiki-news"}' \
--version_no="28:41,43:51,54:59,61:63,65:72"
