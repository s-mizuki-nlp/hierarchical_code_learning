#!/bin/bash

DIR_JUPYTER_NOTEBOOK="/home/sakae/jupyter/notebook/sync_on_cloud/hierarchical_code_learning"
WORK_DIR="/home/sakae/hierarchical_code_learning/"
CHECKPOINT_DIR="${WORK_DIR}/saved_model_on_cloud/lightning_logs/"
SUFFIX="_saved_model"

python ./batch_evaluate_hyponymy_detection.py \
--evaluation_notebook="${DIR_JUPYTER_NOTEBOOK}/analyze_checkpoint_code_assignment_statistics.ipynb" \
--output_summary="${WORK_DIR}/experiment_results/summary_code_assignment_analysis_version_foo${SUFFIX}.jsonl" \
--output_notebook="${DIR_JUPYTER_NOTEBOOK}/evaluation_results/analyze_code_assignment_version_{version_no}${SUFFIX}.ipynb" \
--checkpoint_directory="${CHECKPOINT_DIR}" \
--evaluation_arguments='{"lexical_knowledge_name":"WordNet-hyponymy-noun-verb-ver2", "embedding_name":"fastText-wiki-news", "n_similarity_sample_size":10000}' \
--version_no="97,99,102"
# --version_no="97,99,102,104,106,133:152"
