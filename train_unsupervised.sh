#!/bin/sh

EMBEDDINGS_DIR="/home/sakae/Windows/public_model/embedding"
EMBEDDINGS_W2V="word2vec-google-news-300/word2vec-google-news-300"

python train_unsupervised.py \
--embeddings="${EMBEDDINGS_DIR}/${EMBEDDINGS_W2V}" \
--embeddings_type="word2vec" \
--batch_size=128 \
--epochs=10 \
--validation_split=0.001 \
--config_file="./config_files/template_unsupervised.py" \
--log_dir="./log/" \
--experiment_name="word2vec" \
--saved_model_dir="./saved_model/word2vec/" \
--verbose
# gpus="1,2"
