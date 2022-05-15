# jy_ner
各种版本的NER模型。
softmax, crf, biaffine, globalpointer, efficient globalpointer, ricon

# 依赖
```bash
pip install -r requirements.txt
```

# 训练
```bash
PRETRAINED_MODEL=hfl/chinese-bert-wwm-ext
MODEL_TYPE=bert
MODEL_CACHE_DIR=/mnt/f/hf/models
OUTPUT_DIR=./outputs
TRAIN_BATCH_SIZE=16
EPOCHS=5
SAVE_STEPS=400
ACCUM=1
METHOD=ricon
TASK=zh_cmeee
NGRAM=16
accelerate launch train.py --ngram $NGRAM --gradient_accumulation_steps $ACCUM --method $METHOD --output_dir $OUTPUT_DIR --per_device_train_batch_size $TRAIN_BATCH_SIZE --pretrained_model_name_or_path $PRETRAINED_MODEL --model_type $MODEL_TYPE --num_train_epochs $EPOCHS --save_steps $SAVE_STEPS --dataset_config_name $TASK --model_cache_dir $MODEL_CACHE_DIR
```
