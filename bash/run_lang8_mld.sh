REF=mlr  # ss: split sample, mlr: most levenshtein ratio, mr: multi-reference

for SEED in 1 2 3
do
    MODEL_DIR=saved_models/lang8/mld/seed_$SEED
    python -u run.py train --amp --build --cache \
        --bin preprocessed_data/$REF \
        --ref $REF \
        --conf configs/lang8.train.ini \
        --device 0,1 \
        --seed $SEED \
        --path $MODEL_DIR/model.pt \
        --train data/l2/lang8.train \
        --dev data/l2/mucgec.dev 

    python -u run.py predict --binarize --cache \
        --bin preprocessed_data/$REF \
        --device 1 \
        --seed $SEED \
        --conf configs/mucgec.predict.ini \
        --path $MODEL_DIR/model.pt \
        --data data/l2/mucgec.test \
        --pred $MODEL_DIR/mucgec.test.pred
done