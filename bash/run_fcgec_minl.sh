REF=mr  # ss: split sample, mlr: most levenshtein ratio, mr: multi-reference

for SEED in 1 2 3
do
    MODEL_DIR=saved_models/fcgec/minl/seed_$SEED
    python -u run.py train --amp --build --cache \
        --bin preprocessed_data/$REF \
        --ref $REF \
        --conf configs/fcgec.train.ini \
        --device 0,1 \
        --seed $SEED \
        --path $MODEL_DIR/model.pt \
        --train data/ns_leakage_processed/fcgec.train \
        --dev data/ns_original/fcgec.dev \
        --aggs='min'

    python -u run.py predict --binarize --cache \
        --bin preprocessed_data/$REF \
        --device 1 \
        --seed $SEED \
        --conf configs/fcgec.predict.ini \
        --path $MODEL_DIR/model.pt \
        --data data/ns_original/fcgec.dev.test \
        --pred $MODEL_DIR/fcgec.dev.pred \
        --scorer ChERRANT \
        --gold data/ns_original/fcgec.dev.m2

    wait

    python -u run.py predict --binarize --cache \
        --bin preprocessed_data/$REF \
        --device 1 \
        --seed $SEED \
        --conf configs/fcgec.predict.ini \
        --path $MODEL_DIR/model.pt \
        --data data/ns_original/fcgec.test \
        --pred $MODEL_DIR/fcgec.test.pred

    wait

    python utils/fcgec_submit.py \
        --input data/ns_original/FCGEC_test.json \
        --pred $MODEL_DIR/fcgec.test.pred \
        --out $MODEL_DIR/minl.seed_$SEED.zip \
        --keep_json

    python -u run.py predict --binarize --cache \
        --bin preprocessed_data/$REF \
        --device 1 \
        --seed $SEED \
        --conf configs/fcgec.predict.ini \
        --path $MODEL_DIR/model.pt \
        --data data/ns_leakage_processed/nasgec.exam \
        --pred $MODEL_DIR/nasgec.exam.pred \
        --scorer ChERRANT \
        --gold data/ns_leakage_processed/nasgec.exam.m2
done