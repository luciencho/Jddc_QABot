# Jddc_QABot
QABot for JD Competition

### Data Generate
```
python3 data_gen.py -t $TMP_DIR -d $DATA_DIR
    # --tmp_dir temporary directory
    # --data_dir data directory
```

### Model Training
```
python3 trainer.py -t $TMP_DIR
    # --tmp_dir temporary directory
```

### Evaluation
```
python3 delta_bleu.py -a $ANSWER_PATH -i $INFER_PATH -r $RESULT_PATH
    # --answer_path path for answers50.txt
    # --infer_path path for inferences50.txt
    # --result_path path for results50.txt
```