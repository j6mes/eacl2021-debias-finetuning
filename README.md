# Elastic weight consolidation for better bias inoculation

Code for EACL2021 paper. 

### Download resources folder:
Download this Google Drive folder into your project as `resources/`
```
https://drive.google.com/drive/folders/15LxyLEDr9HSP5gmxBIlUU6bJKsFf5r_g?usp=sharing
```

### FEVER
Train original model

```
allennlp train -f -s work/fever/bert/ configs/fever/cls_bert_base.jsonnet --include-package debias_finetuning
```

Fine-tune with symmetric data

```
allennlp fine-tune-ewc -s work/fever/ft_bert -m work/fever/bert/model.tar.gz --folds 5 --ewc 100000 -c configs/fever/finetune_symmetric_bert_base.jsonnet --include-package debias_finetuning
```

### MultiNLI
Train original model

```
allennlp train -f -s work/multinli/bert configs/multinli/cls_esim.jsonnet --include-package debias_finetuning
```

Fine-tune with stress-test data

```
allennlp fine-tune-ewc -s work/multinli/ft_esim/ -m work/multinli/bert/model.tar.gz -c configs/multinli/finetune_stresstest_esim.jsonnet --include-package debias_finetuning --folds 5
```

The `-o` overrides can be used to pass in any of the stress test datasets.