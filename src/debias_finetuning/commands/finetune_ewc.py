"""
The ``fine-tune`` subcommand is used to continue training (or `fine-tune`) a model on a `different
dataset` than the one it was originally trained on.  It requires a saved model archive file, a path
to the data you will continue training with, and a directory in which to write the results.

.. code-block:: bash

   $ allennlp fine-tune --help
    usage: allennlp fine-tune [-h] -m MODEL_ARCHIVE -c CONFIG_FILE -s
                              SERIALIZATION_DIR [-o OVERRIDES] [--extend-vocab]
                              [--file-friendly-logging]
                              [--batch-weight-key BATCH_WEIGHT_KEY]
                              [--embedding-sources-mapping EMBEDDING_SOURCES_MAPPING]
                              [--include-package INCLUDE_PACKAGE]

    Continues training a saved model on a new dataset.

    optional arguments:
      -h, --help            show this help message and exit
      -m MODEL_ARCHIVE, --model-archive MODEL_ARCHIVE
                            path to the saved model archive from training on the
                            original data
      -c CONFIG_FILE, --config-file CONFIG_FILE
                            configuration file to use for training. Format is the
                            same as for the "train" command, but the "model"
                            section is ignored.
      -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the fine-tuned model and
                            its logs
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the training
                            configuration (only affects the config_file, _not_ the
                            model_archive)
      --extend-vocab        if specified, we will use the instances in your new
                            dataset to extend your vocabulary. If pretrained-file
                            was used to initialize embedding layers, you may also
                            need to pass --embedding-sources-mapping.
      --file-friendly-logging
                            outputs tqdm status on separate lines and slows tqdm
                            refresh rate
      --batch-weight-key BATCH_WEIGHT_KEY
                            If non-empty, name of metric used to weight the loss
                            on a per-batch basis.
      --embedding-sources-mapping EMBEDDING_SOURCES_MAPPING
                            a JSON dict defining mapping from embedding module
                            path to embeddingpretrained-file used during training.
                            If not passed, and embedding needs to be extended, we
                            will try to use the original file paths used during
                            training. If they are not available we will use random
                            vectors for embedding extension.
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
import argparse
import json
import logging
import os
import random
from copy import deepcopy, copy
import re
from glob import glob
from typing import Dict, Any, List, Iterator


import torch
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from sklearn.model_selection import KFold
from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params
from allennlp.common.util import prepare_environment, prepare_global_logging, \
                                 get_frozen_and_tunable_parameter_names
from allennlp.nn.util import get_device_of
from allennlp.data.dataloader import DataLoader, TensorDict
from allennlp.models import load_archive, archive_model
from allennlp.models.archival import CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer, EpochCallback, GradientDescentTrainer, BatchCallback
from allennlp.training.util import datasets_from_params, evaluate
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import move_to_device
from torch import nn
from tqdm import tqdm

from debias_finetuning.losses.ewc import EWC
from debias_finetuning.losses.util import CallLossCallback

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Subcommand.register("fine-tune-ewc")
class FineTune(Subcommand):
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = """Continues training a saved model on a new dataset."""
        subparser = parser.add_parser(self.name,
                                      description=description,
                                      help='Continue training a model on a new dataset.')

        subparser.add_argument('-m', '--model-archive',
                               required=True,
                               type=str,
                               help='path to the saved model archive from training on the original data')

        subparser.add_argument('-c', '--config-file',
                               required=True,
                               type=str,
                               help='configuration file to use for training. Format is the same as '
                               'for the "train" command, but the "model" section is ignored.')

        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='directory in which to save the fine-tuned model and its logs')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the training configuration '
                               '(only affects the config_file, _not_ the model_archive)')

        subparser.add_argument('--extend-vocab',
                               action='store_true',
                               default=False,
                               help='if specified, we will use the instances in your new dataset to '
                                    'extend your vocabulary. If pretrained-file was used to initialize '
                                    'embedding layers, you may also need to pass --embedding-sources-mapping.')
        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        subparser.add_argument('--batch-weight-key',
                               type=str,
                               default="",
                               help='If non-empty, name of metric used to weight the loss on a per-batch basis.')

        subparser.add_argument('--embedding-sources-mapping',
                               type=str,
                               default="",
                               help='a JSON dict defining mapping from embedding module path to embedding'
                               'pretrained-file used during training. If not passed, and embedding needs to be '
                               'extended, we will try to use the original file paths used during training. If '
                               'they are not available we will use random vectors for embedding extension.')

        subparser.add_argument("--fold",
                               type=int,
                               default=None)

        subparser.add_argument("--folds",
                               type=int,
                               default=None)
        subparser.add_argument("--ewc",
                               type=float,
                               default=None)


        subparser.set_defaults(func=fine_tune_model_from_args)

        return subparser


def fine_tune_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    fine_tune_model_from_file_paths(model_archive_path=args.model_archive,
                                    config_file=args.config_file,
                                    serialization_dir=args.serialization_dir,
                                    overrides=args.overrides,
                                    extend_vocab=args.extend_vocab,
                                    file_friendly_logging=args.file_friendly_logging,
                                    batch_weight_key=args.batch_weight_key,
                                    embedding_sources_mapping=args.embedding_sources_mapping,
                                    in_fold=args.fold,
                                    folds=args.folds,
                                    ewc=args.ewc)


def fine_tune_model_from_file_paths(model_archive_path: str,
                                    config_file: str,
                                    serialization_dir: str,
                                    overrides: str = "",
                                    extend_vocab: bool = False,
                                    file_friendly_logging: bool = False,
                                    batch_weight_key: str = "",
                                    embedding_sources_mapping: str = "",
                                    in_fold=None,
                                    folds=None, ewc:float=None) -> Model:
    """
    A wrapper around :func:`fine_tune_model` which loads the model archive from a file.

    Parameters
    ----------
    model_archive_path : ``str``
        Path to a saved model archive that is the result of running the ``train`` command.
    config_file : ``str``
        A configuration file specifying how to continue training.  The format is identical to the
        configuration file for the ``train`` command, but any contents in the ``model`` section is
        ignored (as we are using the provided model archive instead).
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`fine_tune_model`.
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    extend_vocab: ``bool``, optional (default=False)
        If ``True``, we use the new instances to extend your vocabulary.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`fine_tune_model`.
    batch_weight_key : ``str``, optional (default="")
        If non-empty, name of metric used to weight the loss on a per-batch basis.
    embedding_sources_mapping: ``str``, optional (default="")
        JSON string to define dict mapping from embedding paths used during training to
        the corresponding embedding filepaths available during fine-tuning.
    """
    # We don't need to pass in `cuda_device` here, because the trainer will call `model.cuda()` if
    # necessary.
    archive = load_archive(model_archive_path)
    params = Params.from_file(config_file, overrides)

    embedding_sources: Dict[str, str] = json.loads(embedding_sources_mapping) if embedding_sources_mapping else {}
    return fine_tune_model(model=archive.model,
                           params=params,
                           serialization_dir=serialization_dir,
                           extend_vocab=extend_vocab,
                           file_friendly_logging=file_friendly_logging,
                           batch_weight_key=batch_weight_key,
                           embedding_sources_mapping=embedding_sources,
                           in_fold=in_fold,
                           num_folds=folds,
                           ewc_weight=ewc)


class EvalEpochCallback(EpochCallback):
    def __init__(self, fold:int, fold_data_loader, test_data_loader, global_metrics: Dict[str,Any]):
        self._fold = fold
        self._fold_data_loader = fold_data_loader
        self._test_data_loader = test_data_loader
        self._global_metrics = global_metrics

        self._global_metrics["fold-{}".format(fold)] = {}


    def __call__(self, trainer: GradientDescentTrainer, metrics: Dict[str, Any], epoch: int) -> None:
        if epoch<0:
            return
        e_metrics = {}

        test_metrics = evaluate(model=trainer.model,
                                data_loader=self._test_data_loader,
                                cuda_device=trainer.cuda_device,
                                batch_weight_key="")

        for key, value in test_metrics.items():
            e_metrics["test_" + key] = value

        test_metrics = evaluate(model=trainer.model,
                                data_loader=self._fold_data_loader,
                                cuda_device=trainer.cuda_device,
                                batch_weight_key="")
        for key, value in test_metrics.items():
            e_metrics["fold_" + key] = value

        self._global_metrics["fold-{}".format(self._fold)]["epoch-{}".format(epoch)] = e_metrics



def fine_tune_model(model: Model,
                    params: Params,
                    serialization_dir: str,
                    extend_vocab: bool = False,
                    file_friendly_logging: bool = False,
                    batch_weight_key: str = "",
                    embedding_sources_mapping: Dict[str, str] = None,
                    in_fold = None,
                    num_folds = None,
                    ewc_weight=None) -> Model:
    """
    Fine tunes the given model, using a set of parameters that is largely identical to those used
    for :func:`~allennlp.commands.train.train_model`, except that the ``model`` section is ignored,
    if it is present (as we are already given a ``Model`` here).

    The main difference between the logic done here and the logic done in ``train_model`` is that
    here we do not worry about vocabulary construction or creating the model object.  Everything
    else is the same.

    Parameters
    ----------
    model : ``Model``
        A model to fine tune.
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment
    serialization_dir : ``str``
        The directory in which to save results and logs.
    extend_vocab: ``bool``, optional (default=False)
        If ``True``, we use the new instances to extend your vocabulary.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    batch_weight_key : ``str``, optional (default="")
        If non-empty, name of metric used to weight the loss on a per-batch basis.
    embedding_sources_mapping: ``Dict[str, str]``, optional (default=None)
        mapping from model paths to the pretrained embedding filepaths
        used during fine-tuning.
    """
    prepare_environment(params)
    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        raise ConfigurationError(f"Serialization directory ({serialization_dir}) "
                                 f"already exists and is not empty.")

    os.makedirs(serialization_dir, exist_ok=True)
    prepare_global_logging(serialization_dir, file_friendly_logging)

    serialization_params = deepcopy(params).as_dict(quiet=True)
    with open(os.path.join(serialization_dir, CONFIG_NAME), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    if params.pop('model', None):
        logger.warning("You passed parameters for the model in your configuration file, but we "
                       "are ignoring them, using instead the model parameters in the archive.")

    vocabulary_params = params.pop('vocabulary', {})
    if vocabulary_params.get('directory_path', None):
        logger.warning("You passed `directory_path` in parameters for the vocabulary in "
                       "your configuration file, but it will be ignored. ")

    all_datasets = datasets_from_params(params)
    vocab = model.vocab

    if extend_vocab:
        datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

        for dataset in datasets_for_vocab_creation:
            if dataset not in all_datasets:
                raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

        logger.info("Extending model vocabulary using %s data.", ", ".join(datasets_for_vocab_creation))
        vocab.extend_from_instances(vocabulary_params,
                                    (instance for key, dataset in all_datasets.items()
                                     for instance in dataset
                                     if key in datasets_for_vocab_creation))

        model.extend_embedder_vocab(embedding_sources_mapping)

    trainer_params = params.pop("trainer")
    no_grad_regexes = trainer_params.pop("no_grad", ())
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names = \
                   get_frozen_and_tunable_parameter_names(model)
    logger.info("Following parameters are Frozen  (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)
    logger.info("Following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)

    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    train_data = all_datasets['train']
    validation_data = all_datasets.get('validation')
    test_data = all_datasets.get('test')

    dl_params = params.pop("data_loader")
    if test_data is not None:
        rand = random.Random(1234)
        test_data.index_with(vocab)
        shuffled_test = copy(test_data.instances)
        rand.shuffle(shuffled_test)
        extra_test = shuffled_test[:2000]

        keys = deepcopy(dl_params.as_dict())
        keys.update({"dataset": AllennlpDataset(extra_test, vocab)})
        extra_test_loader = DataLoader.from_params(params.pop("test_data_loader", keys))

        keys = deepcopy(dl_params.as_dict())
        keys.update({"dataset": test_data})
        test_loader = DataLoader.from_params(params.pop("test_data_loader", keys))

    master_model = model
    global_metrics = {}
    training_metrics = []
    final_metrics = {}
    master_trainer = trainer_params.as_dict()

    if num_folds is not None:

        rand = random.Random(1234)

        fold_train = []
        fold_test = []

        fold_train_loader = []
        fold_test_loader = []

        shuffled_instances = copy(train_data.instances)
        rand.shuffle(shuffled_instances)



        kfold = KFold(n_splits=num_folds, random_state=None, shuffle=False)
        computed_folds = list(kfold.split(shuffled_instances))

        for fold in range(num_folds):
            train_indexes, test_indexes = computed_folds[fold]
            new_train = [shuffled_instances[i] for i in train_indexes]
            new_test = [shuffled_instances[i] for i in test_indexes]
            fold_train.append(AllennlpDataset(new_train, vocab=vocab))
            fold_test.append(AllennlpDataset(new_test, vocab=vocab))

            keys = deepcopy(dl_params.as_dict())
            keys.update({"dataset": fold_test[-1]})
            fold_test_loader.append(DataLoader.from_params(params.pop("fold_test_data_loader",keys)))

            keys = deepcopy(dl_params.as_dict())
            keys.update({"dataset": fold_train[-1]})
            fold_train_loader.append(DataLoader.from_params(params.pop("fold_train_data_loader", keys)))

        for fold in ([in_fold] if in_fold is not None else range(num_folds)):
            fold_model = deepcopy(master_model)
            eval_epoch_callback = EvalEpochCallback(fold, fold_test_loader[fold], test_loader, global_metrics)
            callbacks = [eval_epoch_callback]
            if ewc_weight is not None:
                ewc = EWC(extra_test_loader)

                def ewc_forward(*args, **kwargs) -> Dict[str, torch.Tensor]:
                    ewc_loss = 0
                    if ewc.model.training:
                        ewc_loss = ewc.penalty(ewc.model)
                    ret = ewc.model.old_forward(*args, **kwargs)
                    ret["loss"] += ewc_weight * ewc_loss
                    return ret

                fold_model.old_forward = fold_model.forward
                fold_model.forward = ewc_forward
                callbacks.append(CallLossCallback(ewc))

            trainer = Trainer.from_params(model=fold_model,
                                          serialization_dir=serialization_dir,
                                          data_loader=fold_train_loader[fold],
                                          train_data=train_data,
                                          validation_data=None,
                                          params=Params(deepcopy(master_trainer)),
                                          validation_data_loader=None,
                                          epoch_callbacks=callbacks)

            training_metrics.append(trainer.train())
            del fold_model
            del trainer
            del eval_epoch_callback

            state = glob(serialization_dir+"/*.th")
            for file in state:
                logger.info("deleting state - {}".format(file))
                os.unlink(file)
    else:
        callbacks = []
        if ewc_weight is not None:
            ewc = EWC(extra_test_loader)

            def ewc_forward(*args, **kwargs) -> Dict[str, torch.Tensor]:
                ewc_loss = 0
                if ewc.model.training:
                    ewc_loss = ewc.penalty(ewc.model)
                ret = ewc.model.old_forward(*args, **kwargs)
                ret["loss"] += ewc_weight * ewc_loss
                return ret

            model.old_forward = model.forward
            model.forward = ewc_forward
            callbacks.append(CallLossCallback(ewc))

        keys = deepcopy(dl_params.as_dict())
        keys.update({"dataset": train_data})
        train_data.index_with(vocab)
        train_data_loader = DataLoader.from_params(params.pop("train_loader",keys))

        if validation_data is not None:
            validation_data.index_with(vocab)
            keys = deepcopy(dl_params.as_dict())
            keys.update({"dataset": validation_data})

            validation_data_loader = DataLoader.from_params(params.pop("validation_loader", keys))
        else:
            validation_data_loader = None

        if "finetune" in dir(model):
            model.finetune()
            logger.info("Fine tuning model")
        trainer = Trainer.from_params(model=model,
                                      serialization_dir=serialization_dir,
                                      data_loader=train_data_loader,
                                      train_data=train_data,
                                      validation_data=None,
                                      params=Params(deepcopy(master_trainer)),
                                      validation_data_loader=validation_data_loader,
                                      epoch_callbacks=callbacks)

        training_metrics = trainer.train()
        archive_model(serialization_dir)

    final_metrics["fine_tune"] = global_metrics
    final_metrics["training"] = training_metrics

    metrics_json = json.dumps(final_metrics, indent=2)
    with open(os.path.join(serialization_dir, "metrics.json"), "w") as metrics_file:
        metrics_file.write(metrics_json)
    logger.info("Metrics: %s", metrics_json)
    return model