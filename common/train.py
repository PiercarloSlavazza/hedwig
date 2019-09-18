from common.trainers.classification_trainer import ClassificationTrainer
from common.trainers.relevance_transfer_trainer import RelevanceTransferTrainer


class TrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    trainer_map = {
        'Reuters': ClassificationTrainer,
        'AAPD': ClassificationTrainer,
        'IMDB': ClassificationTrainer,
        'Wired': ClassificationTrainer,
        'Yelp2014': ClassificationTrainer,
        'Robust04': RelevanceTransferTrainer,
        'Robust05': RelevanceTransferTrainer,
        'Robust45': RelevanceTransferTrainer,
    }

    @staticmethod
    def get_trainer(dataset_name, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):

        if dataset_name not in TrainerFactory.trainer_map:
            raise ValueError('{} does not have a known Trainer.'.format(dataset_name))

        return TrainerFactory.trainer_map[dataset_name](
            model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator
        )
