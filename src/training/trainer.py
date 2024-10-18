import logging
from src.utils.utils import get_logger

logger = get_logger(__name__)

logger.info("Trainer module initialized")

class Trainer:
    def __init__(self, model, dataset):
        logger.debug("Trainer object created.")
        self.model = model
        self.dataset = dataset

    def train(self):
        logger.info("Starting training...")
        data = self.dataset.load_data()
        self.model.train(data)
        logger.info("Training complete.")