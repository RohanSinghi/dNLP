import logging
from src.utils.utils import get_logger

logger = get_logger(__name__)

logger.info("Model module initialized")

class Model:
    def __init__(self):
        logger.debug("Model object created.")

    def train(self, data):
        logger.info("Training Model...")
        # your training logic here
        logger.info("Model trained successfully.")

    def predict(self, data):
        logger.info("Predicting...")
        # your predictions logic here
        logger.info("Prediction complete.")
        return ["Sample", "Prediction"] # Return dummy data to avoid errors