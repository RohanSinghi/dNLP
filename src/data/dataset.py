import logging
from src.utils.utils import get_logger

logger = get_logger(__name__)

# Example of using the logger in a module
logger.info("Dataset module initialized")

class Dataset:
    def __init__(self):
        logger.debug("Dataset object created.")

    def load_data(self):
        logger.info("Loading dataset...")
        # your loading logic here
        logger.info("Dataset loaded successfully.")

        return ["Sample", "Data"] # Return dummy data to avoid errors