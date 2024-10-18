import logging
from src.utils.utils import get_logger

logger = get_logger(__name__)


def preprocess_text(text):
    logger.info("Preprocessing text...")
    # Your text preprocessing logic here
    logger.info("Text preprocessed.")
    return text.lower()