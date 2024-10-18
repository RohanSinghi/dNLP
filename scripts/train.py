import logging
from src.utils.utils import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Starting the training process...")
    # Your training code here
    logger.info("Training completed.")

if __name__ == "__main__":
    main()