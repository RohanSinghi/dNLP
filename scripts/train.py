import argparse
import multiprocessing as mp
import time

# Example dummy training function (replace with your actual training logic)
def train(process_id):
    print(f"Process {process_id}: Starting training...")
    time.sleep(5)  # Simulate some work
    print(f"Process {process_id}: Training complete.")


def main_process():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    # Load config (example)
    print(f"Loading config from {args.config}")
    # Replace this with your actual loading of the config

    # Start training (this is a placeholder for your actual training loop)
    train(mp.current_process().name)

def main():
    processes = []
    # Create 2 processes
    for i in range(2):
        p = mp.Process(name=f"Process-{i}", target=main_process)
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All training processes finished.")


if __name__ == "__main__":
    # Check to see if it is a frozen module
    mp.freeze_support()
    main()
