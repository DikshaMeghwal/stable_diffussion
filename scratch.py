from tqdm import tqdm
from time import sleep

for epoch in range(5):
    prog_bar = tqdm(range(100), desc=f"Epoch {epoch}")  # Create a new tqdm instance
    for i in prog_bar:
        prog_bar.set_description(f"Epoch {epoch}_{i}")
        sleep(0.5)
