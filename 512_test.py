from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ds = load_dataset("brivangl/midjourney-v6-llava", split="train", streaming=True)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for i, example in enumerate(ds):
    if i >= 12:
        break
    
    img = example['image']
    
    # try different sizes
    sizes = [64, 32]
    size = sizes[i % 2]
    
    resized = img.resize((size, size), Image.NEAREST)
    axes[i//4][i%4].imshow(resized)
    axes[i//4][i%4].set_title(f"{size}x{size}")
    axes[i//4][i%4].axis('off')

plt.tight_layout()
plt.show()