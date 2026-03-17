import torch.nn as nn
import torch
from architecture import createdLLM
from image_data import tokens_to_image_test
import joblib
import matplotlib.pyplot as plt
# This is really cool and I think i have to save these lol:
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## same config from image_model
IMAGE_MODE_CONFIG = {
    "vocab_size": 512,
    "context_length": 16,
    "embed_dim" : 64, 
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.3,
    "qkv_bias": False
}
K_MEANS_PATH = "kmeans.pkl"

model = createdLLM(IMAGE_MODE_CONFIG)
char_prev_state = torch.load('character_multimodal_image_first_train_embed_128.pt', map_location=device)
model.load_state_dict(char_prev_state)
model.to(device) # let's go!
model.eval()
reference_kmeans = joblib.load(K_MEANS_PATH)

def generate_character(model, kmeans, device, temper=0.4):
    tokens = []

    with torch.no_grad():
        for i in range(16):
            if len(tokens) == 0:
                next_token = torch.randint(0, IMAGE_MODE_CONFIG["vocab_size"], (1,))
                next_token = next_token.item()
            else:
                x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
                logits = model(x)
                logits = logits[0, -1, :] / temper
                probs = torch.softmax(logits, dim=-1) 
                # this next part is really interesting. I've really used argmax each time after softmax:
                next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)

    print(f" Inference done!! Passing to the decoder")
    decoded_tokens = tokens_to_image_test(tokens, kmeans)
    print(f" Decoder-step reached. Returning...")
    return decoded_tokens

def check_state_dict(path):
    statedict = torch.load(path, map_location=device)
    for k, v in statedict.items():
        print(k, v.shape)

if __name__ == "__main__":
    print(f" In main! ~ ")
    #check_state_dict('character_multimodal_image_first_train.pt')

    # i totally forgot that I actually decoded this earlier:
    returned_image = generate_character(model, reference_kmeans, device, 1.0)
    print(f" Got returned tokens. Shape: {returned_image.shape}")

    # show :)
    plt.imshow(returned_image)
    plt.axis('off')
    plt.savefig('second_generated_character_pixel.png')
    plt.show()
    
    