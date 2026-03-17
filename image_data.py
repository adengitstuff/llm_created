from datasets import load_dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
import joblib
from sklearn.cluster import MiniBatchKMeans

#ds = load_dataset("jiovine/pixel-art-nouns-2k", split="train")
#print(ds)
#print(ds[0])  # see what one example looks like
#print(ds[0]['image'].size)  # check actual image dimensions
#ds[0]['image'].show()

def show_npy_images(imges):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(imges[i])
        axes[i].axis('off')
    plt.show()

# for i in range(5):
#     ds[i]['image'].show()
#     print(ds[i]['text'])

## resize to 32x32! I'm a student on a 16GB GPU, lol!
def load_and_resize(ds):
    images = []
    for example in ds:
        img = example['image'].resize((32, 32), Image.NEAREST) # testing the nearest instead bilinear interpolation (reverse interpolation?). Just grabs 
        images.append(np.array(img))                           #            the nearest input pixel for each output pixel
    return np.array(images)

#images = load_and_resize(ds)
#print(images.shape)  # should be (2000, 32, 32, 3)
#show_npy_images(images)


def extract_patches_from_the_image(images):
    """
    This is taking the 32x32 images and then
    turning them into patches! This reminds me of VIT stuff. 
    I'll do a 4x4 grid, so 16 cells. each is 8 pixels wide, 8 pixels tall
    """
    # images comes in as [batch, 32, 32, 3]
    batch, height, width, rgb = images.shape
    print(f" batch, h, w, rgb: {batch}, {height}, {width}, {rgb}")
    patches = images.reshape(batch, 4, 8, 4, 8, rgb) # reshape 
    # change the row and column dimensions so that if I flatten it, it returns the pixel in a patch (not going
    #       across, down, etc - but just an 8x8 grid's patches)
    patches = patches.transpose(0, 1, 3, 2, 4, 5) # I guess h or width first is arbitrary
    # Flatten!:
    patches = patches.reshape(batch, 16, 8*8*3) # now, each patch is kind of like a linear layer projected it out - a 192-dim "embedding"!
    return patches  # [batch, 16, 192]

#patch_test = extract_patches_from_the_image(images)
#print(patch_test.shape)

def reshape_and_kmeans(patchesx):
    all_patches = patchesx.reshape(-1, 192) # collapse the batch dim
    print(f" all_patches shape: {all_patches.shape}. should be [high number, 192]")

    new_kmeans_name = 'kmeans_2048_full.pkl'

    # The CPU on my humble student rig is at 100%. Let me try 6 cores LOL
    #kmeans = KMeans(n_clusters=512, random_state=42, n_init=10, n_jobs=6) 

    # Testing the MiniBatchKmeans!
    kmeans = MiniBatchKMeans(n_clusters=2048, random_state=42, batch_size=8000)
    kmeans.fit(all_patches) # The fact that this is this abstracted-away is really cool. "kmeans fit". Euclidean distance for pixels! 

    print(f" Done with reshape and kmeans")
    joblib.dump(kmeans, new_kmeans_name)
    return new_kmeans_name

#reshape_and_kmeans(patch_test)
#print(F" Done! Kmeans saved!")

def tokenize_kmeans(path_to_kmeans_patches, patches_flattened):
    """ 
    Pass in the path to a saved kmeans file
    """
    kmeans = ""
    try: 
        kmeans = joblib.load(path_to_kmeans_patches)
    except:
        print(f" Exception/error in trying to load the kmeans file. The path is {path_to_kmeans_patches}")

    patches_collapsed = patches_flattened.reshape(-1, 192)
    token_sequences = kmeans.predict(patches_collapsed)
    token_sequences = token_sequences.reshape(-1, 16) # change from reshape(2000, 16). try new full dataset size
    print(token_sequences.shape)  # (2000, 16)

    ## This is really cool: 
    print(token_sequences[0])     # what the integer-sprite looks like!
    return token_sequences, kmeans # This returns the actual kmeans object


#first_token_sequences, kmeans_object = tokenize_kmeans("kmeans.pkl", patch_test)

def tokens_to_image_test(tokens, kmeans):
    patches = kmeans.cluster_centers_[tokens]  # (16, 192)
    patches = patches.reshape(4, 4, 8, 8, 3)   # 4x4 grid of 8x8 patches
    patches = patches.transpose(0, 2, 1, 3, 4) # put pixels back in order
    image = patches.reshape(32, 32, 3)          # stitch into full image
    image = image.clip(0, 255).astype(np.uint8) # clean up float values
    return image

# # display it
# decoded = decode_tokens_to_image_test(first_token_sequences[0], kmeans_object)
# plt.imshow(decoded)
# plt.title("decoded sprite")
# plt.show()

## Editing to take in a char dataset path:
def imgs_to_tokens(dataset_name="jiovine/pixel-art-nouns", kmeans_path="kmeans_2048_full.pkl"):
    #ds = load_dataset("jiovine/pixel-art-nouns-2k", split="train")
    dataset_chars = load_dataset(dataset_name, split="train")
    images = load_and_resize(dataset_chars)
    patches = extract_patches_from_the_image(images)
    token_sequence, _ = tokenize_kmeans(kmeans_path, patches)
    return token_sequence

if __name__ == "__main__":
    # the 49k parquet rows one... was there the entire time. Lol. Changed from "jiovine/pixel-art-nouns-2k" to:
    character_dataset = load_dataset("jiovine/pixel-art-nouns", split="train")
    npy_images = load_and_resize(character_dataset)
    flatten_patches = extract_patches_from_the_image(npy_images)
    full_kmeans_new = reshape_and_kmeans(flatten_patches)
    # fresh_tokens, kmeans_full_loaded = tokenize_kmeans(full_kmeans_new, flatten_patches)