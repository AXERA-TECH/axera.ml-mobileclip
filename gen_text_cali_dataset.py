import numpy as np
import torch
from imagenet_dataset import ImagenetDataset, imagenet_classes, imagenet_templates
import mobileclip
import open_clip
import tarfile
import os

model_name = "MobileCLIP2-S4"

if model_name in ["mobileclip_b", "mobileclip_s0", "mobileclip_s1", "mobileclip_s2"]:
    tokenizer = mobileclip.get_tokenizer(model_name)
else: # ["MobileCLIP2-S2", "MobileCLIP2-S4"]
    tokenizer = open_clip.get_tokenizer(model_name)

if not os.path.exists(f"calib_data/{model_name}_text_calib"):
    os.makedirs(f"calib_data/{model_name}_text_calib")
    
calib_file = tarfile.open(f"calib_data/{model_name}_text_calib/{model_name}.tar", "w")

for i, classname in enumerate(imagenet_classes):
    if i>=64:
        break

    idx = np.random.randint(0, 79, 3)

    texts = [imagenet_templates[i].format(classname) for i in idx]
    # format with class
    texts = tokenizer(texts)  # tokenize
    texts = texts.to(torch.int64).numpy()
    inputs = { "text": texts }
    s_path = f"calib_data/{model_name}_text_calib/{i}.npy"
    print("save: ", s_path, texts.shape)
    np.save(s_path, inputs)
    calib_file.add(s_path)
    
calib_file.close()
print("calib data saved in: ", f"calib_data/{model_name}_text_calib/{model_name}.tar")