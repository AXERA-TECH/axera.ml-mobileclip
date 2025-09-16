import numpy as np
from PIL import Image
import mobileclip
import open_clip
import onnxruntime as ort
import torch
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def image_transform_v2():
    resolution = 256
    resize_size = resolution
    centercrop_size = resolution
    mean = OPENAI_DATASET_MEAN
    std = OPENAI_DATASET_STD
    aug_list = [
        Resize(
            resize_size,
            interpolation=InterpolationMode.BICUBIC,
        ),
        CenterCrop(centercrop_size),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ]
    preprocess = Compose(aug_list)
    return preprocess
    
    
def image_transform_v1():
    resolution = 256
    resize_size = resolution
    centercrop_size = resolution
    aug_list = [
        Resize(
            resize_size,
            interpolation=InterpolationMode.BILINEAR,
        ),
        CenterCrop(centercrop_size),
        ToTensor(),
    ]
    preprocess = Compose(aug_list)
    return preprocess


def softmax(x, axis=-1):
    """
    对 numpy 数组在指定维度上应用 softmax 函数
    
    参数:
        x: numpy 数组，输入数据
        axis: 计算 softmax 的维度，默认为最后一个维度 (-1)
    
    返回:
        经过 softmax 处理的 numpy 数组，与输入形状相同
    """
    # 减去最大值以防止数值溢出（数值稳定化）
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    # 计算每个元素的指数与所在维度总和的比值
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


model_name = "MobileCLIP2-S2"
image_encoder_path = "models/MobileCLIP2-S2_image_encoder.onnx"
text_encoder_path = "models/MobileCLIP2-S2_text_encoder.onnx"

if model_name in ["mobileclip_b", "mobileclip_s0", "mobileclip_s1", "mobileclip_s2"]:
    is_v1 = True
else: # ["MobileCLIP2-S0"， "MobileCLIP2-S2", "MobileCLIP2-S3"， "MobileCLIP2-S4"]
    is_v1 = False

if is_v1:
    preprocess = image_transform_v1()
    tokenizer = mobileclip.get_tokenizer(model_name) # or open_clip.get_tokenizer("ViT-B-16")
else:
    # NOTICE: MobileCLIP2-S0 和 MobileCLIP2-S2需要v1的预处理，v2的预处理在pulsar2中量化误差比较大
    # MobileCLIP2-S3 和 MobileCLIP2-S4 都可以
    preprocess = image_transform_v1()
    tokenizer = open_clip.get_tokenizer(model_name)
    
image = preprocess(Image.open("docs/fig_accuracy_latency.png").convert('RGB')).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])
text = text.to(torch.int64)

onnx_image_encoder = ort.InferenceSession(image_encoder_path)
onnx_text_encoder = ort.InferenceSession(text_encoder_path)

image_features = onnx_image_encoder.run(["unnorm_image_features"],{"image":np.array(image)})[0]
# 如果text_encoder的输入长度不固定，需要for循环跑
# text_features = []
# for i in range(text.shape[0]):
#     text_feature = onnx_text_encoder.run(["unnorm_text_features"],{"text":np.array([text[i]])})[0]
#     text_features.append(text_feature)
# text_features = np.array([t[0] for t in text_features])
# np.save("image_features.npy",image_features)
text_features = onnx_text_encoder.run(["unnorm_text_features"], {"text": text.numpy()})[0]
image_features /= np.linalg.norm(image_features, ord=2, axis=-1, keepdims=True)
text_features /= np.linalg.norm(text_features, ord=2, axis=-1, keepdims=True)

text_probs = softmax(100.0 * image_features @ text_features.T)

print("Label probs:", text_probs)