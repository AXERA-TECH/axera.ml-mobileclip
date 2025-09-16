import numpy as np
from PIL import Image
import axengine as ort
import torch
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop
from tokenizer import SimpleTokenizer
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ie", "--image_encoder_path", type=str, default="./mobileclip2_s4_image_encoder.axmodel", 
                        help="image encoder axmodel path")
    parser.add_argument("-te", "--text_encoder_path", type=str, default="./mobileclip2_s4_text_encoder.axmodel", 
                        help="text encoder axmodel path")
    parser.add_argument("-i", "--image", type=str, default="./zebra.jpg", 
                        help="input image path")
    parser.add_argument("-t", "--class_text", type=str, nargs='+', default=["a zebra", "a dog", "two zebras"], 
                        help='List of captions, e.g.: "a zebra" "a dog" "two zebras"')
    args = parser.parse_args()

    image_encoder_path = args.image_encoder_path
    text_encoder_path = args.text_encoder_path
    # NOTICE: 使用v1的预处理，v2的预处理方式在pulsar2中量化误差比较大
    preprocess = image_transform_v1() 
    tokenizer = SimpleTokenizer(context_length=77)

    image = preprocess(Image.open(args.image).convert('RGB')).unsqueeze(0)
    text = tokenizer(args.class_text)
    text = text.to(torch.int32)

    onnx_image_encoder = ort.InferenceSession(image_encoder_path)
    onnx_text_encoder = ort.InferenceSession(text_encoder_path)

    image_features = onnx_image_encoder.run(["unnorm_image_features"],{"image":np.array(image)})[0]
    # text_features = []
    # for i in range(text.shape[0]):
    #     text_feature = onnx_text_encoder.run(["unnorm_text_features"],{"text":np.array([text[i]])})[0]
    #     text_features.append(text_feature)
    # text_features = np.array([t[0] for t in text_features])
    text_features = onnx_text_encoder.run(["unnorm_text_features"], {"text": text.numpy()})[0]
    image_features /= np.linalg.norm(image_features, ord=2, axis=-1, keepdims=True)
    text_features /= np.linalg.norm(text_features, ord=2, axis=-1, keepdims=True)

    text_probs = softmax(100.0 * image_features @ text_features.T)

    print("Label probs:", text_probs)