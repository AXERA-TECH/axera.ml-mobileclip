import torch
import open_clip
from PIL import Image
import mobileclip
from mobileclip.modules.common.mobileone import reparameterize_model

model_name = "mobileclip_s2" 
pretrained_path = "/data/wangjian/project/hf_cache/mobileclip/MobileCLIP2-S2/mobileclip2_s2.pt"

if model_name in ["mobileclip_b", "mobileclip_s0", "mobileclip_s1", "mobileclip_s2"]:
    is_v1 = True
else: # ["MobileCLIP2-S2", "MobileCLIP2-S4"]
    is_v1 = False

# v1
if is_v1:
    model, _, preprocess = mobileclip.create_model_and_transforms(model_name, pretrained=pretrained_path)
    tokenizer = mobileclip.get_tokenizer(model_name)
else: # v2
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_path)
    tokenizer = open_clip.get_tokenizer(model_name)

    # For inference/model exporting purposes, please reparameterize first
    model = reparameterize_model(model.eval())

image = preprocess(Image.open("docs/fig_accuracy_latency.png").convert('RGB')).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])  # 如果为了通用性，设置text_encoder的输入长度为1，实际运行时需要for循环跑。这边假设输入长度固定.

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
    
if is_v1:
    image_encoder = model.image_encoder
    text_encoder = model.text_encoder
else:
    image_encoder = model.visual
    text_encoder = model.text
    
# export image onnx
torch.onnx.export(image_encoder,
            image,
            f"./models/{model_name}_image_encoder.onnx",
            input_names=['image'],
            output_names=['unnorm_image_features'],
            export_params=True,
            opset_version=14,)

# import text onnx
torch.onnx.export(text_encoder,
            text,
            f"./models/{model_name}_text_encoder.onnx",
            input_names=['text'],
            output_names=['unnorm_text_features'],
            export_params=True,
            opset_version=14,)
