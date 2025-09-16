# axera.ml-mobileclip

[ml-mobileclip](https://github.com/apple/ml-mobileclip) demo on axera，目前完成以下模型的适配
- MobileCLIP-S2
- MobileCLIP2-S0
- MobileCLIP2-S2
- MobileCLIP-S3
- MobileCLIP2-S4


## 支持平台
- [x] AX650N
- [ ] AX630C

### env

根据原[repo](https://github.com/apple/ml-mobileclip)配置运行环境
```
conda create -n clipenv python=3.10
conda activate clipenv
pip install -e .
```
mobileclip2模型需要用到[open_clip](https://github.com/mlfoundations/open_clip)库，配置环境如下
```bash
conda create -n clipenv python=3.10
conda activate clipenv

# Clone OpenCLIP repository, add MobileCLIP2 models, and install
git clone https://github.com/mlfoundations/open_clip.git
pushd open_clip
git apply ../mobileclip2/open_clip_inference_only.patch
cp -r ../mobileclip2/* ./src/open_clip/
pip install -e .
popd

pip install git+https://github.com/huggingface/pytorch-image-models
```

补充onnx相关包
```
pip install onnx
pip install onnxruntime
pip install opencv-python
```

### 导出模型(PyTorch -> ONNX)
从Hugging Face上获取MobileCLIP-S2，MobileCLIP2-S2，MobileCLIP2-S4模型权重
```bash
# 切换到国内镜像
export HF_ENDPOINT=https://hf-mirror.com

hf download apple/MobileCLIP-S2  --local-dir /data/wangjian/project/hf_cache/mobileclip/MobileCLIP-S2

hf download apple/MobileCLIP2-S2  --local-dir /data/wangjian/project/hf_cache/mobileclip/MobileCLIP2-S2

hf download apple/MobileCLIP2-S4 --local-dir /data/wangjian/project/hf_cache/mobileclip/MobileCLIP2-S4
```
执行脚本
```
python export_onnx.py
```
导出成功后会生成两个onnx模型，以mobileclip_s2为例:
- image encoder: mobileclip_s2_image_encoder.onnx
- text encoder: mobileclip_s2_text_encoder.onnx


#### 转换模型(ONNX -> AXMODEL)
使用模型转换工具 Pulsar2 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 .axmodel，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 Pulsar2 build 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考[AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)


#### 量化数据集准备
此处仅用作demo，建议使用实际参与训练的数据
- image数据:

    imagenet-calib.tar

- text数据:
    ```
    python gen_text_cali_dataset.py
    ```
最终得到两个数据集：

- dataset_v04.zip
- ./calib_data/MobileCLIP2-S2_text_calib/MobileCLIP2-S2.tar

注：对分数据集建议用实际使用场景的数据，此处仅用于演示

#### 模型编译
修改配置文件，检查config.json 中 calibration_dataset 字段，将该字段配置的路径改为上一步准备的量化数据集存放路径，在build_config目录中已提供必要的配置json。


在编译环境中，执行pulsar2 build参考命令：
- mobileclip_s2

    ```bash
    # image encoder
    pulsar2 build --config build_config/mobileclip_s2_image_u16.json --input models/mobileclip_s2_image_encoder.onnx --output_dir build_output/image_encoder --output_name mobileclip_s2_image_encoder.axmodel

    # text encoder
    pulsar2 build --config build_config/mobileclip_s2_text_u16.json --input models/mobileclip_s2_text_encoder.onnx --output_dir build_output/text_encoder --output_name mobileclip_s2_text_encoder.axmodel
    ```
- mobileclip2_s2

    ```bash
    # image encoder
    pulsar2 build --config build_config/mobileclip2_s2_image_u16.json --input models/mobileclip2_s2_image_encoder.onnx --output_dir build_output/image_encoder --output_name mobileclip2_s2_image_encoder.axmodel --onnx_opt.disable_transformation_check

    # text encoder
    pulsar2 build --config build_config/mobileclip2_s2_text_u16.json --input models/mobileclip2_s2_text_encoder.onnx --output_dir build_output/text_encoder --output_name mobileclip2_s2_text_encoder.axmodel
    ```
- mobileclip2_s4
    ```bash
    # image encoder
    pulsar2 build --config build_config/mobileclip2_s4_image_u16.json --input models/mobileclip2_s4_image_encoder.onnx --output_dir build_output/image_encoder --output_name mobileclip2_s4_image_encoder.axmodel

    # text encoder
    pulsar2 build --config build_config/mobileclip2_s4_text_u16.json --input models/mobileclip2_s4_text_encoder.onnx --output_dir build_output/text_encoder --output_name mobileclip2_s4_text_encoder.axmodel
    ```


编译完成后得到两个axmodel模型， 以mobileclip_s2为例：
- mobileclip_s2_image_encoder.axmodel
- mobileclip_s2_text_encoder.axmodel


### Python API 运行
需基于[PyAXEngine](https://github.com/AXERA-TECH/pyaxengine)在AX650N上进行部署

demo基于原repo中的提取图文特征向量并计算相似度，将两个axmodel和run_onboard目录下的所有文件scp到开发板上后，运行run_axmodel.py文件
```bash
python3 run_axmodel.py -ie ./mobileclip2_s4_image_encoder.axmodel -te ./mobileclip2_s4_text_encoder.axmodel -i ./zebra.jpg -t "a zebra" "a dog" "two zebras"
```

1. 输入图片：

    ![](run_onboard/zebra.jpg)

2. 输入文本：

    ["a zebra", "a dog", "two zebras"]

3. 输出类别置信度：

    Label probs: [[6.095444e-02 5.628616e-14 9.390456e-01]]

## 技术讨论

- Github issues
- QQ 群: 139953715