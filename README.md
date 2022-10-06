# thops

基于openmmlab v2的ocr工具库

## PP-OCR

### 转换模型

1. 下载[PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR)的训练和推理模型权重
2. 将ppocr检测和识别模型转换为pytorch

```bash
python tools/ppocr_tools/ppocr_det2torch.py
python tools/ppocr_tools/ppocr_rec2torch.py
```

### 模型推理

```bash
python tools/ppocr_tools/ppocr_inference.py -i demo/2.jpg
```
