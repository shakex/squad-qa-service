# transformers-qa-service


## Pre
```bash
git clone https://github.com/shakex/squad-qa-service.git
pip install -r requirements.txt
```

## Start server

linux
```bash
cd /path/to/transformers-qa-service
./start.sh
```

windows
```bat
cd /path/to/transformers-qa-service
./start.bat
```


## APIs

- [post] http://localhost:8008/questionAnswering

request
```json
{
    "question": "什么是管理费？",
    "context": "6.5.1 第6.1.1条（4）项所称“管理费”系指本有限合伙企业应按本协议及管理协议规定向普通合伙人及/或普通合伙人代表本有限合伙企业指定的第三方支付的管理费。6.5.2 管理费按照各项目投资单独核算，各项目投资管理费计算期间为各项目投资对应的出资到账截止日起至本有限合伙企业完全退出该项目投资之日。管理费以项目投资对应的第一个出资到账截止日起一年届满之日及之后每年为一个收费期间，最后一个收费期间为该收费期间起始日起至管理费计算期间届满之日，每个收费期间的应收管理费于该期间起始日前 5 日支付给普通合伙人及/或普通合伙人代表本有限合伙企业指定的第三方。"
}
```

response
```json
{
    "answer": "系指本有限合伙企业应按本协议及管理协议规定向普通合伙人及/或普通合伙人代表本有限合伙企业指定的第三方支付的管理费"
}
```

## Pretrained Models

| Name      | Size |
| ----------- | ----------- |
| [roberta-base-chinese-extractive-qa (onnx, quantized)](https://huggingface.co/uer/roberta-base-chinese-extractive-qa)      | 97.4M       |



> maintainer: xik18384