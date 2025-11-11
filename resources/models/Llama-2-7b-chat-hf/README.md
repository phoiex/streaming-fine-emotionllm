• # Llama‑2‑7b‑chat‑hf 本地模型仓库说明

  本仓库包含 Meta Llama‑2‑7B Chat（Hugging Face Transformers 格式）的完整权重与分词器文件，便于离线加载与推理。适用
  于需要本地化部署或内网环境的场景。

  ## 目录结构

  .
  ├── config.json
  ├── generation_config.json
  ├── tokenizer.model
  ├── tokenizer.json
  ├── tokenizer_config.json
  ├── special_tokens_map.json
  ├── model-00001-of-00002.safetensors
  ├── model-00002-of-00002.safetensors
  ├── model.safetensors.index.json
  ├── pytorch_model-00001-of-00002.bin
  ├── pytorch_model-00002-of-00002.bin
  ├── pytorch_model.bin.index.json
  ├── LICENSE.txt
  ├── README.md
  ├── USE_POLICY.md
  └── configuration.json

  ## 文件说明

  - config.json：Transformers 模型配置（结构超参、词表大小、层数、头数、精度等）。
  - generation_config.json：推理默认参数（温度、top_p、max_length、pad/eos/bos 等），加载后作为
    model.generation_config。
  - tokenizer.model：SentencePiece 分词模型原始文件。
  - tokenizer.json：Hugging Face Tokenizers 的极速分词格式（等价功能，加载更快）。
  - tokenizer_config.json：分词器行为配置（如 add_bos_token、padding_side、以及聊天 chat_template）。
  - special_tokens_map.json：特殊标记映射（<s>、</s>、<unk>）。
  - model-0000x-of-00002.safetensors + model.safetensors.index.json：safetensors 格式分片与索引（更安全、可流式
    加载）。
  - pytorch_model-0000x-of-00002.bin + pytorch_model.bin.index.json：PyTorch .bin 格式分片与索引（兼容旧环境）。
      - 同时存在两种格式时，Transformers 通常优先使用 safetensors（若已安装该库），否则回退到 .bin。
  - LICENSE.txt：Meta 模型许可条款。
  - USE_POLICY.md：可接受使用政策（AUP），说明禁止用途与合规要求。
  - README.md：上游英文模型卡与背景信息。
  - configuration.json：占位文件（当前为空，可忽略）。

  ## 快速开始（本地加载）

  from transformers import AutoTokenizer, AutoModelForCausalLM
  import torch

  tok = AutoTokenizer.from_pretrained(".")
  model = AutoModelForCausalLM.from_pretrained(
      ".", torch_dtype="auto", device_map="auto"
  )

  # 使用内置 chat_template
  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "你好！用一句话介绍一下你自己。"},
  ]
  inputs = tok.apply_chat_template(
      messages, add_generation_prompt=True,
      return_tensors="pt"
  ).to(model.device)

  outputs = model.generate(inputs, max_new_tokens=64)
  print(tok.decode(outputs[0], skip_special_tokens=True))

  ## 注意与合规

  - 使用与分发需遵守 LICENSE.txt 与 USE_POLICY.md；请勿用于政策中列明的禁止用途。
  - 加载需要与之兼容的 transformers/torch 版本及足够内存（7B FP16 通常需要 ≥14–16GB 显存；CPU 需耐心等待）。