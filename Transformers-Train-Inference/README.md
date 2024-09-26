# 🎯 Transformers-Train-Inference

Bem-vindo ao **Transformers-Train-Inference**, um projeto voltado para o treinamento e inferência de modelos de **Detecção de Objetos** utilizando **Transformers**! Aqui, você encontrará tudo o que precisa para treinar seu próprio modelo e realizar inferência em tempo real com imagens ou streams de câmera IP.

## 📂 Estrutura do Projeto

```bash
Transformers-Train-Inference
├── README.md                 
├── inference                 # Código de inferência
│   ├── cameraIP_RealTime.py   # Inferência em tempo real com câmera IP
│   ├── output.mp4             # Exemplo de saída de inferência em vídeo
│   ├── requirements.txt       # Dependências da inferência
│   └── testImage_camIP.py      # Inferência de imagem estática
└── train                     # Código de treinamento
    ├── code-references        # Referências para código de treinamento
    │   ├── Cópia_de_object_detection.ipynb
    │   └── run_object_detection.py
    ├── datasets               # Datasets de treinamento (detecção de EPI)
    │   ├── Construction-Site-Safety-28
    │   ├── cppe5
    │   └── PPEv5-17
    ├── models                 # Modelos treinados
    │   ├── detr-finetuned-ppe
    │   └── modelo-treinado-1epoch
    ├── info.md                # Informações sobre o treinamento
    ├── requirements.txt       # Dependências do treinamento
    ├── test-trainedModel.py   # Teste do modelo treinado
    └── train.py               # Script de treinamento
```

## 🛠️ Preparando o Ambiente

### Passos para configurar o ambiente de **Inferência** e **Treinamento**:
> Nota: inferência e treinamento devem possuir ambientes separados, eu só testei assim e não sei se funciona um ambiente com todas as dependências. **Tive muitos problemas com dependências por isso fiz assim**. Para cada um, você deve:

1. **Criar ambiente Conda** (Python 3.9):
   ```bash
   conda create --name transformers-train-inference python=3.9
   ```

2. **Ativar o ambiente**:
   ```bash
   conda activate transformers-train-inference
   ```

3. **Instalar PyTorch**:
   ```bash
   pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
   ```
> Nota: pode ser também ``pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124``

4. **Instalar dependências** (para cada diretório):
   - Para **Inferência**:
     ```bash
     cd inference
     pip install -r requirements.txt
     ```
   - Para **Treinamento**:
     ```bash
     cd train
     pip install -r requirements.txt
     ```

## 🚀 Como Usar

### 🔧 Treinamento

A pasta `train` contém o código para fazer **fine-tuning** de um modelo de detecção de objetos em um dataset customizado.

- O script principal para treinar é o `train.py`.
- Você pode testar o modelo treinado usando o arquivo `test-trainedModel.py`.
- Os **modelos treinados** estão na pasta `models`.
- Para explorar datasets de **PPE (Equipamento de Proteção Individual)**, acesse a pasta `datasets`.

#### 📖 Customização do Código

O código para treinamento do modelo é implementado no arquivo `train.py`, onde você pode fazer ajustes conforme necessário:

1. **Modelo Utilizado**:
   - O modelo padrão é definido pela variável `model_name`:
     ```python
     model_name = "facebook/detr-resnet-50"
     ```
   - Para mudar o modelo, basta alterar o valor de `model_name` para outro modelo suportado pela biblioteca Transformers.

2. **Dataset**:
   - O dataset utilizado é carregado com a função `load_dataset`:
     ```python
     ds = load_dataset("Francesco/construction-safety-gsnvb")
     ```
   - Você pode modificar a linha acima para usar seu próprio dataset. Certifique-se de que ele esteja no formato esperado pela função.


>Nota: Para saber mais sobre como preparar seu próprio dataset customizado, confira este [guia](https://github.com/huggingface/transformers/blob/main/examples/pytorch/object-detection/README.md).

### 📸 Inferência

Na pasta `inference`, você encontrará dois scripts de inferência:

1. **testImage_camIP.py**: Realiza a inferência de uma **imagem estática** usando um modelo vindo do **Hugging Face**. Ele carrega a imagem, processa e exibe o resultado com as **bounding boxes** detectadas.
   
2. **cameraIP_RealTime.py**: Faz inferência em **tempo real** com uma câmera IP via RTSP. Este código utiliza threads para capturar os frames da câmera, aplicar o modelo de detecção de EPI e mostrar as **bounding boxes** ao vivo. Para evitar atrasos, o script usa uma fila `frame_queue = queue.Queue(maxsize=1)` e pula frames quando necessário para manter o display em **tempo real**.

📹 Veja um exemplo de saída do código em `output.mp4`!

## 🌟 Exemplos de Uso

- Treinamento: Use o script `train.py` para treinar um modelo com o dataset PPE.
- Inferência de Imagem: Teste a inferência em uma imagem com `testImage_camIP.py`.
- Inferência em Tempo Real: Rode `cameraIP_RealTime.py` para visualizar a detecção de EPIs em tempo real com sua câmera IP.


## ⚠️ **Em Desenvolvimento**:
- [ ] Adicionar suporte para continuar o treinamento a partir de um **checkpoint**.
- [ ] Implementar o **push** do modelo para o **Hugging Face Hub**.



## 📢 Contribuição

Sinta-se à vontade para abrir issues, sugerir melhorias ou contribuir com este projeto! Vamos melhorar juntos! 😊











