# ObjectDetectionApp

## Descrição

Este é um aplicativo de detecção de objetos que utiliza técnicas de visão computacional e aprendizado de máquina para identificar objetos em imagens. Ele utiliza o Custom Vision que é um recurso Azure para treinar um modelo de detecção e fazer inferências.

## Estrutura do Projeto

```plaintext
ObjectDetectionApp/
├── ObjectDetectionApp.sln
├── README.md
├── .gitignore
├── config/
│ ├── appsettings.json
│ └── camera_settings.json
├── data/
│ ├── raw/
│ ├── processed/
│ └── reports/
├── docs/
│ └── app_docs.md
├── src/
│ ├── ObjectDetectionApp/
│ │ ├── ObjectDetectionApp.csproj
│ │ ├── Program.cs
│ │ ├── Services/
│ │ │ ├── CameraService.cs
│ │ │ ├── DetectionService.cs
│ │ │ ├── AlertService.cs
│ │ │ └── ReportService.cs
│ │ ├── Models/
│ │ │ ├── DetectionResult.cs
│ │ │ └── Person.cs
│ │ ├── Controllers/
│ │ │ └── DetectionController.cs
│ │ └── Utils/
│ │ ├── Logger.cs
│ │ └── Helpers.cs
├── tests/
│ ├── ObjectDetectionApp.Tests/
│ │ ├── ObjectDetectionApp.Tests.csproj
│ │ ├── TestCameraService.cs
│ │ ├── TestDetectionService.cs
│ │ ├── TestAlertService.cs
│ │ ├── TestModelTrainer.cs
│ │ └── TestReportService.cs
└── deployment/
```


## Como Usar

1. Clone este repositório.
2. Navegue até o diretório do projeto ``CTL-Tratamento-Imagens/ObjectDetectionApp/src/ObjectDetectionApp``.
3. Execute `dotnet build` para compilar o projeto.
4. Execute `dotnet run` para iniciar o aplicativo.

> Mais informações em: [Quickstart: Create an object detection project with the Custom Vision client library](https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/quickstarts/object-detection?tabs=windows%2Cvisual-studio&pivots=programming-language-csharp)

## Em Desenvolvimento:
Os seguintes arquivos não foram desenvolvidos. Abaixo, apresentamos mais informações sobre quais são e para que foram destinados:
- **config/**: Diretório para arquivos de configuração.
- `camera_settings.json`: Configurações específicas da câmera IP.
- **docs/**: Diretório para documentação adicional do projeto.
- `api_docs.md`: Documentação da API usada no projeto.
- **Services/**: Módulo para serviços específicos do projeto.
- `CameraService.cs`: Código para conectar e obter imagens da câmera IP.
- `DetectionService.cs`: Código para detectar objetos nas imagens.
- `ReportService.cs`: Código para gerar relatórios semanais.
- **Models/**: Modelos de dados utilizados no projeto.
- `DetectionResult.cs`: Classe representando o resultado da detecção.
- `Person.cs`: Classe representando uma pessoa detectada.
- **Controllers/**: Controladores da API.
- `DetectionController.cs`: Controlador principal para lidar com requisições de detecção.
- **Utils/**: Funções utilitárias gerais.
- `Logger.cs`: Configuração de logs.
- `Helpers.cs`: Funções utilitárias diversas.
- **tests/**: Diretório para testes unitários.
- **MyObjectDetectionApp.Tests/**: Diretório do projeto de testes.
- `TestCameraService.cs`: Testes para o serviço de câmera.
- `TestDetectionService.cs`: Testes para o serviço de detecção.
- `TestReportService.cs`: Testes para o serviço de relatórios.

## Contribuindo

Contribuições são bem-vindas! Por favor, leia as diretrizes de contribuição antes de enviar uma solicitação pull.

