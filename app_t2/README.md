# Aplicativo de Detecção Térmica com YOLO

Este aplicativo permite processar vídeos de imagens térmicas com detecção de pessoas usando YOLO e controles de ajuste de imagem em tempo real.

## 🚀 Características

- **Detecção YOLO**: Usa o modelo treinado `yolov8_large_thermal_15-08-2024.pt` para detectar pessoas
- **Controle de Brilho**: Ajuste em tempo real (-100 a +100)
- **Controle de Contraste**: Ajuste em tempo real (0.1x a 3.0x)
- **Ajuste Automático**: Detecta automaticamente os melhores níveis de brilho/contraste
- **Interface Web**: Interface intuitiva com Streamlit
- **Processamento CLI**: Versão linha de comando para processamento batch

## 📦 Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Certifique-se de que o modelo YOLO está na pasta raiz:
   - `yolov8_large_thermal_15-08-2024.pt`

## 🖥️ Interface Web (Recomendado)

Execute o aplicativo web interativo:

```bash
streamlit run app.py
```

### Funcionalidades da Interface:

1. **Upload de Vídeo**: Arraste e solte ou selecione um vídeo térmico
2. **Botão YOLO**: Ativa/desativa a detecção de pessoas
3. **Controle de Brilho**: Slider para ajustar o brilho (-100 a +100)
4. **Controle de Contraste**: Slider para ajustar o contraste (0.1x a 3.0x)
5. **Botão AUTO**: Aplica ajuste automático para otimizar a detecção
6. **Reprodução**: Controles para reproduzir e parar o vídeo

## ⌨️ Versão Linha de Comando

Para processamento batch ou automático:

```bash
python video_processor.py input_video.mp4 [opções]
```

### Opções disponíveis:

- `--output, -o`: Caminho para salvar o vídeo processado
- `--model, -m`: Caminho para o modelo YOLO (padrão: yolov8_large_thermal_15-08-2024.pt)
- `--auto, -a`: Aplica ajuste automático no primeiro frame
- `--brightness, -b`: Brilho inicial (-100 a 100)
- `--contrast, -c`: Contraste inicial (0.1 a 3.0)

### Controles durante a reprodução:
- `q`: Sair
- `a`: Aplicar ajuste automático
- `+/=`: Aumentar brilho
- `-`: Diminuir brilho
- `1`: Aumentar contraste
- `2`: Diminuir contraste

### Exemplos:

```bash
# Processamento básico
python video_processor.py video_termico.mp4

# Com ajuste automático e salvamento
python video_processor.py video_termico.mp4 --auto --output resultado.mp4

# Com ajustes iniciais personalizados
python video_processor.py video_termico.mp4 --brightness 30 --contrast 1.2
```

## 🛠️ Como Funciona

### Detecção YOLO
- Utiliza o modelo treinado especificamente para imagens térmicas
- Detecta pessoas com confiança mínima de 25%
- Desenha caixas delimitadoras verdes ao redor das detecções
- Mostra a confiança da detecção para cada pessoa

### Ajustes de Imagem
- **Brilho**: Adiciona/subtrai valores de pixel uniformemente
- **Contraste**: Multiplica os valores de pixel por um fator
- **Ajuste Automático**: Analisa a distribuição de intensidade da imagem para determinar os melhores ajustes

### Algoritmo de Ajuste Automático
1. Calcula a média e desvio padrão da imagem
2. Para imagens escuras (média < 100): Aumenta brilho e contraste
3. Para imagens claras (média > 180): Diminui brilho, ajusta contraste
4. Para baixo contraste (desvio < 30): Aumenta contraste
5. Otimizado para melhorar a detecção de pessoas em imagens térmicas

## 📁 Estrutura do Projeto

```
app_Teste1_imp/
├── app.py                                    # Interface web Streamlit
├── video_processor.py                        # Processador CLI
├── requirements.txt                          # Dependências
├── README.md                                # Este arquivo
├── yolov8_large_thermal_15-08-2024.pt      # Modelo YOLO treinado
└── yolov8n.pt                              # Modelo YOLO padrão
```

## 🎯 Formatos Suportados

- **Vídeo**: MP4, AVI, MOV, MKV
- **Modelo**: PyTorch (.pt)

## 🔧 Troubleshooting

### Modelo não carrega
- Verifique se o arquivo `yolov8_large_thermal_15-08-2024.pt` existe na pasta
- Confirme que o PyTorch está instalado corretamente

### Vídeo não reproduz
- Verifique se o formato do vídeo é suportado
- Teste com um vídeo menor primeiro
- Verifique se o OpenCV está instalado corretamente

### Performance lenta
- Reduza a resolução do vídeo
- Use GPU se disponível (PyTorch com CUDA)
- Processe o vídeo em lote usando a versão CLI

## 📊 Exemplo de Uso

1. Abra o aplicativo web: `streamlit run app.py`
2. Carregue o modelo YOLO clicando em "Carregar Modelo YOLO"
3. Faça upload de um vídeo térmico
4. Ative a detecção com o checkbox "YOLO - Detectar Pessoas"
5. Use "AUTO - Ajuste Automático" para otimizar a imagem
6. Ajuste manualmente brilho e contraste conforme necessário
7. Clique em "Reproduzir Vídeo" para ver o resultado

## 🎥 Demonstração

O aplicativo foi projetado como um protótipo para demonstrar:
- Detecção automática de pessoas em vídeos térmicos
- Melhoria da qualidade de imagem para otimizar detecções
- Interface intuitiva para controle em tempo real
- Processamento eficiente de vídeo

## ⚡ Performance

- Otimizado para vídeos térmicos
- Suporte a GPU para detecção YOLO
- Processamento frame-by-frame eficiente
- Interface responsiva com feedback em tempo real