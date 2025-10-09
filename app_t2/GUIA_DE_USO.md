# 🚀 GUIA DE USO - Aplicativo de Detecção Térmica

## 📋 Resumo do Projeto

Você agora tem um **aplicativo completo** para processamento de vídeos térmicos com detecção de pessoas usando YOLO. O aplicativo inclui:

### ✅ Funcionalidades Implementadas:

1. **🎯 BOTÃO YOLO**: Ativa/desativa a detecção de pessoas usando o modelo `yolov8_large_thermal_15-08-2024.pt`
2. **🌓 BOTÃO CONTRASTE**: Slider para ajustar contraste (0.1x até 3.0x) em tempo real
3. **☀️ BOTÃO BRILHO**: Slider para ajustar brilho (-100 até +100) em tempo real  
4. **🤖 BOTÃO AUTO**: Função automática que analisa a imagem e otimiza brilho/contraste para melhor detecção
5. **📹 Interface Web**: Interface intuitiva com Streamlit
6. **⌨️ Versão CLI**: Para processamento batch via linha de comando

---

## 🖥️ COMO USAR O APLICATIVO

### Método 1: Interface Web (RECOMENDADO)

1. **Inicie o aplicativo:**
   ```bash
   streamlit run app.py
   ```
   OU execute o arquivo: `run_app.bat`

2. **Acesse no navegador:** `http://localhost:8501`

3. **Passos para usar:**
   - ✅ Clique em "Carregar Modelo YOLO" (carrega o modelo treinado)
   - 📁 Faça upload de um vídeo térmico
   - 🎯 Ative "YOLO - Detectar Pessoas" para ver as detecções
   - 🤖 Use "AUTO - Ajuste Automático" para otimizar a imagem
   - 🎛️ Ajuste Brilho e Contraste manualmente conforme necessário
   - ▶️ Clique "Reproduzir Vídeo" para processar

### Método 2: Linha de Comando (Para processamento batch)

```bash
# Exemplo básico
python video_processor.py meu_video_termico.mp4

# Com todas as opções
python video_processor.py video.mp4 --output resultado.mp4 --auto --brightness 30 --contrast 1.2
```

**Controles durante reprodução CLI:**
- `q`: Sair
- `a`: Ajuste automático
- `+/-`: Ajustar brilho
- `1/2`: Ajustar contraste

---

## 🎬 CRIAR VÍDEO DE TESTE

Se você não tem um vídeo térmico, pode gerar um para demonstração:

```bash
streamlit run demo_generator.py
```

Isso criará um vídeo térmico simulado com pessoas em movimento que você pode usar para testar o aplicativo.

---

## 📁 ESTRUTURA DOS ARQUIVOS

```
app_Teste1_imp/
├── app.py                              # 🖥️ Interface web principal
├── video_processor.py                  # ⌨️ Processador CLI
├── demo_generator.py                   # 🎬 Gera vídeos de teste
├── run_app.bat                        # 🚀 Inicializar app (Windows)
├── test_app.py                        # 🧪 Testes automatizados
├── requirements.txt                   # 📦 Dependências
├── README.md                         # 📖 Documentação detalhada
└── yolov8_large_thermal_15-08-2024.pt # 🧠 Seu modelo YOLO treinado
```

---

## 🛠️ COMO FUNCIONA CADA FUNCIONALIDADE

### 🎯 Detecção YOLO
- Usa seu modelo treinado especificamente para imagens térmicas
- Detecta pessoas com confiança mínima de 25%
- Desenha caixas verdes ao redor das detecções
- Mostra percentual de confiança

### 🌓 Controle de Contraste
- Multiplica os valores dos pixels por um fator
- Valores > 1.0: Aumenta contraste
- Valores < 1.0: Diminui contraste
- Útil para destacar pessoas do fundo

### ☀️ Controle de Brilho
- Adiciona/subtrai valores uniformemente
- Valores positivos: Imagem mais clara
- Valores negativos: Imagem mais escura
- Útil para compensar condições de iluminação

### 🤖 Ajuste Automático
O algoritmo analisa a imagem e aplica os melhores ajustes:

1. **Imagem Escura** (média < 100): ↑ Brilho + ↑ Contraste
2. **Imagem Clara** (média > 180): ↓ Brilho + Ajusta Contraste  
3. **Baixo Contraste** (desvio < 30): ↑ Contraste
4. **Otimizado** para melhorar detecção de pessoas

---

## 🔧 TROUBLESHOOTING

### ❌ "Modelo não encontrado"
- Verifique se `yolov8_large_thermal_15-08-2024.pt` está na pasta raiz
- O arquivo deve ter exatamente esse nome

### ❌ "Erro ao carregar vídeo"
- Formatos suportados: MP4, AVI, MOV, MKV
- Teste com um vídeo menor primeiro
- Use o gerador de demo para criar um vídeo de teste

### 🐌 Performance lenta
- Reduza a resolução do vídeo
- Use a versão CLI para processamento batch
- Certifique-se de ter GPU disponível para YOLO

---

## 🎯 EXEMPLO DE USO COMPLETO

1. **Inicie:** `streamlit run app.py`
2. **Carregue modelo:** Clique "Carregar Modelo YOLO"
3. **Upload vídeo:** Arraste um arquivo de vídeo térmico
4. **Configure detecção:** ✅ Marque "YOLO - Detectar Pessoas"
5. **Otimize imagem:** Clique "AUTO - Ajuste Automático"
6. **Ajuste fino:** Use sliders de Brilho/Contraste se necessário
7. **Processe:** Clique "Reproduzir Vídeo"
8. **Observe:** Veja as caixas verdes ao redor das pessoas detectadas

---

## 📊 RESULTADOS ESPERADOS

- ✅ Detecção automática de pessoas em vídeos térmicos
- ✅ Melhoria da qualidade visual em tempo real
- ✅ Interface intuitiva com controles ao vivo
- ✅ Processamento eficiente frame-by-frame
- ✅ Funcionalidade de otimização automática

---

## 🚀 PRÓXIMOS PASSOS

Este protótipo demonstra todas as funcionalidades solicitadas. Você pode:

1. **Testar** com seus próprios vídeos térmicos
2. **Ajustar** os parâmetros de detecção no código se necessário
3. **Expandir** adicionando mais funcionalidades
4. **Otimizar** para performance em produção
5. **Integrar** com sistemas de monitoramento existentes

---

**🎉 Seu aplicativo está pronto para uso!** 

Comece executando `streamlit run app.py` e explore todas as funcionalidades.