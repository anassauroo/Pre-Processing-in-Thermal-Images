import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time

st.set_page_config(
    page_title="Vigilância Térmica - Oleodutos",
    page_icon="",
    layout="wide"
)

class ThermalVideoProcessor:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
        # Sistema de modos automáticos para vigilância noturna
        self.current_mode = "pipeline_patrol"
        self.auto_mode_active = True
        if st.button("▶️ Iniciar Vigilância"):    # Botão de emergência
    st.markdown("---")
    if st.button("🚨 ALERTA MÁXIMO", type="primary"):
        st.session_state.processor.current_mode = "threat_assessment"
        st.session_state.processor.apply_surveillance_mode("threat_assessment")
        st.error("🚨 MODO DE AMEAÇA ATIVADO!")
        st.rerun()
    
    # Controles de Performance para Streaming
    st.markdown("---")
    st.subheader("⚡ Performance")
    
    performance_mode = st.selectbox(
        "Modo de Streaming:",
        ["🚀 Máximo FPS (baixa precisão)", "⚖️ Balanceado", "🎯 Máxima Precisão (baixo FPS)"],
        index=1
    )
    
    if performance_mode == "🚀 Máximo FPS (baixa precisão)":
        st.session_state.processor.detection_interval = 8
        st.session_state.processor.max_frame_size = (320, 240)
    elif performance_mode == "⚖️ Balanceado":
        st.session_state.processor.detection_interval = 3
        st.session_state.processor.max_frame_size = (640, 480)
    else:  # Máxima Precisão
        st.session_state.processor.detection_interval = 1
        st.session_state.processor.max_frame_size = (1024, 768)
    
    # Mostra FPS estimado se disponível
    if hasattr(st.session_state.processor, 'processing_time_history') and st.session_state.processor.processing_time_history:
        avg_time = np.mean(st.session_state.processor.processing_time_history[-5:])
        fps_estimate = 1.0 / avg_time if avg_time > 0 else 0
        st.metric("FPS Estimado", f"{fps_estimate:.1f}")
    
    st.info(f"🔄 Detectando a cada {st.session_state.processor.detection_interval} frames")  progress_bar = st.progress(0)
        frame_count = 0
        
        # Otimizações de performance para streaming
        st.session_state.processor.streaming_mode = True
        
        # Controles de performance dinâmicos
        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            detection_interval = st.selectbox("Intervalo de Detecção:", 
                                            [1, 2, 3, 5, 8], 
                                            index=2, 
                                            help="1=Máxima precisão (lento), 8=Máxima velocidade")
            st.session_state.processor.detection_interval = detection_interval
        
        with col_perf2:
            stream_quality = st.selectbox("Qualidade:", 
                                        ["Alta (lento)", "Média (balanceado)", "Baixa (rápido)"], 
                                        index=1)
            
        # Configura qualidade baseada na seleção
        if stream_quality == "Alta (lento)":
            st.session_state.processor.max_frame_size = (1024, 768)
            skip_frames = 2
        elif stream_quality == "Média (balanceado)":
            st.session_state.processor.max_frame_size = (640, 480)
            skip_frames = 1
        else:  # Baixa (rápido)
            st.session_state.processor.max_frame_size = (480, 360)
            skip_frames = 1
        
        # Container para métricas de performance
        metrics_container = st.container()
        
        start_time = time.time()
        frames_processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % skip_frames == 0:
                # Processa com sistema de vigilância inteligente otimizado
                processed_frame = st.session_state.processor.process_frame_realtime(frame, frame_count)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Otimização: usa width fixo para melhor performance
                video_placeholder.image(frame_rgb, channels="RGB", width=640)
                
                frames_processed += 1
                progress_bar.progress(min(1.0, frame_count / total_frames))
                
                # Calcula e exibe métricas de performance a cada 30 frames
                if frames_processed % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps_real = frames_processed / elapsed_time if elapsed_time > 0 else 0
                    
                    with metrics_container:
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("FPS Real", f"{fps_real:.1f}")
                        with col_m2:
                            st.metric("Frames Processados", frames_processed)
                        with col_m3:
                            avg_proc_time = np.mean(st.session_state.processor.processing_time_history[-5:]) if st.session_state.processor.processing_time_history else 0
                            st.metric("Tempo/Frame", f"{avg_proc_time*1000:.0f}ms")
                
                # Remove delay para streaming em tempo real
                # time.sleep(0.02)  # Comentado para máxima velocidade= 0
        self.contrast = 1.0
        self.use_clahe = False
        self.clahe_limit = 2.0
        self.clahe_grid = 8
        
        self.detect_persons = False
        self.detect_cars = False
        self.detect_motorcycles = False
        self.detect_trucks = False
        self.detect_animals = False
        
        # Parâmetros de guided filter
        self.use_guided_filter = False
        self.guided_radius = 8
        self.guided_eps = 0.01
        
        # Sistema de análise automática de contexto
        self.frame_analysis_history = []
        self.detection_performance = []
        self.auto_adjustment_enabled = True
        self.last_detections = []
        self.detection_confidence_history = []
        
        # Otimizações de performance para streaming
        self.frame_cache = {}
        self.last_processed_frame_id = -1
        self.detection_interval = 3  # Detecta a cada 3 frames
        self.last_detection_results = None
        self.processing_time_history = []
        
        # Configurações de streaming otimizadas
        self.streaming_mode = True
        self.max_frame_size = (640, 480)  # Reduz resolução para speed
        self.jpeg_quality = 70  # Compressão otimizada
        
        # Modos pré-configurados para vigilância noturna de oleodutos/gasodutos
        self.surveillance_modes = {
            "auto_optimal": {
                "name": " Automático Inteligente",
                "description": "Ajusta automaticamente baseado no contexto detectado",
                "auto_adjust": True,
                "priority_classes": ["Person", "Car", "Truck", "Motorcycle", "Animal"],
                "alert_level": "medium"
            },
            "perimeter_security": {
                "name": " Segurança Perimetral",
                "description": "Otimizado para detectar intrusos humanos próximos aos dutos",
                "brightness": 25,
                "contrast": 2.2,
                "clahe": True,
                "clahe_limit": 4.5,
                "clahe_grid": 8,
                "guided_filter": True,
                "guided_radius": 6,
                "guided_eps": 0.008,
                "priority_classes": ["Person"],
                "sensitivity": "high",
                "alert_level": "high"
            },
            "vehicle_monitoring": {
                "name": " Monitoramento Veicular",
                "description": "Detecta veículos suspeitos próximos à infraestrutura",
                "brightness": 15,
                "contrast": 1.8,
                "clahe": True,
                "clahe_limit": 3.5,
                "clahe_grid": 10,
                "guided_filter": False,
                "priority_classes": ["Car", "Truck", "Motorcycle"],
                "sensitivity": "medium",
                "alert_level": "medium"
            },
            "wildlife_detection": {
                "name": " Detecção de Fauna",
                "description": "Identifica animais que podem danificar a infraestrutura",
                "brightness": 30,
                "contrast": 2.5,
                "clahe": True,
                "clahe_limit": 5.0,
                "clahe_grid": 6,
                "guided_filter": True,
                "guided_radius": 8,
                "guided_eps": 0.005,
                "priority_classes": ["Animal"],
                "sensitivity": "maximum",
                "alert_level": "low"
            },
            "pipeline_patrol": {
                "name": " Patrulha Geral",
                "description": "Monitoramento geral balanceado de oleodutos/gasodutos",
                "brightness": 20,
                "contrast": 2.0,
                "clahe": True,
                "clahe_limit": 4.0,
                "clahe_grid": 8,
                "guided_filter": True,
                "guided_radius": 7,
                "guided_eps": 0.006,
                "priority_classes": ["Person", "Car", "Truck", "Motorcycle", "Animal"],
                "sensitivity": "high",
                "alert_level": "medium"
            },
            "threat_assessment": {
                "name": " Avaliação de Ameaças",
                "description": "Máxima sensibilidade para situações de risco",
                "brightness": 35,
                "contrast": 2.8,
                "clahe": True,
                "clahe_limit": 6.0,
                "clahe_grid": 6,
                "guided_filter": True,
                "guided_radius": 5,
                "guided_eps": 0.003,
                "priority_classes": ["Person", "Car", "Truck", "Motorcycle"],
                "sensitivity": "maximum",
                "alert_level": "critical"
            }
        }
        
        # Armazenar frame original e processado
        self.original_frame = None
        self.last_filter_params = None
        
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=(self.clahe_grid, self.clahe_grid))
        self.auto_load_model()
        
    def auto_load_model(self):
        model_path = "yolov8_large_thermal_15-08-2024.pt"
        if os.path.exists(model_path):
            try:
                import torch
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = patched_load
                
                self.model = YOLO(model_path)
                torch.load = original_load
                self.model_loaded = True
                print(f" Modelo carregado: {model_path}")
                
            except Exception as e:
                print(f" Erro: {str(e)}")
                self.model_loaded = False
        else:
            print(f" Arquivo não encontrado: {model_path}")
            self.model_loaded = False
    
    def analyze_frame_context(self, frame):
        """Analisa o contexto do frame para sugerir modo automático"""
        mean_intensity = np.mean(frame)
        contrast_ratio = np.std(frame)
        
        # Análise específica para imagens térmicas noturnas
        if mean_intensity < 40:  # Muito escuro
            if contrast_ratio < 25:  # Baixo contraste
                return "wildlife_detection"  # Pode ser animal distante
            else:
                return "threat_assessment"  # Contraste alto, possível ameaça
        elif mean_intensity < 80:  # Escuro normal
            return "pipeline_patrol"  # Patrulha normal
        elif mean_intensity < 120:  # Iluminação moderada
            return "vehicle_monitoring"  # Possível veículo
        else:  # Objeto quente próximo
            return "perimeter_security"  # Possível intruso
    
    def apply_surveillance_mode(self, mode_key):
        """Aplica configurações do modo de vigilância selecionado"""
        if mode_key not in self.surveillance_modes:
            return
        
        mode = self.surveillance_modes[mode_key]
        
        if mode_key == "auto_optimal":
            # Modo automático não aplica configurações fixas
            return
        
        # Aplica configurações do modo
        self.brightness = mode.get('brightness', 0)
        self.contrast = mode.get('contrast', 1.0)
        self.use_clahe = mode.get('clahe', False)
        self.clahe_limit = mode.get('clahe_limit', 2.0)
        self.clahe_grid = mode.get('clahe_grid', 8)
        self.use_guided_filter = mode.get('guided_filter', False)
        self.guided_radius = mode.get('guided_radius', 8)
        self.guided_eps = mode.get('guided_eps', 0.01)
        
        # Atualiza configurações de detecção baseadas na prioridade
        priority_classes = mode.get('priority_classes', [])
        self.detect_persons = "Person" in priority_classes
        self.detect_cars = "Car" in priority_classes
        self.detect_motorcycles = "Motorcycle" in priority_classes
        self.detect_trucks = "Truck" in priority_classes
        self.detect_animals = "Animal" in priority_classes
        
        # Recria CLAHE com novos parâmetros
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=(self.clahe_grid, self.clahe_grid))
    
    def apply_filters(self, image):
        """Aplica todos os filtros na imagem com otimizações de performance"""
        filtered = image.copy()
        
        # Reduz resolução para processamento mais rápido se em modo streaming
        if self.streaming_mode and (image.shape[1] > self.max_frame_size[0] or image.shape[0] > self.max_frame_size[1]):
            scale_factor = min(self.max_frame_size[0] / image.shape[1], self.max_frame_size[1] / image.shape[0])
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            filtered = cv2.resize(filtered, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Aplica ajustes de brilho e contraste primeiro
        if self.brightness != 0 or self.contrast != 1.0:
            filtered = cv2.convertScaleAbs(filtered, alpha=self.contrast, beta=self.brightness)
        
        # Aplica CLAHE se habilitado (otimizado)
        if self.use_clahe:
            if len(filtered.shape) == 3:
                # Converte para LAB apenas se necessário
                lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = self.clahe.apply(lab[:,:,0])
                filtered = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                filtered = self.clahe.apply(filtered)
        
        # Aplica guided filter se habilitado (versão mais rápida)
        if self.use_guided_filter:
            try:
                # Usa versão mais rápida do bilateral filter
                d = min(self.guided_radius, 9)  # Limita para performance
                filtered = cv2.bilateralFilter(filtered, d, 
                                             self.guided_eps * 50, self.guided_eps * 50)
            except:
                pass  # Se falhar, continua sem o filtro
        
        return filtered
    
    def detect_objects_with_surveillance_mode(self, image):
        """Detecta objetos aplicando o modo de vigilância ativo"""
        if not self.model or not self.model_loaded:
            return self.apply_filters(image)
        
        # Aplica modo automático se ativo
        if self.current_mode == "auto_optimal" and self.auto_adjustment_enabled:
            suggested_mode = self.analyze_frame_context(image)
            self.apply_surveillance_mode(suggested_mode)
        
        any_detection_active = (self.detect_persons or self.detect_cars or 
                              self.detect_motorcycles or self.detect_trucks or 
                              self.detect_animals)
        
        if not any_detection_active:
            return self.apply_filters(image)
        
        try:
            # Aplica filtros ANTES da detecção para melhorar a precisão
            filtered_image = self.apply_filters(image)
            
            # Executa detecção na imagem filtrada
            results = self.model(filtered_image, conf=0.25, verbose=False)
            
            person_count = 0
            car_count = 0
            truck_count = 0
            motorcycle_count = 0
            animal_count = 0
            
            class_names = {
                0: 'Car',
                1: 'Person', 
                2: 'Motorcycle',
                3: 'Truck',
                4: 'Animal'
            }
            
            # Cores específicas para vigilância noturna
            class_colors = {
                0: (0, 255, 255),     # Amarelo para carros (alta visibilidade)
                1: (0, 0, 255),       # Vermelho para pessoas (máxima prioridade)
                2: (255, 0, 255),     # Magenta para motos
                3: (255, 165, 0),     # Laranja para caminhões
                4: (0, 255, 0)        # Verde para animais
            }
            
            # Desenha detecções na imagem filtrada
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        class_name = class_names.get(cls, f'Class_{cls}')
                        color = class_colors.get(cls, (0, 255, 0))
                        
                        # Contadores por classe
                        if cls == 1:
                            person_count += 1
                        elif cls == 0:
                            car_count += 1
                        elif cls == 3:
                            truck_count += 1
                        elif cls == 2:
                            motorcycle_count += 1
                        elif cls == 4:
                            animal_count += 1
                        
                        should_draw = False
                        if cls == 0 and self.detect_cars:
                            should_draw = True
                        elif cls == 1 and self.detect_persons:
                            should_draw = True
                        elif cls == 2 and self.detect_motorcycles:
                            should_draw = True
                        elif cls == 3 and self.detect_trucks:
                            should_draw = True
                        elif cls == 4 and self.detect_animals:
                            should_draw = True
                        
                        if should_draw:
                            # Desenha retângulo com espessura maior para melhor visibilidade
                            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), color, 3)
                            
                            # Label com fundo para melhor legibilidade
                            label = f'{class_name}: {conf:.2f}'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(filtered_image, (x1, y1-label_size[1]-10), 
                                        (x1+label_size[0], y1), color, -1)
                            cv2.putText(filtered_image, label, (x1, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Estatísticas com priorização para vigilância
            stats_lines = []
            if person_count > 0:
                stats_lines.append(f' PESSOAS: {person_count}')
            if car_count > 0:
                stats_lines.append(f' Carros: {car_count}')
            if truck_count > 0:
                stats_lines.append(f' Caminhões: {truck_count}')
            if motorcycle_count > 0:
                stats_lines.append(f' Motos: {motorcycle_count}')
            if animal_count > 0:
                stats_lines.append(f' Animais: {animal_count}')
            
            # Exibe estatísticas
            for i, stats_text in enumerate(stats_lines):
                y_pos = 30 + (i * 25)
                cv2.putText(filtered_image, stats_text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostra modo ativo no canto superior direito
            mode_text = f"Modo: {self.surveillance_modes[self.current_mode]['name']}"
            text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(filtered_image, mode_text, 
                       (filtered_image.shape[1] - text_size[0] - 10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            return filtered_image
            
        except Exception as e:
            st.error(f"Erro na detecção: {str(e)}")
            return self.apply_filters(image)
    
    def process_frame_realtime(self, image, frame_number=0):
        """Processa frame com detecção em tempo real otimizada para streaming"""
        # Armazena frame original se necessário
        if self.original_frame is None:
            self.original_frame = image.copy()
        
        # Otimização: só executa detecção a cada N frames
        force_detection = (frame_number % self.detection_interval == 0)
        
        # Processa com o modo de vigilância ativo
        return self.detect_objects_with_surveillance_mode(image, force_detection)

# Inicialização
if 'processor' not in st.session_state:
    st.session_state.processor = ThermalVideoProcessor()

# Interface principal otimizada para vigilância
st.title(" Vigilância Térmica - Oleodutos/Gasodutos")
st.markdown("**Sistema de Detecção Automatizado para MAVIC 3T - Operações Noturnas**")

# Layout principal
col1, col2 = st.columns([3, 1])

with col1:
    st.header(" Monitoramento Térmico")
    uploaded_file = st.file_uploader("Carregar vídeo térmico do drone", type=['mp4', 'avi', 'mov', 'mkv'])
    video_placeholder = st.empty()

with col2:
    st.header(" Controle de Vigilância")
    
    # Status do modelo
    if st.session_state.processor.model_loaded:
        st.success(" Sistema: Operacional")
    else:
        st.error(" Sistema: Erro no Modelo")
    
    # Seleção de modo de vigilância
    st.subheader(" Modo de Operação")
    
    mode_options = []
    mode_keys = []
    for key, mode in st.session_state.processor.surveillance_modes.items():
        mode_options.append(f"{mode['name']}")
        mode_keys.append(key)
    
    selected_mode_index = st.selectbox(
        "Selecione o modo:",
        range(len(mode_options)),
        format_func=lambda x: mode_options[x],
        index=mode_keys.index(st.session_state.processor.current_mode)
    )
    
    selected_mode = mode_keys[selected_mode_index]
    
    if selected_mode != st.session_state.processor.current_mode:
        st.session_state.processor.current_mode = selected_mode
        st.session_state.processor.apply_surveillance_mode(selected_mode)
        st.rerun()
    
    # Mostra descrição do modo selecionado
    current_mode_info = st.session_state.processor.surveillance_modes[selected_mode]
    st.info(f"ℹ {current_mode_info['description']}")
    
    # Status das detecções ativas
    st.markdown("---")
    st.subheader(" Classes Detectadas")
    
    detection_status = []
    if st.session_state.processor.detect_persons:
        detection_status.append(" Pessoas")
    if st.session_state.processor.detect_cars:
        detection_status.append(" Carros")
    if st.session_state.processor.detect_trucks:
        detection_status.append(" Caminhões")
    if st.session_state.processor.detect_motorcycles:
        detection_status.append(" Motocicletas")
    if st.session_state.processor.detect_animals:
        detection_status.append(" Animais")
    
    if detection_status:
        for status in detection_status:
            st.success(f" {status}")
    else:
        st.warning(" Nenhuma classe ativa")
    
    # Botão de emergência
    st.markdown("---")
    if st.button(" ALERTA MÁXIMO", type="primary"):
        st.session_state.processor.current_mode = "threat_assessment"
        st.session_state.processor.apply_surveillance_mode("threat_assessment")
        st.error(" MODO DE AMEAÇA ATIVADO!")
        st.rerun()

# Processamento de vídeo
if uploaded_file is not None:
    # Mostra informações do arquivo
    file_size = uploaded_file.size
    file_size_mb = file_size / (1024 * 1024)
    st.success(f" Arquivo: {uploaded_file.name} ({file_size_mb:.1f} MB)")
    
    if file_size_mb > 1000:
        st.warning(" Arquivo grande - Processamento pode ser lento")
    
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    st.info(f" Vídeo: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
    
    if st.button(" Iniciar Vigilância"):
        progress_bar = st.progress(0)
        frame_count = 0
        
        # Otimização baseada no tamanho do arquivo
        skip_frames = 3 if file_size_mb < 200 else 5 if file_size_mb < 500 else 8
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % skip_frames == 0:
                # Processa com sistema de vigilância inteligente
                processed_frame = st.session_state.processor.process_frame_realtime(frame)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                progress_bar.progress(min(1.0, frame_count / total_frames))
                time.sleep(0.02)
        
        cap.release()
        os.unlink(tfile.name)
        st.success(" Vigilância concluída!")

# Rodapé com informações do sistema
st.markdown("---")
st.markdown("** Sistema de Vigilância Térmica para Oleodutos/Gasodutos | Otimizado para MAVIC 3T**")
