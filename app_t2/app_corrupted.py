import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time
import warnings
try:
    import torch
except ImportError:
    torch = None  # Fallback se torch não estiver disponível

# Configuração da página
st.set_page_config(
    page_title="Vigilância Térmica OTIMIZADA",
    page_icon="",
    layout="wide"
)

# Função para corrigir PyTorch
def fix_torch_load():
    warnings.filterwarnings("ignore")
    import torch
    original_load = torch.loa    if st.button("📹 INICIAR STREAMING OTIMIZADO"):
        # Placeholder para    st.info(f"ℹ️ {total_frames} frames | {fps:.1f} FPS | {duration:.1f}s")
    
           # Fim do vídeo
        st.success("✅ Processamento concluído!")
        st.session_state.current_frame_count = 0  # Reset para próxima execução
        
    cap.release()
    os.unlink(tfile.name)

st.markdown("---")
st.markdown("** Sistema de Vigilância Térmica OTIMIZADO | Performance Máxima para MAVIC 3T**")produção inteligentes
    col_play1, col_play2 = st.columns(2)
    
    with col_play1:
        start_new = st.button("🎬 INICIAR NOVO", help="Inicia do começo")
    
    with col_play2:
        # Só mostra botão continuar se há um frame salvo
        if 'current_frame_count' in st.session_state and st.session_state.current_frame_count > 0:
            continue_video = st.button(f"▶️ CONTINUAR ({st.session_state.current_frame_count}/{total_frames})", help="Continua de onde parou")
        else:
            continue_video = False
    
    # Reset frame count se clicar em iniciar novo
    if start_new:
        st.session_state.current_frame_count = 0
        st.session_state.video_running = True
    
    if start_new or continue_video:ntroles dinâmicos
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            st.markdown("### 🎮 Controles em Tempo Real")
            
        # Container para controles que mudam durante execução
        dynamic_controls = st.container()
        
        progress_bar = st.progress(0)
        metrics_placeholder = st.empty()
        detection_status = st.empty()  # Novo indicador de status
        video_placeholder = st.empty()  # Container para o vídeo
        
        # Configurações do modo atual
        mode_config = st.session_state.processor.surveillance_modes[st.session_state.processor.current_mode]
        skip_frames = mode_config["skip_frames"]
        detection_interval = mode_config["detection_interval"]
        
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        # Flag para controlar execução
        if 'video_running' not in st.session_state:
            st.session_state.video_running = True
        if 'current_frame_count' not in st.session_state:
            st.session_state.current_frame_count = 0
        
        # Posiciona o vídeo no frame correto se estava executando
        if st.session_state.current_frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_count)
            frame_count = st.session_state.current_frame_count
        
        while st.session_state.video_running:
            # Controles dinâmicos durante execução
            with dynamic_controls:
                dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                
                with dcol1:
                    if st.button("⏸️ Pausar", key=f"pause_{frame_count}"):
                        st.session_state.video_running = False
                        st.session_state.current_frame_count = frame_count
                        st.info("⏸️ Vídeo pausado. Clique 'Continuar' para retomar.")
                        break
                
                with dcol2:
                    new_mode = st.selectbox(
                        "Trocar Modo:", 
                        options=list(st.session_state.processor.surveillance_modes.keys()),
                        format_func=lambda x: st.session_state.processor.surveillance_modes[x]['name'],
                        index=list(st.session_state.processor.surveillance_modes.keys()).index(st.session_state.processor.current_mode),
                        key=f"mode_selector_{frame_count//10}"  # Atualiza a cada 10 frames
                    )
                    
                    # Se mudou o modo, aplica instantaneamente
                    if new_mode != st.session_state.processor.current_mode:
                        st.session_state.processor.apply_surveillance_mode(new_mode)
                        mode_config = st.session_state.processor.surveillance_modes[new_mode]
                        detection_interval = mode_config["detection_interval"]
                        st.success(f"✅ Modo alterado para: {mode_config['name']}")
                
                with dcol3:
                    emergency_active = st.button("🚨 EMERGÊNCIA", key=f"emergency_{frame_count//5}")
                    if emergency_active:
                        st.session_state.processor.apply_surveillance_mode("threat_assessment")
                        mode_config = st.session_state.processor.surveillance_modes["threat_assessment"]
                        detection_interval = mode_config["detection_interval"]
                        st.error("🚨 MODO EMERGÊNCIA ATIVO!")
                
                with dcol4:
                    speed_boost = st.checkbox("🚀 Boost", key=f"boost_{frame_count//10}")
                    if speed_boost:
                        detection_interval = min(detection_interval * 2, 15)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            st.session_state.current_frame_count = frame_counted_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    return original_load

# Cache do modelo
@st.cache_resource
def load_model():
    fix_torch_load()
    model_path = "yolov8_large_thermal_15-08-2024.pt"
    if os.path.exists(model_path):
        try:
            model = YOLO(model_path)
            return model, True
        except Exception as e:
            st.error(f"Erro: {str(e)}")
            return None, False
    else:
        st.error(f"Modelo não encontrado: {model_path}")
        return None, False

class OptimizedProcessor:
    def __init__(self):
        self.model, self.model_loaded = load_model()
        
        # Modos de vigilância otimizados
        self.surveillance_modes = {
            "speed": {
                "name": "🚀 VELOCIDADE MÁXIMA",
                "description": "Máxima velocidade para streaming",
                "detection_interval": 10,  # Detecta a cada 10 frames
                "resolution": (320, 240),
                "skip_frames": 1,
                "conf_threshold": 0.4,
                "priority_classes": [0, 1, 2, 3, 4]
            },
            "balanced": {
                "name": "⚖️ BALANCEADO",
                "description": "Equilíbrio entre velocidade e qualidade",
                "detection_interval": 5,  # Detecta a cada 5 frames
                "resolution": (640, 480),
                "skip_frames": 1,
                "conf_threshold": 0.3,
                "priority_classes": [0, 1, 2, 3, 4]
            },
            "quality": {
                "name": "🎯 QUALIDADE MÁXIMA",
                "description": "Máxima qualidade de detecção",
                "detection_interval": 2,  # Detecta a cada 2 frames
                "resolution": (1024, 768),
                "skip_frames": 1,
                "conf_threshold": 0.25,
                "priority_classes": [0, 1, 2, 3, 4]
            },
            "perimeter_security": {
                "name": "🛡️ Segurança Perimetral",
                "description": "Foco em detectar intrusos humanos",
                "detection_interval": 2,  # Mais rápido para segurança
                "resolution": (640, 480),
                "skip_frames": 0,  # Não pula frames
                "conf_threshold": 0.15,  # Mais sensível para detectar pessoas
                "priority_classes": [1],  # Apenas pessoas
                "filters": {"brightness": 25, "contrast": 2.2},
                "tracking_enabled": True,
                "smooth_detections": True
            },
            "vehicle_monitoring": {
                "name": "🚗 Monitoramento Veicular",
                "description": "Detecta veículos suspeitos",
                "detection_interval": 2,  # Detecta a cada 2 frames (mais rápido)
                "resolution": (640, 480),
                "skip_frames": 0,  # Não pula frames
                "conf_threshold": 0.2,  # Menor threshold = mais sensível
                "priority_classes": [0, 2, 3],  # Carros, motos, caminhões
                "filters": {"brightness": 15, "contrast": 1.8},
                "tracking_enabled": True,  # Ativa rastreamento
                "smooth_detections": True  # Suaviza detecções
            },
            "wildlife_detection": {
                "name": "🦌 Detecção de Fauna",
                "description": "Identifica animais próximos à infraestrutura",
                "detection_interval": 3,
                "resolution": (640, 480),
                "skip_frames": 1,
                "conf_threshold": 0.25,
                "priority_classes": [4],  # Apenas animais
                "filters": {"brightness": 30, "contrast": 2.5}
            },
            "pipeline_patrol": {
                "name": "🛢️ Patrulha Geral",
                "description": "Monitoramento balanceado de oleodutos",
                "detection_interval": 3,
                "resolution": (640, 480),
                "skip_frames": 1,
                "conf_threshold": 0.3,
                "priority_classes": [0, 1, 2, 3, 4],  # Todas as classes
                "filters": {"brightness": 20, "contrast": 2.0}
            },
            "threat_assessment": {
                "name": "⚠️ Avaliação de Ameaças",
                "description": "Máxima sensibilidade para emergências",
                "detection_interval": 1,  # Detecta em TODOS os frames
                "resolution": (640, 480),
                "skip_frames": 0,  # Nunca pula frames
                "conf_threshold": 0.1,  # Sensibilidade máxima
                "priority_classes": [0, 1, 2, 3],  # Sem animais
                "filters": {"brightness": 35, "contrast": 2.8},
                "tracking_enabled": True,
                "smooth_detections": False  # Resposta imediata
            }
        }
        
        # Configurações atuais
        self.current_mode = "pipeline_patrol"
        self.last_detections = None
        self.last_frame_id = -1
        self.processing_times = []
        self.frame_count = 0  # Contador de frames
        self.detection_history = []  # Histórico para suavização
        
        # Classes de detecção
        self.class_names = {0: 'Car', 1: 'Person', 2: 'Motorcycle', 3: 'Truck', 4: 'Animal'}
        self.class_colors = {0: (0, 255, 255), 1: (0, 0, 255), 2: (255, 0, 255), 3: (255, 165, 0), 4: (0, 255, 0)}
        
        # Classes ativas (atualizadas pelo modo)
        self.active_classes = [0, 1, 2, 3, 4]  # Todas ativas por padrão
        
        # Filtros de imagem
        self.brightness = 0
        self.contrast = 1.0
        
        # Sistema de rastreamento
        self.tracking_enabled = False
        self.smooth_detections = True
    
    def apply_surveillance_mode(self, mode_key):
        """Aplica configurações do modo de vigilância selecionado"""
        if mode_key not in self.surveillance_modes:
            return
        
        mode = self.surveillance_modes[mode_key]
        self.current_mode = mode_key
        
        # Atualiza classes ativas baseadas na prioridade do modo
        self.active_classes = mode.get('priority_classes', [0, 1, 2, 3, 4])
        
        # Aplica filtros de imagem se disponíveis
        filters = mode.get('filters', {})
        self.brightness = filters.get('brightness', 0)
        self.contrast = filters.get('contrast', 1.0)
        
        # Configura sistema de rastreamento
        self.tracking_enabled = mode.get('tracking_enabled', False)
        self.smooth_detections = mode.get('smooth_detections', True)
        
        # Limpa histórico ao trocar de modo
        self.detection_history = []
        self.frame_count = 0
    
    def optimize_frame(self, frame):
        """Otimiza o frame para processamento rápido"""
        mode_config = self.surveillance_modes[self.current_mode]
        target_resolution = mode_config["resolution"]
        
        # Aplica filtros de brilho/contraste se configurados
        if self.brightness != 0 or self.contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
        
        # Redimensiona se necessário
        h, w = frame.shape[:2]
        if w > target_resolution[0] or h > target_resolution[1]:
            scale = min(target_resolution[0]/w, target_resolution[1]/h)
            new_w, new_h = int(w*scale), int(h*scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    def should_detect_now(self):
        """Determina se deve executar detecção neste frame"""
        mode_config = self.surveillance_modes[self.current_mode]
        detection_interval = mode_config["detection_interval"]
        
        # Incrementa contador de frames
        self.frame_count += 1
        
        # Sempre detecta no primeiro frame
        if self.frame_count == 1:
            return True
        
        # Se não tem detecções recentes, força detecção
        if not self.last_detections:
            return True
        
        # Detecta baseado no intervalo configurado
        return (self.frame_count % detection_interval) == 0
    
    def detect_objects(self, frame, force_detection=False):
        """Detecção otimizada e mais responsiva"""
        if not self.model or not self.model_loaded:
            return frame
        
        mode_config = self.surveillance_modes[self.current_mode]
        
        # Decide se deve detectar neste frame
        should_detect = force_detection or self.should_detect_now()
        
        # Se não deve detectar, usa últimas detecções com suavização
        if not should_detect and self.last_detections:
            return self.apply_cached_detections(frame)
        
        try:
            start_time = time.time()
            
            # Otimiza frame
            optimized_frame = self.optimize_frame(frame)
            
            # Executa detecção YOLO com configurações otimizadas
            results = self.model(
                optimized_frame, 
                conf=mode_config["conf_threshold"], 
                verbose=False,
                imgsz=640,  # Tamanho otimizado para qualidade/velocidade
                half=True,  # Usa precisão reduzida para velocidade
                device='cuda' if torch and torch.cuda.is_available() else 'cpu'
            )
            
            # Mede tempo de processamento
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 5:
                self.processing_times.pop(0)
            
            # Processa detecções
            detections = []
            counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Escala as coordenadas de volta para o frame original
                        scale_x = frame.shape[1] / optimized_frame.shape[1]
                        scale_y = frame.shape[0] / optimized_frame.shape[0]
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                        
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if cls in self.active_classes:
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'conf': conf,
                                'class': cls,
                                'name': self.class_names.get(cls, f'Class_{cls}'),
                                'color': self.class_colors.get(cls, (255, 255, 255))
                            })
                            counts[cls] += 1
            
            # Cache das detecções
            self.last_detections = {'detections': detections, 'counts': counts}
            self.last_frame_id = self.frame_count  # Usa contador de frames
            
            # Adiciona ao histórico para suavização
            if self.smooth_detections:
                self.detection_history.append(detections)
                if len(self.detection_history) > 3:  # Mantém apenas últimas 3 detecções
                    self.detection_history.pop(0)
            
            return self.apply_cached_detections(frame)
            
        except Exception as e:
            st.error(f"Erro na detecção: {str(e)}")
            return frame
    
    def get_smoothed_detections(self):
        """Retorna detecções suavizadas baseadas no histórico"""
        if not self.smooth_detections or not self.detection_history:
            return self.last_detections['detections'] if self.last_detections else []
        
        # Se tem histórico, retorna a detecção mais recente com melhor confiança
        all_detections = []
        for detection_frame in self.detection_history:
            all_detections.extend(detection_frame)
        
        # Remove duplicatas próximas e mantém a de maior confiança
        unique_detections = []
        for det in all_detections:
            is_duplicate = False
            for existing in unique_detections:
                # Verifica se as bounding boxes se sobrepõem significativamente
                overlap = self.bbox_overlap(det['bbox'], existing['bbox'])
                if overlap > 0.5 and det['class'] == existing['class']:
                    is_duplicate = True
                    # Mantém a detecção com maior confiança
                    if det['conf'] > existing['conf']:
                        unique_detections.remove(existing)
                        unique_detections.append(det)
                    break
            
            if not is_duplicate:
                unique_detections.append(det)
        
        return unique_detections
    
    def bbox_overlap(self, bbox1, bbox2):
        """Calcula overlap entre duas bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Área de interseção
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def apply_cached_detections(self, frame):
        """Aplica detecções cached no frame com suavização"""
        if not self.last_detections:
            return frame
        
        result_frame = frame.copy()
        
        # Usa detecções suavizadas se habilitado
        detections = self.get_smoothed_detections()
        
        # Desenha detecções
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            name = det['name']
            color = det['color']
            
            # Desenha retângulo com efeito visual melhorado
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Label com fundo para melhor visibilidade
            label = f'{name}: {conf:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_frame, (x1, y1-20), (x1+label_size[0], y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Estatísticas atualizadas
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for det in detections:
            counts[det['class']] += 1
        y_pos = 20
        stats = []
        if counts[1] > 0: stats.append(f' PESSOAS: {counts[1]}')
        if counts[0] > 0: stats.append(f' Carros: {counts[0]}')
        if counts[3] > 0: stats.append(f' Caminhões: {counts[3]}')
        if counts[2] > 0: stats.append(f' Motos: {counts[2]}')
        if counts[4] > 0: stats.append(f' Animais: {counts[4]}')
        
        for stat in stats:
            cv2.putText(result_frame, stat, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 22
        
        # Info de performance
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            mode_info = f"Modo: {self.surveillance_modes[self.current_mode]['name']} | FPS: {fps:.1f}"
            cv2.putText(result_frame, mode_info, (10, result_frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return result_frame

# Inicialização
if 'processor' not in st.session_state:
    st.session_state.processor = OptimizedProcessor()

# Interface
st.title(" VIGILÂNCIA TÉRMICA - OTIMIZADA PARA STREAMING")
st.markdown("**Sistema Ultra-Rápido para MAVIC 3T | Oleodutos/Gasodutos**")

# Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.header(" Streaming de Vídeo")
    uploaded_file = st.file_uploader("Carregar vídeo térmico", type=['mp4', 'avi', 'mov'])
    video_placeholder = st.empty()

with col2:
    st.header("⚡ Controles Rápidos")
    
    # Status
    if st.session_state.processor.model_loaded:
        st.success("✅ SISTEMA ONLINE")
    else:
        st.error("❌ SISTEMA OFFLINE")
    
    # Seleção de modo de vigilância
    st.subheader("🔧 Modo de Vigilância")
    
    mode_names = [mode['name'] for mode in st.session_state.processor.surveillance_modes.values()]
    mode_keys = list(st.session_state.processor.surveillance_modes.keys())
    
    current_index = mode_keys.index(st.session_state.processor.current_mode)
    selected_index = st.selectbox("Modo de Vigilância:", range(len(mode_names)), 
                                 format_func=lambda x: mode_names[x], index=current_index)
    
    new_mode = mode_keys[selected_index]
    if new_mode != st.session_state.processor.current_mode:
        st.session_state.processor.apply_surveillance_mode(new_mode)
    
    # Info do modo atual
    current_config = st.session_state.processor.surveillance_modes[st.session_state.processor.current_mode]
    st.info(f"ℹ️ {current_config['description']}")
    
    # Configurações específicas do modo
    st.markdown("**Configurações do Modo:**")
    detection_interval = current_config['detection_interval']
    resolution = current_config['resolution']
    priority_classes = current_config.get('priority_classes', [])
    
    st.text(f"🔄 Detecção: 1 a cada {detection_interval} frames")
    st.text(f"📐 Resolução: {resolution[0]}x{resolution[1]}")
    
    # Classes detectadas pelo modo atual
    class_names_display = {0: '🚗 Carros', 1: '👤 Pessoas', 2: '🏍️ Motos', 3: '🚛 Caminhões', 4: '🦌 Animais'}
    active_classes_display = [class_names_display[cls] for cls in priority_classes if cls in class_names_display]
    
    if active_classes_display:
        st.markdown("**Classes Detectadas:**")
        for class_name in active_classes_display:
            st.write(f"✅ {class_name}")
    
    # Controle de performance adicional
    st.markdown("---")
    st.subheader("⚡ Performance")
    
    performance_boost = st.checkbox("🚀 Boost de Velocidade", value=False, help="Reduz qualidade para máxima velocidade")
    if performance_boost:
        # Aumenta ainda mais o intervalo de detecção para máxima velocidade
        current_config['detection_interval'] = min(current_config['detection_interval'] * 2, 15)
        current_config['resolution'] = (320, 240)
        st.warning("⚠️ Qualidade reduzida para máxima velocidade!")
    
    # Métricas
    if st.session_state.processor.processing_times:
        avg_time = np.mean(st.session_state.processor.processing_times)
        fps_est = 1.0 / avg_time if avg_time > 0 else 0
        st.metric("FPS Estimado", f"{fps_est:.1f}")
    
    # Botão de emergência
    st.markdown("---")
    if st.button("🚨 EMERGÊNCIA", type="primary"):
        st.session_state.processor.apply_surveillance_mode("threat_assessment")
        st.error("🚨 MODO EMERGÊNCIA ATIVO!")
        st.rerun()

# Processamento de vídeo
if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.success(f" {uploaded_file.name} ({file_size_mb:.1f} MB)")
    
    # Arquivo temporário
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    st.info(f" {total_frames} frames | {fps:.1f} FPS | {duration:.1f}s")
    
    if st.button(" INICIAR STREAMING OTIMIZADO"):
        progress_bar = st.progress(0)
        metrics_placeholder = st.empty()
        detection_status = st.empty()  # Novo indicador de status
        
        # Configurações do modo atual
        mode_config = st.session_state.processor.surveillance_modes[st.session_state.processor.current_mode]
        skip_frames = mode_config["skip_frames"]
        detection_interval = mode_config["detection_interval"]
        
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Status visual da detecção
            is_detecting = st.session_state.processor.should_detect_now()
            if is_detecting:
                detection_status.success("🔍 **DETECTANDO AGORA**")
            else:
                detection_status.info("⏳ **Usando cache anterior**")
            
            # OTIMIZAÇÃO MÁXIMA: Processa todo frame mas detecta com intervalo
            if frame_count % 1 == 0:  # Processa todo frame mas detecta com intervalo
                # Força detecção baseada no intervalo
                force_detection = (processed_count % detection_interval == 0)
                
                # Processa frame
                processed_frame = st.session_state.processor.detect_objects(frame, force_detection)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Exibe com largura fixa para performance
                video_placeholder.image(frame_rgb, channels="RGB", width=640)
                
                processed_count += 1
                progress_bar.progress(min(1.0, frame_count / total_frames))
                
                # Atualiza métricas a cada 10 frames processados para maior velocidade
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_real = processed_count / elapsed if elapsed > 0 else 0
                    
                    with metrics_placeholder:
                        col_perf1, col_perf2, col_perf3 = st.columns(3)
                        with col_perf1:
                            st.metric("FPS Real", f"{fps_real:.1f}")
                        with col_perf2:
                            st.metric("Frame", f"{frame_count}/{total_frames}")
                        with col_perf3:
                            current_mode_name = st.session_state.processor.surveillance_modes[st.session_state.processor.current_mode]['name']
                            st.metric("Modo Ativo", current_mode_name)
                
                # Pequena pausa para permitir interação
                time.sleep(0.01)
        
        # Fim do vídeo
        st.success("✅ Processamento concluído!")
        st.session_state.current_frame_count = 0  # Reset para próxima execução
        
    cap.release()
                        with col_perf2:
                            st.metric("Processados", processed_count)
                        with col_perf3:
                            if st.session_state.processor.processing_times:
                                avg_proc = np.mean(st.session_state.processor.processing_times)
                                st.metric("Latência", f"{avg_proc*1000:.0f}ms")
        
        cap.release()
        os.unlink(tfile.name)
        st.success(" Streaming concluído com sucesso!")

st.markdown("---")
st.markdown("** Sistema de Vigilância Térmica OTIMIZADO | Performance Máxima para MAVIC 3T**")
