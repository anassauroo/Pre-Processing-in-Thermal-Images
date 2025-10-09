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
    torch = None

# Configuração da página
st.set_page_config(
    page_title="Vigilância Térmica OTIMIZADA",
    page_icon="🔥",
    layout="wide"
)

# Função para corrigir PyTorch
def fix_torch_load():
    warnings.filterwarnings("ignore")
    if torch:
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.pop('weights_only', None)
            return original_load(*args, **kwargs)
        torch.load = patched_load

fix_torch_load()

class OptimizedProcessor:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.load_model()
        
        # Modos de vigilância otimizados
        self.surveillance_modes = {
            "speed": {
                "name": "🚀 VELOCIDADE MÁXIMA",
                "description": "Máxima velocidade para streaming",
                "detection_interval": 10,
                "resolution": (320, 240),
                "skip_frames": 1,
                "conf_threshold": 0.4,
                "priority_classes": [0, 1, 2, 3, 4]
            },
            "balanced": {
                "name": "⚖️ BALANCEADO",
                "description": "Equilíbrio entre velocidade e qualidade",
                "detection_interval": 5,
                "resolution": (640, 480),
                "skip_frames": 1,
                "conf_threshold": 0.3,
                "priority_classes": [0, 1, 2, 3, 4]
            },
            "quality": {
                "name": "🎯 QUALIDADE MÁXIMA",
                "description": "Máxima qualidade de detecção",
                "detection_interval": 2,
                "resolution": (1024, 768),
                "skip_frames": 1,
                "conf_threshold": 0.25,
                "priority_classes": [0, 1, 2, 3, 4]
            },
            "perimeter_security": {
                "name": "🛡️ Segurança Perimetral",
                "description": "Foco em detectar intrusos humanos",
                "detection_interval": 2,
                "resolution": (640, 480),
                "skip_frames": 0,
                "conf_threshold": 0.15,
                "priority_classes": [1],
                "filters": {"brightness": 25, "contrast": 2.2},
                "tracking_enabled": True,
                "smooth_detections": True
            },
            "vehicle_monitoring": {
                "name": "🚗 Monitoramento Veicular",
                "description": "Detecta veículos suspeitos",
                "detection_interval": 2,
                "resolution": (640, 480),
                "skip_frames": 0,
                "conf_threshold": 0.2,
                "priority_classes": [0, 2, 3],
                "filters": {"brightness": 15, "contrast": 1.8},
                "tracking_enabled": True,
                "smooth_detections": True
            },
            "wildlife_detection": {
                "name": "🦌 Detecção de Fauna",
                "description": "Identifica animais próximos à infraestrutura",
                "detection_interval": 3,
                "resolution": (640, 480),
                "skip_frames": 1,
                "conf_threshold": 0.25,
                "priority_classes": [4],
                "filters": {"brightness": 30, "contrast": 2.5}
            },
            "pipeline_patrol": {
                "name": "🛢️ Patrulha Geral",
                "description": "Monitoramento balanceado de oleodutos",
                "detection_interval": 3,
                "resolution": (640, 480),
                "skip_frames": 1,
                "conf_threshold": 0.3,
                "priority_classes": [0, 1, 2, 3, 4],
                "filters": {"brightness": 20, "contrast": 2.0}
            },
            "threat_assessment": {
                "name": "⚠️ Avaliação de Ameaças",
                "description": "Máxima sensibilidade para emergências",
                "detection_interval": 1,
                "resolution": (640, 480),
                "skip_frames": 0,
                "conf_threshold": 0.1,
                "priority_classes": [0, 1, 2, 3],
                "filters": {"brightness": 35, "contrast": 2.8},
                "tracking_enabled": True,
                "smooth_detections": False
            }
        }
        
        # Configurações atuais
        self.current_mode = "pipeline_patrol"
        self.last_detections = None
        self.last_frame_id = -1
        self.processing_times = []
        self.frame_count = 0
        self.detection_history = []
        
        # Classes de detecção
        self.class_names = {0: 'Car', 1: 'Person', 2: 'Motorcycle', 3: 'Truck', 4: 'Animal'}
        self.class_colors = {0: (0, 255, 255), 1: (0, 0, 255), 2: (255, 0, 255), 3: (255, 165, 0), 4: (0, 255, 0)}
        
        # Classes ativas
        self.active_classes = [0, 1, 2, 3, 4]
        
        # Filtros de imagem
        self.brightness = 0
        self.contrast = 1.0
        
        # Sistema de rastreamento
        self.tracking_enabled = False
        self.smooth_detections = True
    
    def load_model(self):
        try:
            model_path = "yolov8_large_thermal_15-08-2024.pt"
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.model_loaded = True
            else:
                st.error(f"❌ Modelo não encontrado: {model_path}")
        except Exception as e:
            st.error(f"❌ Erro ao carregar modelo: {str(e)}")
    
    def apply_surveillance_mode(self, mode_key):
        if mode_key not in self.surveillance_modes:
            return
        
        mode = self.surveillance_modes[mode_key]
        self.current_mode = mode_key
        
        # Atualiza classes ativas
        self.active_classes = mode.get('priority_classes', [0, 1, 2, 3, 4])
        
        # Aplica filtros
        filters = mode.get('filters', {})
        self.brightness = filters.get('brightness', 0)
        self.contrast = filters.get('contrast', 1.0)
        
        # Configura rastreamento
        self.tracking_enabled = mode.get('tracking_enabled', False)
        self.smooth_detections = mode.get('smooth_detections', True)
        
        # Limpa histórico
        self.detection_history = []
        self.frame_count = 0
    
    def should_detect_now(self):
        mode_config = self.surveillance_modes[self.current_mode]
        detection_interval = mode_config["detection_interval"]
        
        self.frame_count += 1
        
        if self.frame_count == 1:
            return True
        
        if not self.last_detections:
            return True
        
        return (self.frame_count % detection_interval) == 0
    
    def optimize_frame(self, frame):
        mode_config = self.surveillance_modes[self.current_mode]
        target_resolution = mode_config["resolution"]
        
        # Aplica filtros
        if self.brightness != 0 or self.contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
        
        # Redimensiona
        h, w = frame.shape[:2]
        if w > target_resolution[0] or h > target_resolution[1]:
            scale = min(target_resolution[0]/w, target_resolution[1]/h)
            new_w, new_h = int(w*scale), int(h*scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    def detect_objects(self, frame, force_detection=False):
        if not self.model or not self.model_loaded:
            return frame
        
        mode_config = self.surveillance_modes[self.current_mode]
        
        should_detect = force_detection or self.should_detect_now()
        
        if not should_detect and self.last_detections:
            return self.apply_cached_detections(frame)
        
        try:
            start_time = time.time()
            optimized_frame = self.optimize_frame(frame)
            
            results = self.model(
                optimized_frame, 
                conf=mode_config["conf_threshold"], 
                verbose=False,
                imgsz=640,
                half=True,
                device='cuda' if torch and torch.cuda.is_available() else 'cpu'
            )
            
            # Processa detecções
            detections = []
            counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
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
            self.last_frame_id = self.frame_count
            
            # Histórico para suavização
            if self.smooth_detections:
                self.detection_history.append(detections)
                if len(self.detection_history) > 3:
                    self.detection_history.pop(0)
            
            return self.apply_cached_detections(frame)
            
        except Exception as e:
            st.error(f"Erro na detecção: {str(e)}")
            return frame
    
    def apply_cached_detections(self, frame):
        if not self.last_detections:
            return frame
        
        result_frame = frame.copy()
        detections = self.last_detections['detections']
        
        # Desenha detecções
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            name = det['name']
            color = det['color']
            
            # Desenha retângulo
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Label com fundo
            label = f'{name}: {conf:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_frame, (x1, y1-20), (x1+label_size[0], y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Estatísticas
        counts = self.last_detections['counts']
        y_pos = 20
        stats = []
        if counts[1] > 0: stats.append(f'👤 PESSOAS: {counts[1]}')
        if counts[0] > 0: stats.append(f'🚗 Carros: {counts[0]}')
        if counts[2] > 0: stats.append(f'🏍️ Motos: {counts[2]}')
        if counts[3] > 0: stats.append(f'🚛 Caminhões: {counts[3]}')
        if counts[4] > 0: stats.append(f'🦌 Animais: {counts[4]}')
        
        for stat in stats:
            cv2.putText(result_frame, stat, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
        
        return result_frame

# Inicialização
if 'processor' not in st.session_state:
    st.session_state.processor = OptimizedProcessor()

# Interface principal
st.title("🔥 VIGILÂNCIA TÉRMICA - OTIMIZADA PARA STREAMING")
st.markdown("**Sistema Ultra-Rápido para MAVIC 3T | Oleodutos/Gasodutos**")

# Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.header("📹 Streaming de Vídeo")
    uploaded_file = st.file_uploader("Carregar vídeo térmico", type=['mp4', 'avi', 'mov', 'mpeg4'])

with col2:
    st.header("⚡ Controles Rápidos")
    
    # Status
    if st.session_state.processor.model_loaded:
        st.success("✅ SISTEMA ONLINE")
    else:
        st.error("❌ SISTEMA OFFLINE")
    
    # Seleção de modo
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
    
    # Configurações do modo
    st.markdown("**Configurações do Modo:**")
    detection_interval = current_config['detection_interval']
    resolution = current_config['resolution']
    priority_classes = current_config.get('priority_classes', [])
    
    st.text(f"🔄 Detecção: 1 a cada {detection_interval} frames")
    st.text(f"📐 Resolução: {resolution[0]}x{resolution[1]}")
    
    # Classes detectadas
    class_names_display = {0: '🚗 Carros', 1: '👤 Pessoas', 2: '🏍️ Motos', 3: '🚛 Caminhões', 4: '🦌 Animais'}
    active_classes_display = [class_names_display[cls] for cls in priority_classes if cls in class_names_display]
    
    if active_classes_display:
        st.markdown("**Classes Detectadas:**")
        for class_name in active_classes_display:
            st.write(f"✅ {class_name}")
    
    # Performance
    st.markdown("---")
    st.subheader("⚡ Performance")
    
    performance_boost = st.checkbox("🚀 Boost de Velocidade", value=False, help="Reduz qualidade para máxima velocidade")
    if performance_boost:
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
    st.success(f"📹 {uploaded_file.name} ({file_size_mb:.1f} MB)")
    
    # Arquivo temporário
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    st.info(f"ℹ️ {total_frames} frames | {fps:.1f} FPS | {duration:.1f}s")
    
    # Controles de reprodução
    col_play1, col_play2 = st.columns(2)
    
    with col_play1:
        start_new = st.button("🎬 INICIAR NOVO", help="Inicia do começo")
    
    with col_play2:
        if 'current_frame_count' in st.session_state and st.session_state.current_frame_count > 0:
            continue_video = st.button(f"▶️ CONTINUAR ({st.session_state.current_frame_count}/{total_frames})", help="Continua de onde parou")
        else:
            continue_video = False
    
    # Reset frame count se clicar em iniciar novo
    if start_new:
        st.session_state.current_frame_count = 0
        st.session_state.video_running = True
    
    if start_new or continue_video:
        # Controles dinâmicos
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            st.markdown("### 🎮 Controles em Tempo Real")
        
        dynamic_controls = st.container()
        progress_bar = st.progress(0)
        metrics_placeholder = st.empty()
        detection_status = st.empty()
        video_placeholder = st.empty()
        
        # Configurações iniciais
        mode_config = st.session_state.processor.surveillance_modes[st.session_state.processor.current_mode]
        detection_interval = mode_config["detection_interval"]
        
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        # Flags de controle
        if 'video_running' not in st.session_state:
            st.session_state.video_running = True
        if 'current_frame_count' not in st.session_state:
            st.session_state.current_frame_count = 0
        
        # Posiciona vídeo
        if st.session_state.current_frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_count)
            frame_count = st.session_state.current_frame_count
        
        while st.session_state.video_running:
            # Controles dinâmicos
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
                        key=f"mode_selector_{frame_count//10}"
                    )
                    
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
            st.session_state.current_frame_count = frame_count
            
            # Status de detecção
            is_detecting = st.session_state.processor.should_detect_now()
            if is_detecting:
                detection_status.success("🔍 **DETECTANDO AGORA**")
            else:
                detection_status.info("⏳ **Usando cache anterior**")
            
            # Processamento
            if frame_count % 1 == 0:
                force_detection = (processed_count % detection_interval == 0)
                processed_frame = st.session_state.processor.detect_objects(frame, force_detection)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                video_placeholder.image(frame_rgb, channels="RGB", width=640)
                
                processed_count += 1
                progress_bar.progress(min(1.0, frame_count / total_frames))
                
                # Métricas
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
                
                time.sleep(0.01)
        
        # Fim do vídeo
        st.success("✅ Processamento concluído!")
        st.session_state.current_frame_count = 0
        
    cap.release()
    os.unlink(tfile.name)

st.markdown("---")
st.markdown("** Sistema de Vigilância Térmica OTIMIZADO | Performance Máxima para MAVIC 3T**")