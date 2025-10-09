import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time

def create_simple_video_app():
    """Versão simplificada para testar o problema de reprodução"""
    st.title("🧪 Teste de Reprodução Contínua")
    
    # Estados persistentes
    if 'video_playing' not in st.session_state:
        st.session_state.video_playing = False
    if 'video_paused' not in st.session_state:
        st.session_state.video_paused = False
    if 'brightness' not in st.session_state:
        st.session_state.brightness = 0
    if 'contrast' not in st.session_state:
        st.session_state.contrast = 1.0
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = 0
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Vídeo")
        video_placeholder = st.empty()
        
        uploaded_file = st.file_uploader("Carregar vídeo", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            # Salva arquivo temporário apenas uma vez
            if 'temp_file' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()
                
                st.session_state.temp_file = tfile.name
                st.session_state.current_file = uploaded_file.name
                st.session_state.video_cap = cv2.VideoCapture(tfile.name)
                st.session_state.total_frames = int(st.session_state.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                st.session_state.fps = st.session_state.video_cap.get(cv2.CAP_PROP_FPS)
    
    with col2:
        st.subheader("🎛️ Controles")
        
        # Controles de reprodução
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶️ Play"):
                st.session_state.video_playing = True
                st.session_state.video_paused = False
        
        with col_b:
            if st.button("⏸️ Pause"):
                st.session_state.video_paused = True
        
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.video_playing = False
            st.session_state.video_paused = False
            if 'video_cap' in st.session_state and st.session_state.video_cap:
                st.session_state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                st.session_state.current_frame = 0
        
        st.markdown("---")
        
        # Controles de ajuste - com chaves únicas
        new_brightness = st.slider("☀️ Brilho", -100, 100, st.session_state.brightness, key="brightness_slider")
        if new_brightness != st.session_state.brightness:
            st.session_state.brightness = new_brightness
            st.info(f"Brilho atualizado: {new_brightness}")
        
        new_contrast = st.slider("🌓 Contraste", 0.1, 3.0, st.session_state.contrast, 0.1, key="contrast_slider")
        if new_contrast != st.session_state.contrast:
            st.session_state.contrast = new_contrast
            st.info(f"Contraste atualizado: {new_contrast:.1f}")
        
        # Status
        st.markdown("---")
        if st.session_state.video_playing and not st.session_state.video_paused:
            st.success("▶️ Reproduzindo")
        elif st.session_state.video_paused:
            st.warning("⏸️ Pausado")
        else:
            st.info("⏹️ Parado")
    
    # Reprodução de vídeo
    if uploaded_file is not None and 'video_cap' in st.session_state:
        if st.session_state.video_playing and not st.session_state.video_paused:
            # Processa alguns frames
            for _ in range(3):  # Máximo 3 frames por execução
                if not st.session_state.video_playing or st.session_state.video_paused:
                    break
                
                ret, frame = st.session_state.video_cap.read()
                if not ret:
                    st.session_state.video_playing = False
                    st.session_state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break
                
                # Aplica ajustes
                if st.session_state.brightness != 0 or st.session_state.contrast != 1.0:
                    frame = cv2.convertScaleAbs(frame, alpha=st.session_state.contrast, beta=st.session_state.brightness)
                
                # Converte e mostra
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_column_width=True)
                
                st.session_state.current_frame += 1
                
                # Controla velocidade
                time.sleep(1/st.session_state.fps if st.session_state.fps > 0 else 0.033)
            
            # Agenda próxima atualização
            time.sleep(0.1)
            st.rerun()
    
    # Informações
    if uploaded_file is not None and 'total_frames' in st.session_state:
        progress = st.session_state.current_frame / st.session_state.total_frames
        st.progress(progress)
        st.text(f"Frame {st.session_state.current_frame}/{st.session_state.total_frames}")

if __name__ == "__main__":
    create_simple_video_app()