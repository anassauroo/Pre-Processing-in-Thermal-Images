import streamlit as st
import cv2
import numpy as np
from PIL import Image

def create_demo_video():
    """Cria um vídeo de demonstração térmico simulado"""
    st.subheader("🎬 Criar Vídeo de Demonstração")
    
    if st.button("Gerar Vídeo Térmico de Teste"):
        # Configurações do vídeo
        width, height = 640, 480
        fps = 15
        duration = 10  # segundos
        total_frames = fps * duration
        
        # Cria um arquivo temporário
        output_file = "demo_thermal_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for frame_num in range(total_frames):
            # Cria um fundo térmico simulado (tons de azul/roxo para frio, vermelho/amarelo para quente)
            background = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Gradiente térmico de base
            for y in range(height):
                for x in range(width):
                    temp_val = int(50 + 30 * np.sin(x * 0.01) + 20 * np.cos(y * 0.01))
                    background[y, x] = [temp_val, temp_val//2, max(0, 100-temp_val)]
            
            # Adiciona "pessoas" térmicas (regiões mais quentes)
            # Pessoa 1 - se movendo da esquerda para direita
            person1_x = int(50 + (frame_num * 4) % (width - 100))
            person1_y = 200
            cv2.ellipse(background, (person1_x, person1_y), (25, 60), 0, 0, 360, (255, 200, 100), -1)
            cv2.ellipse(background, (person1_x, person1_y-40), (15, 20), 0, 0, 360, (255, 220, 150), -1)  # Cabeça
            
            # Pessoa 2 - se movendo verticalmente
            person2_x = 400
            person2_y = int(100 + 50 * np.sin(frame_num * 0.1)) + 150
            cv2.ellipse(background, (person2_x, person2_y), (30, 65), 0, 0, 360, (240, 180, 80), -1)
            cv2.ellipse(background, (person2_x, person2_y-45), (18, 22), 0, 0, 360, (250, 200, 120), -1)  # Cabeça
            
            # Adiciona ruído térmico
            noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
            background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Simula diferentes condições térmicas ao longo do tempo
            if frame_num < total_frames // 3:
                # Início: condições normais
                pass
            elif frame_num < 2 * total_frames // 3:
                # Meio: reduz o contraste (simula condições difíceis)
                background = cv2.convertScaleAbs(background, alpha=0.7, beta=10)
            else:
                # Final: escurece a imagem (simula condições noturnas)
                background = cv2.convertScaleAbs(background, alpha=0.5, beta=-20)
            
            # Adiciona informações do frame
            cv2.putText(background, f"Frame: {frame_num+1}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(background, f"Time: {frame_num/fps:.1f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(background)
            
            # Atualiza progresso
            progress = (frame_num + 1) / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Gerando frame {frame_num+1}/{total_frames}")
        
        out.release()
        progress_bar.progress(1.0)
        status_text.text("✅ Vídeo de demonstração criado!")
        
        st.success(f"🎬 Vídeo criado: {output_file}")
        st.info("💡 Agora você pode usar este vídeo para testar o aplicativo principal!")
        
        # Mostra um frame de exemplo
        cap = cv2.VideoCapture(output_file)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Preview do vídeo térmico de demonstração", use_column_width=True)
        cap.release()

def show_thermal_colormap_info():
    """Mostra informações sobre colormaps térmicos"""
    st.subheader("🌡️ Sobre Imagens Térmicas")
    
    st.write("""
    **Como interpretar imagens térmicas:**
    - 🔵 **Azul/Roxo**: Temperaturas mais baixas (frio)
    - 🟡 **Amarelo**: Temperaturas médias
    - 🔴 **Vermelho/Branco**: Temperaturas mais altas (quente)
    
    **Pessoas em imagens térmicas:**
    - Aparecem como regiões mais quentes devido ao calor corporal
    - Contraste com o ambiente mais frio
    - Cabeça e tronco são geralmente as partes mais visíveis
    """)
    
    # Cria uma demonstração de colormap térmico
    thermal_demo = np.zeros((100, 400, 3), dtype=np.uint8)
    for x in range(400):
        temp_ratio = x / 400.0
        if temp_ratio < 0.33:
            # Frio - azul para roxo
            r = int(temp_ratio * 3 * 100)
            g = 0
            b = int(255 - temp_ratio * 3 * 100)
        elif temp_ratio < 0.66:
            # Médio - roxo para amarelo
            ratio = (temp_ratio - 0.33) * 3
            r = int(100 + ratio * 155)
            g = int(ratio * 200)
            b = int(155 - ratio * 155)
        else:
            # Quente - amarelo para branco
            ratio = (temp_ratio - 0.66) * 3
            r = 255
            g = int(200 + ratio * 55)
            b = int(ratio * 255)
        
        thermal_demo[:, x] = [r, g, b]
    
    st.image(thermal_demo, caption="Escala térmica: Frio ← → Quente", use_column_width=True)

# Interface principal
st.set_page_config(
    page_title="Demo Térmico",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Gerador de Vídeo Térmico de Demonstração")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    create_demo_video()

with col2:
    show_thermal_colormap_info()

st.markdown("---")
st.info("💡 **Dica**: Após gerar o vídeo, use-o no aplicativo principal (`streamlit run app.py`) para testar todas as funcionalidades!")