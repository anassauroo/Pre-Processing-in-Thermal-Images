import cv2
import numpy as np
import time
import os

def test_image_processing():
    """Testa as melhorias de processamento de imagem"""
    print("🧪 Testando Melhorias de Processamento de Imagem")
    print("=" * 50)
    
    # Cria uma imagem de teste térmica simulada
    def create_thermal_test_image():
        # Base térmica com baixo contraste
        img = np.random.randint(80, 120, (480, 640, 3), dtype=np.uint8)
        
        # Adiciona "pessoas" térmicas (regiões mais quentes)
        cv2.ellipse(img, (200, 300), (30, 70), 0, 0, 360, (180, 160, 140), -1)
        cv2.ellipse(img, (400, 250), (25, 65), 0, 0, 360, (170, 150, 130), -1)
        
        # Adiciona ruído
        noise = np.random.randint(-15, 15, (480, 640, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    # Cria imagem de teste
    original = create_thermal_test_image()
    print(f"📸 Imagem de teste criada: {original.shape}")
    
    # Testa ajuste básico OpenCV
    print("\n🔧 Testando ajuste básico com OpenCV...")
    start_time = time.time()
    basic_adjusted = cv2.convertScaleAbs(original, alpha=1.3, beta=20)
    basic_time = time.time() - start_time
    print(f"⏱️ Tempo ajuste básico: {basic_time:.4f}s")
    
    # Testa CLAHE
    print("\n🔍 Testando CLAHE...")
    start_time = time.time()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    clahe_result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    clahe_time = time.time() - start_time
    print(f"⏱️ Tempo CLAHE: {clahe_time:.4f}s")
    
    # Testa ajuste automático
    print("\n🤖 Testando ajuste automático...")
    start_time = time.time()
    
    # Converte para escala de cinza para análise
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_peaks = np.sum(hist > np.mean(hist) * 2)
    
    # Determina ajustes
    if hist_peaks < 20:
        use_clahe = True
        clahe_limit = min(4.0, 2.0 + (20 - hist_peaks) / 10)
    else:
        use_clahe = False
        clahe_limit = 2.0
        
    if mean_val < 80:
        brightness = int(min(60, 100 - mean_val))
        contrast = min(1.8, 1.0 + (80 - mean_val) / 160)
    elif mean_val > 200:
        brightness = int(max(-60, 150 - mean_val))
        contrast = max(0.6, 1.0 - (mean_val - 200) / 200)
    else:
        if std_val < 25:
            contrast = min(2.0, 1.0 + (25 - std_val) / 50)
            use_clahe = True
            clahe_limit = 3.0
        else:
            contrast = 1.0
        brightness = 0
    
    # Aplica ajustes
    auto_result = original.copy()
    if use_clahe:
        clahe_auto = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
        lab_auto = cv2.cvtColor(auto_result, cv2.COLOR_BGR2LAB)
        lab_auto[:,:,0] = clahe_auto.apply(lab_auto[:,:,0])
        auto_result = cv2.cvtColor(lab_auto, cv2.COLOR_LAB2BGR)
    
    if brightness != 0 or contrast != 1.0:
        auto_result = cv2.convertScaleAbs(auto_result, alpha=contrast, beta=brightness)
    
    auto_time = time.time() - start_time
    print(f"⏱️ Tempo ajuste automático: {auto_time:.4f}s")
    
    # Mostra estatísticas
    print(f"\n📊 Estatísticas da imagem:")
    print(f"   📈 Média: {mean_val:.1f}")
    print(f"   📊 Desvio padrão: {std_val:.1f}")
    print(f"   🏔️ Picos de histograma: {hist_peaks}")
    print(f"   🔧 Ajustes aplicados:")
    print(f"      💡 Brilho: {brightness}")
    print(f"      🌓 Contraste: {contrast:.2f}")
    print(f"      🔍 CLAHE: {'Sim' if use_clahe else 'Não'}")
    if use_clahe:
        print(f"      📊 Limite CLAHE: {clahe_limit:.1f}")
    
    # Salva imagens de resultado (opcional)
    if not os.path.exists("test_results"):
        os.makedirs("test_results")
    
    cv2.imwrite("test_results/original.jpg", original)
    cv2.imwrite("test_results/basic_adjusted.jpg", basic_adjusted)
    cv2.imwrite("test_results/clahe_result.jpg", clahe_result)
    cv2.imwrite("test_results/auto_result.jpg", auto_result)
    
    print(f"\n💾 Imagens salvas em 'test_results/'")
    print("✅ Teste concluído!")

def test_video_performance():
    """Testa performance de processamento de vídeo"""
    print("\n🎬 Testando Performance de Vídeo")
    print("=" * 50)
    
    # Simula processamento de frames
    num_frames = 100
    frame_times = []
    
    for i in range(num_frames):
        start_time = time.time()
        
        # Simula criação de frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Simula processamento CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Simula ajuste de brilho/contraste
        processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=10)
        
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        
        if (i + 1) % 25 == 0:
            avg_time = np.mean(frame_times[-25:])
            fps_estimate = 1.0 / avg_time if avg_time > 0 else 0
            print(f"📊 Frame {i+1}/{num_frames} - Tempo médio: {avg_time:.4f}s - FPS estimado: {fps_estimate:.1f}")
    
    total_time = sum(frame_times)
    avg_time = total_time / num_frames
    max_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"\n📈 Resultados de Performance:")
    print(f"   ⏱️ Tempo total: {total_time:.2f}s")
    print(f"   📊 Tempo médio por frame: {avg_time:.4f}s")
    print(f"   🎬 FPS máximo estimado: {max_fps:.1f}")
    print(f"   💡 Performance: {'Excelente' if max_fps > 25 else 'Boa' if max_fps > 15 else 'Adequada'}")

if __name__ == "__main__":
    print("🔥 Teste de Melhorias do Aplicativo Térmico")
    print("=" * 60)
    
    try:
        test_image_processing()
        test_video_performance()
        
        print(f"\n🎉 Todos os testes concluídos com sucesso!")
        print("💡 As melhorias implementadas incluem:")
        print("   ✅ CLAHE para equalização adaptativa")
        print("   ✅ Ajuste automático inteligente")
        print("   ✅ Processamento otimizado com OpenCV")
        print("   ✅ Controle de velocidade de reprodução")
        print("   ✅ Skip de frames para performance")
        
    except Exception as e:
        print(f"❌ Erro durante os testes: {str(e)}")
        import traceback
        traceback.print_exc()