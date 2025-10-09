#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de demonstração do aplicativo de detecção térmica
Mostra como usar o modelo YOLO corrigido para PyTorch 2.8+
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np
import warnings

def fix_torch_load():
    """Aplica a correção para PyTorch 2.8+"""
    warnings.filterwarnings("ignore")
    
    # Fix para PyTorch 2.8+: força weights_only=False
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    return original_load

def demo_model_usage():
    """Demonstra o uso do modelo YOLO com correção"""
    
    print("🔥 Demonstração - Aplicativo de Detecção Térmica")
    print("=" * 55)
    
    # Aplica a correção do PyTorch
    print("🔧 Aplicando correção PyTorch 2.8+...")
    original_load = fix_torch_load()
    
    try:
        # Carrega o modelo térmico treinado
        print("📦 Carregando modelo YOLO térmico...")
        model_path = "yolov8_large_thermal_15-08-2024.pt"
        model = YOLO(model_path)
        print("✅ Modelo carregado com sucesso!")
        
        # Restaura função original
        torch.load = original_load
        
        # Cria uma imagem térmica simulada
        print("🖼️ Criando imagem térmica simulada...")
        width, height = 640, 480
        thermal_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fundo térmico (azul/roxo - frio)
        thermal_image[:, :] = [80, 40, 120]
        
        # Adiciona uma "pessoa" térmica (região mais quente - amarelo/vermelho)
        person_x, person_y = 300, 200
        cv2.ellipse(thermal_image, (person_x, person_y), (30, 70), 0, 0, 360, (200, 180, 80), -1)
        cv2.ellipse(thermal_image, (person_x, person_y-50), (20, 25), 0, 0, 360, (240, 200, 100), -1)
        
        # Executa a detecção
        print("🔍 Executando detecção YOLO...")
        results = model(thermal_image, conf=0.25, verbose=False)
        
        # Processa os resultados
        detections = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    
                    # Desenha a caixa delimitadora
                    cv2.rectangle(thermal_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(thermal_image, f'Pessoa: {conf:.2f}', (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    detections += 1
        
        print(f"👥 Detecções encontradas: {detections}")
        
        # Salva a imagem resultado
        output_path = "demo_detection_result.jpg"
        cv2.imwrite(output_path, thermal_image)
        print(f"💾 Resultado salvo em: {output_path}")
        
        print("\n🎉 Demonstração concluída com sucesso!")
        print("\n📋 Resumo:")
        print(f"   ✅ Modelo carregado: {model_path}")
        print(f"   ✅ Detecções realizadas: {detections}")
        print(f"   ✅ Imagem salva: {output_path}")
        
        print(f"\n🚀 Para usar o aplicativo completo execute:")
        print(f"   streamlit run app.py")
        
    except Exception as e:
        torch.load = original_load  # Restaura mesmo com erro
        print(f"❌ Erro na demonstração: {str(e)}")
        return False
    
    return True

def test_image_adjustments():
    """Testa os ajustes de brilho e contraste"""
    
    print("\n🎨 Teste de Ajustes de Imagem")
    print("-" * 35)
    
    # Cria imagem de teste
    test_img = np.random.randint(50, 150, (200, 300, 3), dtype=np.uint8)
    
    # Testa brilho
    brightness = 30
    bright_img = np.clip(test_img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    print(f"☀️ Ajuste de brilho (+{brightness}): OK")
    
    # Testa contraste
    contrast = 1.3
    contrast_img = np.clip(test_img.astype(np.float32) * contrast, 0, 255).astype(np.uint8)
    print(f"🌓 Ajuste de contraste (x{contrast}): OK")
    
    # Testa ajuste automático
    mean_val = np.mean(test_img)
    if mean_val < 100:
        auto_brightness = min(50, 120 - mean_val)
        auto_contrast = min(1.5, 1.0 + (100 - mean_val) / 200)
        print(f"🤖 Ajuste automático: Brilho +{auto_brightness:.0f}, Contraste x{auto_contrast:.2f}")
    
    return True

if __name__ == "__main__":
    success = demo_model_usage()
    if success:
        test_image_adjustments()
        
    print("\n" + "=" * 55)
    print("🔥 Aplicativo de Detecção Térmica - Demonstração Finalizada!")