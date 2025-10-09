import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_yolo_detection():
    """Testa especificamente a detecção YOLO"""
    print("🧪 Teste de Detecção YOLO")
    print("=" * 40)
    
    model_path = "yolov8_large_thermal_15-08-2024.pt"
    
    # Verifica se o modelo existe
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return False
    
    try:
        # Fix para PyTorch 2.8+
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        # Carrega o modelo
        print("📥 Carregando modelo YOLO...")
        model = YOLO(model_path)
        torch.load = original_load
        
        print(f"✅ Modelo carregado: {model_path}")
        print(f"🏷️ Classes disponíveis: {len(model.names) if hasattr(model, 'names') else 'Desconhecido'}")
        
        if hasattr(model, 'names'):
            print(f"📋 Lista de classes: {model.names}")
        
        # Cria imagem de teste com formas que podem ser detectadas
        print("\n🖼️ Criando imagem de teste...")
        test_image = np.random.randint(50, 150, (640, 480, 3), dtype=np.uint8)
        
        # Adiciona algumas formas que podem ser interpretadas como objetos
        # Retângulo grande (pode ser detectado como carro/veículo)
        cv2.rectangle(test_image, (100, 200), (300, 350), (200, 180, 160), -1)
        
        # Elipse (pode ser detectada como pessoa)
        cv2.ellipse(test_image, (400, 300), (40, 80), 0, 0, 360, (180, 160, 140), -1)
        
        # Outro retângulo menor
        cv2.rectangle(test_image, (200, 100), (350, 180), (190, 170, 150), -1)
        
        print(f"📐 Imagem de teste criada: {test_image.shape}")
        
        # Executa detecção
        print("\n🔍 Executando detecção...")
        results = model(test_image, conf=0.1, verbose=True)  # Confiança baixa para teste
        
        print(f"📊 Resultados obtidos: {len(results)}")
        
        detection_count = 0
        result_image = test_image.copy()
        
        # Processa resultados
        for i, result in enumerate(results):
            print(f"\n📋 Resultado {i+1}:")
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"   📦 Caixas encontradas: {len(boxes)}")
                
                for j in range(len(boxes)):
                    # Extrai informações da caixa
                    x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().astype(int)
                    conf = float(boxes.conf[j].cpu().numpy())
                    
                    if hasattr(boxes, 'cls'):
                        cls_id = int(boxes.cls[j].cpu().numpy())
                        class_name = model.names.get(cls_id, f'classe_{cls_id}') if hasattr(model, 'names') else 'objeto'
                    else:
                        cls_id = -1
                        class_name = 'objeto'
                    
                    print(f"   🎯 Detecção {j+1}: {class_name} (conf: {conf:.3f}) [{x1},{y1},{x2},{y2}]")
                    
                    # Desenha na imagem
                    color = (0, 255, 0)  # Verde
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
                    
                    label = f'{class_name}: {conf:.2f}'
                    cv2.putText(result_image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    detection_count += 1
            else:
                print("   ❌ Nenhuma caixa encontrada neste resultado")
        
        print(f"\n📈 Total de detecções: {detection_count}")
        
        # Salva imagens
        cv2.imwrite("test_original.jpg", test_image)
        cv2.imwrite("test_detection.jpg", result_image)
        
        print("💾 Imagens salvas:")
        print("   📄 test_original.jpg - Imagem original")
        print("   📄 test_detection.jpg - Com detecções")
        
        if detection_count > 0:
            print("✅ Detecção funcionando corretamente!")
        else:
            print("⚠️ Nenhuma detecção encontrada - pode ser normal com imagem sintética")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_yolov8n():
    """Testa com modelo padrão para comparação"""
    print("\n🆚 Teste com YOLO Padrão")
    print("=" * 40)
    
    try:
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        print("📥 Carregando YOLOv8n padrão...")
        model = YOLO("yolov8n.pt")
        torch.load = original_load
        
        # Imagem de teste simples
        test_image = np.random.randint(100, 200, (640, 480, 3), dtype=np.uint8)
        
        # Adiciona formas mais óbvias
        cv2.rectangle(test_image, (200, 150), (450, 400), (255, 255, 255), -1)  # Objeto branco
        cv2.circle(test_image, (320, 275), 50, (100, 100, 100), -1)  # Círculo
        
        print("🔍 Executando detecção com modelo padrão...")
        results = model(test_image, conf=0.1, verbose=False)
        
        detection_count = 0
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                detection_count += len(result.boxes)
        
        print(f"📊 Detecções com modelo padrão: {detection_count}")
        
        return detection_count > 0
        
    except Exception as e:
        print(f"❌ Erro com modelo padrão: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔥 Teste Completo de Detecção YOLO")
    print("=" * 50)
    
    # Testa modelo treinado
    success = test_yolo_detection()
    
    # Testa modelo padrão para comparação
    test_with_yolov8n()
    
    print("\n🎉 Teste concluído!")
    print("💡 Se as detecções não aparecem no app, verifique:")
    print("   ✅ Modelo carregado corretamente")
    print("   ✅ Checkbox 'YOLO - Detectar Pessoas' ativado")
    print("   ✅ Vídeo contém objetos detectáveis")
    print("   ✅ Confiança não muito alta (teste com 0.1-0.3)")
    print("   ✅ Console do Streamlit para mensagens de debug")