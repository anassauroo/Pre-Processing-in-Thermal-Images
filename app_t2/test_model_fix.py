import torch
import os
from ultralytics import YOLO
import warnings

def test_model_loading_v2():
    """Testa carregamento com weights_only=False"""
    
    print("🧪 Teste de Carregamento YOLO (PyTorch 2.8 Fix)")
    print("=" * 50)
    
    models_to_test = [
        "yolov8_large_thermal_15-08-2024.pt",
        "yolov8n.pt"
    ]
    
    # Suprime avisos
    warnings.filterwarnings("ignore")
    
    for model_path in models_to_test:
        print(f"\n📦 Testando modelo: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Arquivo não encontrado: {model_path}")
            continue
            
        try:
            # Força o carregamento com weights_only=False
            print("   Carregando com weights_only=False...")
            
            # Monkey patch do torch.load para forçar weights_only=False
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            
            model = YOLO(model_path)
            
            # Restaura o torch.load original
            torch.load = original_load
            
            print(f"✅ Sucesso! Modelo carregado: {model_path}")
            
            # Teste básico de inferência
            print("   Testando inferência...")
            import numpy as np
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = model(test_image, verbose=False)
            print(f"   - Inferência OK: {len(results)} resultado(s)")
            
            del model
            
        except Exception as e:
            print(f"❌ Erro: {str(e)}")
            
            # Restaura o torch.load original em caso de erro
            torch.load = original_load

if __name__ == "__main__":
    test_model_loading_v2()