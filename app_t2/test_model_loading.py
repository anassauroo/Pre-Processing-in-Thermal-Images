import torch
import os
from ultralytics import YOLO
import warnings

def test_model_loading():
    """Testa o carregamento dos modelos YOLO disponíveis"""
    
    print("🧪 Teste de Carregamento de Modelos YOLO")
    print("=" * 50)
    
    # Lista de modelos para testar
    models_to_test = [
        "yolov8_large_thermal_15-08-2024.pt",
        "yolov8n.pt"
    ]
    
    # Configurações para resolver problemas do PyTorch
    try:
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
    except:
        pass
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    for model_path in models_to_test:
        print(f"\n📦 Testando modelo: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Arquivo não encontrado: {model_path}")
            continue
            
        try:
            # Primeira tentativa
            print("   Tentativa 1: Carregamento padrão...")
            model = YOLO(model_path)
            print(f"✅ Sucesso! Modelo carregado: {model_path}")
            
            # Teste básico
            print("   Testando estrutura do modelo...")
            print(f"   - Tipo: {type(model)}")
            print(f"   - Task: {getattr(model, 'task', 'N/A')}")
            
            del model  # Libera memória
            
        except Exception as e:
            print(f"❌ Erro no carregamento padrão: {str(e)}")
            
            try:
                # Segunda tentativa com configurações alternativas
                print("   Tentativa 2: Carregamento alternativo...")
                
                # Força carregamento sem restrições
                import torch.serialization
                original_weights_only = torch.serialization.DEFAULT_PROTOCOL
                
                model = YOLO(model_path)
                print(f"✅ Sucesso alternativo! Modelo carregado: {model_path}")
                
                del model  # Libera memória
                
            except Exception as e2:
                print(f"❌ Erro no carregamento alternativo: {str(e2)}")
    
    print("\n" + "=" * 50)
    print("🔍 Informações do Sistema:")
    print(f"PyTorch Version: {torch.__version__}")
    
    try:
        from ultralytics import __version__ as ultralytics_version
        print(f"Ultralytics Version: {ultralytics_version}")
    except:
        print("Ultralytics Version: N/A")
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    test_model_loading()