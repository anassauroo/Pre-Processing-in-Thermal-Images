import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

class ThermalVideoProcessorCLI:
    def __init__(self, model_path):
        """Inicializa o processador de vídeo térmico"""
        self.model_path = model_path
        self.model = None
        self.brightness = 0
        self.contrast = 1.0
        self.yolo_enabled = True
        self.use_clahe = False
        self.clahe_limit = 2.0
        self.clahe_grid = 8
        # Cria objeto CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=(self.clahe_grid, self.clahe_grid))
        
        # Carrega o modelo
        self.load_model()
    
    def load_model(self):
        """Carrega o modelo YOLO"""
        try:
            import torch
            import warnings
            warnings.filterwarnings("ignore")
            
            if not os.path.exists(self.model_path):
                print(f"❌ Arquivo do modelo não encontrado: {self.model_path}")
                return False
            
            # Fix para PyTorch 2.8+: força weights_only=False
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            
            try:
                self.model = YOLO(self.model_path)
                torch.load = original_load  # Restaura função original
                print(f"✅ Modelo YOLO carregado: {self.model_path}")
                return True
            except Exception as e:
                torch.load = original_load  # Restaura função original mesmo com erro
                raise e
                
        except Exception as e:
            print(f"❌ Erro ao carregar o modelo: {str(e)}")
            print("💡 Verifique se o arquivo do modelo está correto")
            return False
    
    def adjust_image(self, image):
        """Aplica ajustes de brilho, contraste e CLAHE usando OpenCV"""
        # Cria uma cópia da imagem para processamento
        adjusted = image.copy()
        
        # Aplica CLAHE se habilitado
        if self.use_clahe:
            # Aplica CLAHE por canal se imagem colorida
            if len(adjusted.shape) == 3:
                # Converte para LAB para melhor resultado
                lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = self.clahe.apply(lab[:,:,0])  # Aplica CLAHE apenas no canal L
                adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                adjusted = self.clahe.apply(adjusted)
        
        # Aplica ajustes de brilho e contraste usando OpenCV
        if self.brightness != 0 or self.contrast != 1.0:
            adjusted = cv2.convertScaleAbs(adjusted, alpha=self.contrast, beta=self.brightness)
        
        return adjusted
    
    def auto_adjust_image(self, image):
        """Ajuste automático inteligente para melhorar a detecção"""
        # Converte para escala de cinza se necessário
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Calcula estatísticas da imagem
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        print(f"📊 Estatísticas - Média: {mean_val:.1f}, Desvio: {std_val:.1f}")
        
        # Verifica se precisa de CLAHE baseado na distribuição
        hist_peaks = np.sum(hist > np.mean(hist) * 2)
        
        if hist_peaks < 20:  # Distribuição concentrada
            self.use_clahe = True
            self.clahe_limit = min(4.0, 2.0 + (20 - hist_peaks) / 10)
            print(f"🔍 CLAHE ativado - Limite: {self.clahe_limit:.1f}")
        else:
            self.use_clahe = False
            print("🔍 CLAHE desativado")
            
        # Ajuste automático baseado na distribuição da imagem
        if mean_val < 80:  # Imagem muito escura
            self.brightness = int(min(60, 100 - mean_val))
            self.contrast = min(1.8, 1.0 + (80 - mean_val) / 160)
            print(f"🌙 Imagem escura - Brilho: +{self.brightness}, Contraste: {self.contrast:.2f}")
        elif mean_val > 200:  # Imagem muito clara
            self.brightness = int(max(-60, 150 - mean_val))
            self.contrast = max(0.6, 1.0 - (mean_val - 200) / 200)
            print(f"☀️ Imagem clara - Brilho: {self.brightness}, Contraste: {self.contrast:.2f}")
        else:
            # Ajuste baseado no desvio padrão para melhorar contraste
            if std_val < 25:  # Baixo contraste
                self.contrast = min(2.0, 1.0 + (25 - std_val) / 50)
                self.use_clahe = True
                self.clahe_limit = 3.0
                print(f"🌫️ Baixo contraste - Contraste: {self.contrast:.2f}, CLAHE: {self.clahe_limit}")
            else:
                self.contrast = 1.0
            self.brightness = 0
            
        # Atualiza o objeto CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=(self.clahe_grid, self.clahe_grid))
    
    def detect_objects(self, image):
        """Executa a detecção YOLO"""
        if not self.model or not self.yolo_enabled:
            return image, 0
            
        try:
            # Executa a detecção
            results = self.model(image, conf=0.25, verbose=False)
            
            person_count = 0
            
            # Desenha as caixas delimitadoras
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Coordenadas da caixa
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        
                        # Desenha a caixa delimitadora
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Adiciona o texto com a confiança
                        label = f'Pessoa: {conf:.2f}'
                        cv2.putText(image, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        person_count += 1
            
            return image, person_count
            
        except Exception as e:
            print(f"❌ Erro na detecção: {str(e)}")
            return image, 0
    
    def process_video(self, input_path, output_path=None, auto_adjust=False):
        """Processa o vídeo completo"""
        # Abre o vídeo de entrada
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"❌ Erro ao abrir o vídeo: {input_path}")
            return False
        
        # Obtém propriedades do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Vídeo: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Configura o writer para salvar o vídeo processado
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        
        print("🚀 Iniciando processamento do vídeo...")
        print("Controles:")
        print("  'q': Sair")
        print("  'a': Ajuste automático")
        print("  '+'/'-': Ajustar brilho")
        print("  '1'/'2': Ajustar contraste")
        print("  'c': Ativar/desativar CLAHE")
        print("  'l'/'k': Ajustar limite CLAHE")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Aplica ajuste automático se solicitado
                if auto_adjust and frame_count == 0:
                    self.auto_adjust_image(frame)
                
                # Aplica ajustes de brilho e contraste
                frame = self.adjust_image(frame)
                
                # Aplica detecção YOLO
                processed_frame, detections = self.detect_objects(frame)
                total_detections += detections
                
                # Adiciona informações na tela
                info_text = f"Frame: {frame_count+1}/{total_frames} | Pessoas: {detections}"
                cv2.putText(processed_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Mostra o frame processado
                cv2.imshow('Detecção Térmica', processed_frame)
                
                # Salva o frame se especificado
                if output_path:
                    out.write(processed_frame)
                
                frame_count += 1
                
                # Verifica teclas pressionadas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    self.auto_adjust_image(frame)
                    print("🤖 Ajuste automático aplicado")
                elif key == ord('+') or key == ord('='):
                    self.brightness = min(100, self.brightness + 10)
                    print(f"☀️ Brilho: {self.brightness}")
                elif key == ord('-'):
                    self.brightness = max(-100, self.brightness - 10)
                    print(f"🌙 Brilho: {self.brightness}")
                elif key == ord('1'):
                    self.contrast = min(3.0, self.contrast + 0.1)
                    print(f"🌓 Contraste: {self.contrast:.1f}")
                elif key == ord('2'):
                    self.contrast = max(0.1, self.contrast - 0.1)
                    print(f"🌫️ Contraste: {self.contrast:.1f}")
                elif key == ord('c'):
                    self.use_clahe = not self.use_clahe
                    status = "ativado" if self.use_clahe else "desativado"
                    print(f"🔍 CLAHE {status}")
                elif key == ord('l'):
                    if self.use_clahe:
                        self.clahe_limit = min(8.0, self.clahe_limit + 0.5)
                        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=(self.clahe_grid, self.clahe_grid))
                        print(f"📊 CLAHE Limite: {self.clahe_limit:.1f}")
                elif key == ord('k'):
                    if self.use_clahe:
                        self.clahe_limit = max(1.0, self.clahe_limit - 0.5)
                        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=(self.clahe_grid, self.clahe_grid))
                        print(f"📊 CLAHE Limite: {self.clahe_limit:.1f}")
                
                # Mostra progresso a cada 100 frames
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"📊 Progresso: {progress:.1f}% - {detections} pessoas detectadas neste frame")
        
        except KeyboardInterrupt:
            print("\n⏹️ Processamento interrompido pelo usuário")
        
        finally:
            # Cleanup
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"✅ Processamento concluído!")
            print(f"📊 Total de frames processados: {frame_count}")
            print(f"👥 Total de detecções: {total_detections}")
            if output_path:
                print(f"💾 Vídeo salvo em: {output_path}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Processador de Vídeo Térmico com Detecção YOLO')
    parser.add_argument('input_video', help='Caminho para o vídeo de entrada')
    parser.add_argument('--output', '-o', help='Caminho para salvar o vídeo processado')
    parser.add_argument('--model', '-m', default='yolov8_large_thermal_15-08-2024.pt', 
                       help='Caminho para o modelo YOLO (padrão: yolov8_large_thermal_15-08-2024.pt)')
    parser.add_argument('--auto', '-a', action='store_true', 
                       help='Aplica ajuste automático no primeiro frame')
    parser.add_argument('--brightness', '-b', type=int, default=0, 
                       help='Ajuste inicial de brilho (-100 a 100)')
    parser.add_argument('--contrast', '-c', type=float, default=1.0, 
                       help='Ajuste inicial de contraste (0.1 a 3.0)')
    parser.add_argument('--clahe', action='store_true',
                       help='Ativa CLAHE (equalização adaptativa)')
    parser.add_argument('--clahe-limit', type=float, default=2.0,
                       help='Limite CLAHE (1.0 a 8.0)')
    
    args = parser.parse_args()
    
    # Verifica se o arquivo de entrada existe
    if not os.path.exists(args.input_video):
        print(f"❌ Arquivo de vídeo não encontrado: {args.input_video}")
        return
    
    # Inicializa o processador
    processor = ThermalVideoProcessorCLI(args.model)
    
    if not processor.model:
        print("❌ Não foi possível carregar o modelo YOLO. Verifique o caminho do arquivo.")
        return
    
    # Define valores iniciais
    processor.brightness = args.brightness
    processor.contrast = args.contrast
    processor.use_clahe = args.clahe
    processor.clahe_limit = args.clahe_limit
    
    if processor.use_clahe:
        processor.clahe = cv2.createCLAHE(clipLimit=processor.clahe_limit, tileGridSize=(processor.clahe_grid, processor.clahe_grid))
    
    # Processa o vídeo
    processor.process_video(args.input_video, args.output, args.auto)

if __name__ == '__main__':
    main()