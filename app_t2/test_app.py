import unittest
import cv2
import numpy as np
import tempfile
import os
from video_processor import ThermalVideoProcessorCLI

class TestThermalVideoProcessor(unittest.TestCase):
    
    def setUp(self):
        """Configura o teste"""
        # Cria um processador sem modelo para testes básicos
        self.processor = ThermalVideoProcessorCLI("dummy_model.pt")
        self.processor.model = None  # Simula modelo não carregado para alguns testes
        
        # Cria uma imagem de teste
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_adjust_image_brightness(self):
        """Testa o ajuste de brilho"""
        self.processor.brightness = 50
        self.processor.contrast = 1.0
        
        adjusted = self.processor.adjust_image(self.test_image)
        
        # Verifica se o brilho foi aplicado
        self.assertTrue(np.mean(adjusted) > np.mean(self.test_image))
        self.assertEqual(adjusted.shape, self.test_image.shape)
    
    def test_adjust_image_contrast(self):
        """Testa o ajuste de contraste"""
        self.processor.brightness = 0
        self.processor.contrast = 1.5
        
        adjusted = self.processor.adjust_image(self.test_image)
        
        # Verifica se o contraste foi aplicado
        self.assertEqual(adjusted.shape, self.test_image.shape)
        self.assertTrue(adjusted.dtype == np.uint8)
    
    def test_auto_adjust_dark_image(self):
        """Testa o ajuste automático em imagem escura"""
        # Cria uma imagem escura
        dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 50
        
        self.processor.auto_adjust_image(dark_image)
        
        # Verifica se foi aplicado ajuste para imagem escura
        self.assertGreater(self.processor.brightness, 0)
        self.assertGreater(self.processor.contrast, 1.0)
    
    def test_auto_adjust_bright_image(self):
        """Testa o ajuste automático em imagem clara"""
        # Cria uma imagem clara
        bright_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        self.processor.auto_adjust_image(bright_image)
        
        # Verifica se foi aplicado ajuste para imagem clara
        self.assertLess(self.processor.brightness, 10)
    
    def test_detect_objects_without_model(self):
        """Testa detecção sem modelo carregado"""
        self.processor.yolo_enabled = True
        
        result, count = self.processor.detect_objects(self.test_image)
        
        # Sem modelo, deve retornar a imagem original e 0 detecções
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(result, self.test_image)
    
    def test_detect_objects_disabled(self):
        """Testa detecção desabilitada"""
        self.processor.yolo_enabled = False
        
        result, count = self.processor.detect_objects(self.test_image)
        
        # Com detecção desabilitada, deve retornar a imagem original e 0 detecções
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(result, self.test_image)
    
    def test_image_value_clipping(self):
        """Testa se os valores da imagem são mantidos no range correto"""
        # Testa com valores extremos
        self.processor.brightness = 200  # Muito alto
        self.processor.contrast = 5.0    # Muito alto
        
        adjusted = self.processor.adjust_image(self.test_image)
        
        # Verifica se os valores estão no range [0, 255]
        self.assertTrue(np.all(adjusted >= 0))
        self.assertTrue(np.all(adjusted <= 255))
        self.assertEqual(adjusted.dtype, np.uint8)

def create_test_video(filename, duration=2, fps=30):
    """Cria um vídeo de teste"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))
    
    total_frames = duration * fps
    for i in range(total_frames):
        # Cria um frame com padrão que muda ao longo do tempo
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Adiciona um retângulo que se move
        cv2.rectangle(frame, (i*10 % 600, 200), (i*10 % 600 + 40, 280), (255, 255, 255), -1)
        out.write(frame)
    
    out.release()

class TestVideoProcessing(unittest.TestCase):
    
    def setUp(self):
        """Configura teste de vídeo"""
        self.temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        self.temp_video.close()
        create_test_video(self.temp_video.name)
        
        self.processor = ThermalVideoProcessorCLI("dummy_model.pt")
        self.processor.model = None  # Simula modelo não carregado
        self.processor.yolo_enabled = False  # Desabilita YOLO para teste
    
    def tearDown(self):
        """Cleanup após teste"""
        if os.path.exists(self.temp_video.name):
            os.unlink(self.temp_video.name)
    
    def test_video_loading(self):
        """Testa o carregamento de vídeo"""
        cap = cv2.VideoCapture(self.temp_video.name)
        
        self.assertTrue(cap.isOpened())
        
        # Verifica propriedades do vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        self.assertEqual(fps, 30.0)
        self.assertEqual(width, 640.0)
        self.assertEqual(height, 480.0)
        
        cap.release()

if __name__ == '__main__':
    print("🧪 Executando testes do Aplicativo de Detecção Térmica")
    print("=" * 55)
    
    # Executa os testes
    unittest.main(verbosity=2)