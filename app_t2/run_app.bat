@echo off
echo 🔥 Aplicativo de Detecção Térmica
echo ================================

echo.
echo Instalando dependências...
pip install -r requirements.txt

echo.
echo Verificando modelo YOLO...
if exist "yolov8_large_thermal_15-08-2024.pt" (
    echo ✅ Modelo YOLO encontrado
) else (
    echo ❌ Modelo YOLO não encontrado: yolov8_large_thermal_15-08-2024.pt
    echo Por favor, certifique-se de que o arquivo está na pasta raiz
    pause
    exit /b 1
)

echo.
echo 🚀 Iniciando aplicativo web...
echo Acesse: http://localhost:8501
echo.
streamlit run app.py

pause