print("treinando no dataset original")
orig = YOLO('yolov8n.pt').train(
    data='/content/thermal.yaml',
    epochs=10, 
    imgsz=640,
    batch=4,
    name='run_original'
)

print("avaliando modelo original")
metrics_orig = YOLO('/content/runs/detect/run_original/weights/best.pt').val(
    data='/content/thermal.yaml'
)

print("treinando no dataset pre processado")
proc = YOLO('yolov8n.pt').train(
    data='/content/thermal_proc.yaml',
    epochs=10,
    imgsz=640,
    batch=4,
    name='run_processed'
)

print("analisando pre processado")
metrics_proc = YOLO('/content/runs/detect/run_original2/weights/best.pt').val(
    data='/content/thermal_proc.yaml'
)


print("----RESULTADOS---")
print(f"ORIGINAL   mAP@0.5      = {metrics_orig.box.map50:.3f}")
print(f"PRÉ-PROCESSADO   mAP@0.5 = {metrics_proc.box.map50:.3f}")
print(f"ORIGINAL   mAP@0.5:0.95 = {metrics_orig.box.map:.3f}")
print(f"PRÉ-PROCESSADO   mAP@0.5:0.95 = {metrics_proc.box.map:.3f}")