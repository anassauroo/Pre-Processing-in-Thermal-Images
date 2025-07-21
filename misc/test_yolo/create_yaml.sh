%%bash
cat << 'EOF' > /content/thermal.yaml
path: /content/thermal
train: train/images
val:   val/images

nc: 1
names: ['pessoa'] 
EOF

cat << 'EOF' > /content/thermal_proc.yaml
path: /content/thermal_proc
train: train/images
val:   val/images

nc: 1
names: ['pessoa']
EOF