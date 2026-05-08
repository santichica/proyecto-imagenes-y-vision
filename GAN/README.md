# Pasos
# 1. Crear entorno virtual
conda create -n gan-tf python=3.10 -y
conda activate gan-tf

# 2. Instalar dependencias
pip install tensorflow==2.15.0
pip install pandas matplotlib

# 3. Validar version de keras: debe ser 2.15.x
pip list | grep keras