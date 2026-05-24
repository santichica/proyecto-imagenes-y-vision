# 1. Crear y activar el entorno virtual
conda create -n gan-api python=3.10                                        
conda activate gan-api

# 2. Instalar dependencias
pip install tensorflow fastapi uvicorn  

# 3. Instalar tensorflow 2.15.0
pip install tensorflow==2.15.0   

# 4. Levantar el back
python -m uvicorn main:app --reload

# Stack tecnologico
- Backend: FastAPI
- Front: HTML, JavaScript
- Version instalada de tensorflow: 2.15.0
- Version instalada de pytorch: 2.12.0