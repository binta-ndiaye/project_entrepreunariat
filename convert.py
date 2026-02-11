import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. Chargez votre modèle .pkl
clr = joblib.load('modele.pkl')

# 2. Définissez l'entrée (ex: si votre modèle attend une image 128x128 aplatie)
# Changez 16384 par le nombre exact de caractéristiques de votre modèle (128*128=16384)
initial_type = [('float_input', FloatTensorType([None, 16384]))]

# 3. Convertir
onx = convert_sklearn(clr, initial_types=initial_type)

# 4. Enregistrer
with open("modele.onnx", "wb") as f:
    f.write(onx.serialize_to_string())
print("Modèle converti en modele.onnx !")