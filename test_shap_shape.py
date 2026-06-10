import numpy as np
import tensorflow as tf
from src.inference.predictor import AudioPredictor
from src.inference.explainability import generate_shap_explanation, _resize_to_mel

predictor = AudioPredictor("resnet50_v3", "models/resnet50_v3.keras")
audio = np.random.randn(32000).astype(np.float32)
res = predictor.predict_from_audio(audio, 16000)
mel = res['mel']

shap_values, _ = generate_shap_explanation(
    predictor.get_model(),
    mel,
    predictor.get_arch(),
    background_samples=10
)

print("SHAP VALUES TYPE:", type(shap_values))
if isinstance(shap_values, list):
    print("SHAP VALUES LIST LEN:", len(shap_values))
    print("SHAP VALUES[0] SHAPE:", np.array(shap_values[0]).shape)
else:
    print("SHAP VALUES SHAPE:", np.array(shap_values).shape)

sv = shap_values
if isinstance(sv, list):
    sv = sv[0] if len(sv) > 0 else sv

sv = np.array(sv)
print("After list extraction:", sv.shape)

if sv.ndim == 4:
    sv = sv[0]
elif sv.ndim == 3 and sv.shape[0] == 1:
    sv = sv[0]

print("After batch removal:", sv.shape)

if sv.ndim == 3:
    sv_signed = sv.sum(axis=-1)
else:
    sv_signed = sv

print("sv_signed shape:", sv_signed.shape)

try:
    _resize_to_mel(sv_signed, mel.shape)
    print("Resize worked!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("RESIZE ERROR:", e)
