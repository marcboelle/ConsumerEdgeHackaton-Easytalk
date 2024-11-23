from transformers import MarianMTModel, MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import OptimizationConfig
from pathlib import Path
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Charger le modèle et le tokenizer
model_id = "Helsinki-NLP/opus-mt-fr-en"
tokenizer = MarianTokenizer.from_pretrained(model_id)
model = MarianMTModel.from_pretrained(model_id)

# Exporter en ONNX
onnx_model_dir = "onnx_helsinki_fr_to_en"

ort_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
ort_model.save_pretrained(onnx_model_dir)

# Directory where the ONNX models are saved

# Identify ONNX files
onnx_files = list(Path(onnx_model_dir).glob("*.onnx"))
print("ONNX Files Found:", onnx_files)

# Quantization configuration
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

# Output directory for quantized models
quantized_model_dir = "onnx_quantized_fr_to_en"
quantized_model_dir_path = Path(quantized_model_dir)
quantized_model_dir_path.mkdir(parents=True, exist_ok=True)

# Quantize each ONNX file separately
for onnx_file in onnx_files:
    print(f"Quantizing {onnx_file.name}...")
    quantizer = ORTQuantizer.from_pretrained(onnx_model_dir, file_name=onnx_file.name)
    quantizer.quantize(save_dir=quantized_model_dir, quantization_config=qconfig)

print(f"Quantized models saved to: {quantized_model_dir}")
