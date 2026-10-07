import os
import sys

# Ensure UTF-8 stdout on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import tensorflow as tf
import numpy as np

H5_MODEL_PATH = 'rose_model.h5'
TFLITE_MODEL_PATH = 'rose_model.tflite'

def convert_h5_to_tflite():
    if not os.path.exists(H5_MODEL_PATH):
        print(f"[ERROR] {H5_MODEL_PATH} not found!")
        return False

    print(f"[*] Loading {H5_MODEL_PATH}...")
    keras_model = tf.keras.models.load_model(H5_MODEL_PATH)

    print("[*] Converting Keras model to TensorFlow Lite FlatBuffer...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # Use standard optimizations (quantizes weights to float16 or dynamic range where suitable)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

    h5_size_mb = os.path.getsize(H5_MODEL_PATH) / (1024 * 1024)
    tflite_size_mb = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
    print(f"[OK] Model successfully converted and saved to {TFLITE_MODEL_PATH}!")
    print(f"  - Original H5 size:     {h5_size_mb:.2f} MB")
    print(f"  - Optimized TFLite size: {tflite_size_mb:.2f} MB ({(1 - tflite_size_mb/h5_size_mb)*100:.1f}% reduction)")

    # Parity verification with random tensor
    print("\n[*] Verifying numerical parity between H5 and TFLite...")
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)

    # H5 prediction
    h5_pred = keras_model.predict(dummy_input, verbose=0)[0]

    # TFLite prediction
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    tflite_pred = interpreter.get_tensor(output_details[0]['index'])[0]

    max_diff = float(np.max(np.abs(h5_pred - tflite_pred)))
    print(f"  - H5 Output:     {np.round(h5_pred, 4)}")
    print(f"  - TFLite Output: {np.round(tflite_pred, 4)}")
    print(f"  - Max absolute discrepancy: {max_diff:.6f}")

    if max_diff < 1e-2:
        print("[OK] Parity check PASSED! Model predictions are virtually identical.")
        return True
    else:
        print("[WARN] Output discrepancy is higher than expected.")
        return False

if __name__ == '__main__':
    convert_h5_to_tflite()
