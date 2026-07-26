import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model("models/resnet.keras")

spec = (

    tf.TensorSpec(

        (None,224,224,3),

        tf.float32,

        name="input"

    ),

)

tf2onnx.convert.from_keras(

    model,

    input_signature=spec,

    output_path="models/history_classifier.onnx"

)

print("ONNX model exported.")
