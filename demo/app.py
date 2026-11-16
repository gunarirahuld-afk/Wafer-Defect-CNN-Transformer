import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers as L
import gradio as gr

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
IMG, NC = 64, 9


@keras.saving.register_keras_serializable()
class PositionEmbedding(L.Layer):
    def build(self, input_shape):
        self.pos = self.add_weight(name="pos", shape=(1, input_shape[1], input_shape[2]), initializer="random_normal")

    def call(self, x):
        return x + self.pos


model = keras.models.load_model("Hybrid_CNN_Transformer_v2.keras", compile=False)
gm = keras.Model(model.inputs, [model.get_layer("last_conv").output, model.output])

d = np.load("test_wafers.npz")
X_te, y_te = d["X_te"], d["y_te"]          # held-out TEST wafers only (never seen in training)
onehot = lambda a: np.eye(3, dtype=np.float32)[a]


def predict(defect):
    i = int(np.random.choice(np.where(y_te == CLASSES.index(defect))[0]))
    wafer = X_te[i]
    x = tf.convert_to_tensor(onehot(wafer)[None])
    with tf.GradientTape() as tape:
        conv, pred = gm(x)
        c = int(tf.argmax(pred[0]))
        score = pred[:, c]
    g = tape.gradient(score, conv)
    w = tf.reduce_mean(g, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * w, -1)).numpy()
    cam /= cam.max() + 1e-8
    cam = tf.image.resize(cam[..., None], (IMG, IMG)).numpy()[..., 0]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(wafer, cmap="viridis"); ax[0].set_title(f"Test wafer (true: {defect})"); ax[0].axis("off")
    ax[1].imshow(wafer, cmap="gray"); ax[1].imshow(cam, cmap="jet", alpha=0.5)
    ax[1].set_title(f"Grad-CAM (predicted: {CLASSES[c]})"); ax[1].axis("off")
    plt.tight_layout()
    return fig, {CLASSES[k]: float(pred[0, k]) for k in range(NC)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Dropdown(CLASSES, value="Scratch", label="Pick a defect type"),
    outputs=[gr.Plot(label="Wafer and Grad-CAM heatmap"), gr.Label(num_top_classes=3, label="Prediction")],
    title="Explainable Hybrid CNN-Transformer: Wafer Defect Classifier",
    description=("Lightweight hybrid CNN-Transformer (563K parameters) trained on WM-811K. "
                 "Pick a defect type to load a random held-out test wafer and see the prediction "
                 "and the Grad-CAM heatmap showing where the model looked."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
