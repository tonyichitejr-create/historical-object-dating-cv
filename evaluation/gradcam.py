import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def gradcam(model,image,last_conv):

    grad_model=tf.keras.models.Model(

        [model.inputs],

        [

            model.get_layer(last_conv).output,

            model.output

        ]

    )

    with tf.GradientTape() as tape:

        conv,preds=grad_model(np.array([image]))

        idx=np.argmax(preds[0])

        loss=preds[:,idx]

    grads=tape.gradient(loss,conv)

    pooled=tf.reduce_mean(grads,axis=(0,1,2))

    conv=conv[0]

    heatmap=conv@pooled[...,tf.newaxis]

    heatmap=tf.squeeze(heatmap)

    heatmap=np.maximum(heatmap,0)

    heatmap/=heatmap.max()

    plt.imshow(image)

    plt.imshow(

        cv2.resize(

            heatmap.numpy(),

            (224,224)

        ),

        alpha=.4,

        cmap="jet"

    )

    plt.axis("off")

    plt.show()
