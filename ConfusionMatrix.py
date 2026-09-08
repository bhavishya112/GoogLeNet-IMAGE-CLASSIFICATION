import numpy as np
import tensorflow as tf
from traffic import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

TEST_SIZE = 0.3

def main():
    Model = tf.keras.models.load_model('TM1.keras')

    images,labels = load_data("gtsrb/gtsrb")

    labels = tf.keras.utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    ypred = np.argmax(Model.predict(x_test),axis=1)
    ytrue = np.argmax(y_test,axis=1)

    # CM = confusion_matrix(ytrue,ypred, normalize='all')
    disp = ConfusionMatrixDisplay.from_predictions(y_true=ytrue,
                                                y_pred=ypred,normalize='pred',
                                                include_values=False,
                                                display_labels=np.unique(ytrue))
    
    # (confusion_matrix=CM, display_labels=np.unique(ytrue))
    disp.plot()
    plt.show()

if __name__ == "__main__":
    main()