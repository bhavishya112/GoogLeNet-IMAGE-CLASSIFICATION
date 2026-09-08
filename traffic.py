"""
IMPORTANT NOTE FOR THE READER
The lack of confidence as shown in confusion matrices(CM & CM1) is merely due to class imbalance in the Dataset(by gtsrb)

Result:
    Accuracy:
    TM -> 85% with 30%(of 60%) validation and overall 40% test size 
    TM1 -> 92% with 15%(of 85%) validation and 15% test size

Yes Inception Model was not needed, but i was just reading research papers 😜
"""
import datetime
import termcolor
import cv2
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import keras
import tensorflow as tf
from tensorflow.keras.layers import ( # type: ignore
    Input, Conv2D, AveragePooling2D, MaxPooling2D, 
    GlobalMaxPooling2D, Dense, Dropout, ReLU, Concatenate
)
from tensorflow.keras.models import Model # type: ignore

# EPOCHS = 80
EPOCHS = 20
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.25


def main():
    sys.argv = ["traffic.py","gtsrb","TM1.keras"]
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Preprocess Images
    images = preprocess(images)

    # Balance unbalanced classes using their weights stored in weightDict as label:weight
    weights = class_weight.compute_class_weight(class_weight="balanced",classes=np.unique(labels),y=labels)
    weightDict = dict()
    for label in np.unique(labels):
        weightDict[label] = weights[label]
    
    # One-Hot encoding
    labels = keras.utils.to_categorical(labels)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=100, validation_split=0.15, class_weight=weightDict,
                      callbacks=[tensorboard_callback])


    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Get Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 8), dpi=500)  # 8x8 inches, 300 DPI

    # CM = confusion_matrix(ytrue,ypred, normalize='all')
    disp = ConfusionMatrixDisplay.from_predictions(y_true=ytrue,
                      y_pred=ypred,normalize='true',
                      include_values=True,
                      display_labels=np.unique(ytrue),
                      text_kw = {'fontsize': 2},
                      ax = ax,
                      values_format = '.1%')

    # (confusion_matrix=CM, display_labels=np.unique(ytrue))
    disp.ax_.tick_params(labelsize=5)
    disp.ax_.set_xlabel('Predicted label', fontsize=10)
    disp.ax_.set_ylabel('True label', fontsize=10)
    # disp.plot()
    plt.savefig('confusion_matrix.png', dpi=500, bbox_inches='tight')
    plt.show()

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


    

def preprocess(images):
    """
    Adjusts luma in images: increases luma in dark images, decreases in overly bright images.
    Uses TensorFlow operations for efficiency.
    Returns processed images as a numpy array.
    """
    images_tf = tf.convert_to_tensor(images, dtype=tf.float32)
    # Convert RGB to YUV to access luma (Y channel)
    yuv = tf.image.rgb_to_yuv(images_tf / 255.0)
    y = yuv[..., 0]
    # Calculate mean luma per image
    mean_luma = tf.reduce_mean(y, axis=[1,2])
    # Define thresholds for dark and bright images
    dark_thresh = 0.35
    bright_thresh = 0.75
    # Increase luma for dark images
    y = tf.where(mean_luma[:, None, None] < dark_thresh, y * 1.25, y)
    # Decrease luma for bright images
    y = tf.where(mean_luma[:, None, None] > bright_thresh, y * 0.75, y)
    # Clip luma to [0,1]
    y = tf.clip_by_value(y, 0.0, 1.0)
    # Replace Y channel and convert back to RGB
    yuv = tf.concat([y[..., None], yuv[..., 1:]], axis=-1)
    rgb = tf.image.yuv_to_rgb(yuv)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    rgb = tf.cast(rgb * 255.0, tf.uint8)

    termcolor.cprint("Preprocessing Done Successfully","light_green")
    return rgb.numpy()


def load_category(category_data):
    category, data_dir = category_data
    path = os.path.join(data_dir, str(category))
    imgs, lbls = [], []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        imgs.append(img)
        lbls.append(category)
    return imgs, lbls

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    import multiprocessing

    with multiprocessing.Pool() as pool:
        results = pool.map(load_category, [(category, data_dir) for category in range(NUM_CATEGORIES)])

    images, labels = [], []
    for imgs, lbls in results:
        images.extend(imgs)
        labels.extend(lbls)

    termcolor.cprint("Dataset Loaded Successfully","light_green")
    return (images, labels)


def Inception(x, f1b1, f3b3_red, f3b3, f5b5_red, f5b5, fmax_pool_red):
    # 1x1 conv branch
    branch1 = Conv2D(f1b1, (1,1), padding='same', activation='relu')(x)

    # 1x1 -> 3x3 conv branch
    branch2 = Conv2D(f3b3_red, (1,1), padding='same', activation='relu')(x)
    branch2 = Conv2D(f3b3, (3,3), padding='same', activation='relu')(branch2)

    # 1x1 -> 5x5 conv branch
    branch3 = Conv2D(f5b5_red, (1,1), padding='same', activation='relu')(x)
    branch3 = Conv2D(f5b5, (5,5), padding='same', activation='relu')(branch3)

    # MaxPool -> 1x1 conv branch
    branch4 = MaxPooling2D((3,3), strides=1, padding='same')(x)
    branch4 = Conv2D(fmax_pool_red, (1,1), padding='same', activation='relu')(branch4)

    # Concatenate all branches
    output = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    return output

# -----------------------------
# Full Model Definition
# -----------------------------
def get_model(IMG_WIDTH=30, IMG_HEIGHT=30, NUM_CLASSES=43):
    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Initial conv
    x = Conv2D(32, (3,3), padding='same', use_bias=True)(input)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=(3,3), padding='same')(x)

    # Inception Block 1 (⚡ reduced filters for speed)
    x = Inception(x, f1b1=16, f3b3_red=8, f3b3=64, f5b5_red=8, f5b5=32, fmax_pool_red=8)
    x = AveragePooling2D(pool_size=(2,2), padding='same')(x)

    # Inception Block 2 (moderate filters)
    x = Inception(x, f1b1=32, f3b3_red=32, f3b3=64, f5b5_red=16, f5b5=32, fmax_pool_red=32)
    x = AveragePooling2D(pool_size=(2,2), padding='same')(x)

    # Inception Block 3 (smaller)
    x = Inception(x, f1b1=32, f3b3_red=16, f3b3=32, f5b5_red=8, f5b5=16, fmax_pool_red=16)

    # Global pooling
    x = GlobalMaxPooling2D()(x)

    # Dense + Dropout
    x = Dense(64, activation='relu', use_bias=True)(x)
    x = Dropout(0.4)(x)

    # Output layer
    output = Dense(NUM_CLASSES, activation='softmax', use_bias=True)(x)

    model = Model(input, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.000125),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 'precision',
                 'recall',
        tf.keras.metrics.F1Score(
            average="macro"
        ),
      tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_acc')

],
    )
    return model


if __name__ == "__main__":
    main()