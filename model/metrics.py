"""Métricas de evaluación"""
from math import floor
from numpy import expand_dims, log, mean, std, exp, asarray
from keras.applications.inception_v3 import preprocess_input
from keras.layers import MaxPooling2D, Flatten, Conv2D, Dense, Dropout
from keras.regularizers import l2
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from utils import load_bin, common


interfaces = load_bin.load_data('images_train')
labels = load_bin.load_data('labels_train')
normal_labels = common.LABELS_LIST
cgan_images = load_bin.load_data('images_generated')


def metrics_model():
    """Modelo personalizado para las métricas"""
    conv1 = Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3))
    conv2 = Conv2D(128, (3, 3), activation='relu',
                   kernel_regularizer=l2(l=0.01))
    conv3 = Conv2D(256, (3, 3), activation='relu',
                   kernel_regularizer=l2(l=0.01))
    max_pool_2 = MaxPooling2D((2, 2))
    max_pool_3 = MaxPooling2D((2, 2))
    flat_layer = Flatten()
    fc = Dense(256, activation='relu')
    output = Dense(20, 'softmax')
    drop_2 = Dropout(0.5)
    drop_3 = Dropout(0.5)

    new_model = Sequential()
    new_model.add(conv1)
    new_model.add(conv2)
    new_model.add(max_pool_2)
    new_model.add(drop_2)
    new_model.add(conv3)
    new_model.add(max_pool_3)
    new_model.add(drop_3)
    new_model.add(flat_layer)
    new_model.add(fc)
    new_model.add(output)

    new_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    return new_model


def train_metrics_model():
    """Entrenamiento del modelo para las metricas"""
    new_model = metrics_model()
    history = new_model.fit(interfaces, labels, epochs=50, batch_size=16,
                            shuffle=True, validation_split=0.1)
    new_model.save(common.BACKUP_WEIGHTS + '/metrics.h5')
    return history


def plot_training_metrics():
    """
    Verificacion del progreso de entrenamiento / Accuracy y Loss
    """
    history = train_metrics_model()

    # ACCURACY
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # LOSS
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def deploy_model():
    """Deploy del modelo + todas las tareas previas"""
    plot_training_metrics()
    new_model = load_model(common.BACKUP_WEIGHTS + '/metrics.h5')
    gen_loss, gen_accuracy = new_model.evaluate(cgan_images, labels)
    print('Accuracy ->', gen_accuracy, 'Loss ->', gen_loss)


def scale_images(images, new_shape):
    """
    Escalador de imagenes para las metricas
    """
    images_list = []
    for image in images:
        new_image = image.reshape(new_shape)
        new_image = new_image.astype('float32')
        new_image = new_image / 255.0
        images_list.append(new_image)
    return asarray(images_list)


def calculate_inception_score(images, n_split=10, eps=1E-16):
    """
    Calculo de la IS
    """
    model = load_model(common.BACKUP_WEIGHTS + '/metrics.h5')
    scores = []
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = images[ix_start:ix_end]
        subset = scale_images(subset, (128, 128, 3))

        # [-1,1]
        subset = preprocess_input(subset)
        p_yx = model.predict(subset)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)

        avg_kl_d = mean(sum_kl_d)
        is_score = exp(avg_kl_d)
        scores.append(is_score)

    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std, scores
