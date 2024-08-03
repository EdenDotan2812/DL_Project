import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2, ResNet50, MobileNet
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def split_data(image_arrays, test_size=0.2, val_size=0.2):

    data_splits = {}

    for class_label, images in image_arrays.items():
        # Split the data into training and test sets
        X_train, X_test = train_test_split(images, test_size=test_size, random_state=42)

        # Split the training data into training and validation sets
        X_train, X_val = train_test_split(X_train, test_size=val_size, random_state=42)

        # Store the data splits in the dictionary
        data_splits[class_label] = {
            'train': X_train,
            'val': X_val,
            'test': X_test
        }

    return data_splits

def create_model(base_model, input_shape, num_classes):
    base_model = base_model(include_top=False, weights=None, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)  # Change to sigmoid for multi-label classification
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])  # Change to binary_crossentropy for multi-label classification
    return model

def plot_history(histories, metric):
    plt.figure(figsize=(12, 8))
    for model_name, history in histories.items():
        plt.plot(history.history[metric], label=f"{model_name} {metric.capitalize()}")
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.title(f'Model {metric.capitalize()}')
    plt.show()

def main():
    # Load the data
    SW620_OPD = np.load('SW620_OPD.npy')
    SW480_OPD = np.load('SW480_OPD.npy')
    Monocytes_OPD = np.load('MNC_OPD.npy')
    PBMC_OPD = np.load('PBMC_OPD.npy')
    Granulocytes_OPD = np.load('GRC_OPD.npy')

    # Create a dictionary to store the image arrays
    image_arrays = {
        'SW620': SW620_OPD,
        'SW480': SW480_OPD,
        'Monocytes': Monocytes_OPD,
        'PBMC': PBMC_OPD,
        'Granulocytes': Granulocytes_OPD
    }

    # Split the data
    data_splits = split_data(image_arrays)

    # Define the input shape
    input_shape = (256, 256, 1)

    # Define the number of classes
    num_classes = len(image_arrays)

    # Define the base models
    base_models = {
        'MobileNet': MobileNet,
        'ResNet50': ResNet50,
        'MobileNetV2': MobileNetV2
    }

    # Train the models
    histories = {}
    for model_name, data_split in base_models.items():
        model = create_model(data_split, input_shape, num_classes)
        history = model.fit(data_splits['SW620']['train'], validation_data=data_splits['SW620']['val'], epochs=10, callbacks=[EarlyStopping(patience=3)], verbose=2)
        histories[model_name] = history

        # Evaluate the model
        for class_label in data_splits.items():
            y_pred = model.predict(data_splits[class_label]['test'])
            y_pred = np.argmax(y_pred, axis=1)
            # Compute and print the classification report
            print(f"Classification Report for {model_name} on {class_label}:")
            print(classification_report(data_split['test'], y_pred))

            # Compute and print the confusion matrix
            print(f"Confusion Matrix for {model_name} on {class_label}:")
            print(confusion_matrix(data_split['test'], y_pred))

            # Compute and plot the ROC curve for each class
            lb = LabelBinarizer()
            lb.fit(data_split['test'])
            y_test_bin = lb.transform(data_split['test'])
            y_pred_bin = lb.transform(y_pred)
            n_classes = y_test_bin.shape[1]

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot the ROC curve
            for i in range(n_classes):
                plt.figure()
                plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve for {model_name} on {class_label}')
                plt.legend(loc="lower right")
                plt.show()
            # Plot the training history
    plot_history(histories, 'accuracy')
    plot_history(histories, 'loss')