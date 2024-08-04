import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, images, labels, class_to_idx, transform=None):
        self.images = images
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert the numpy array to a PIL image
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.class_to_idx[label], dtype=torch.long)

        return image, label


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


def create_model(base_model_name, num_classes, dropout_prob=0.5):
    if base_model_name == 'mobilenet_v2':
        # Load the MobileNet V2 model with default weights
        weights = MobileNet_V2_Weights.DEFAULT
        base_model = models.mobilenet_v2(weights=weights)
        # Modify the first convolutional layer to accept 1 channel instead of 3
        base_model.features[0][0] = nn.Conv2d(1, base_model.features[0][0].out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # Modify the classifier to include dropout
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(base_model.last_channel, num_classes)
        )
    elif base_model_name == 'resnet50':
        # Load the ResNet50 model with default weights
        weights = ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)
        # Modify the first convolutional layer to accept 1 channel instead of 3
        base_model.conv1 = nn.Conv2d(1, base_model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the fully connected layer to include dropout
        base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(base_model.fc.in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")

    return base_model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, dataloaders, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(confusion_matrix(all_labels, all_preds))

    # Compute and plot ROC curve and AUC
    lb = LabelBinarizer()
    lb.fit(all_labels)
    y_test_bin = lb.transform(all_labels)
    y_pred_bin = lb.transform(all_preds)
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for class {}'.format(i))
        plt.legend(loc="lower right")
        plt.show()


def main():
    # Load the data
    SW620_OPD = np.load('SW620_OPD.npy')
    SW480_OPD_1 = np.load('SW480_OPD_1.npy')
    SW480_OPD_2 = np.load('SW480_OPD_2.npy')
    SW480_OPD = np.concatenate((SW480_OPD_1, SW480_OPD_2), axis=0)
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

    # Create a class-to-index mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(image_arrays.keys())}

    # Split the data
    data_splits = split_data(image_arrays)

    # Define the input shape
    input_shape = (256, 256)

    # Define the number of classes
    num_classes = len(image_arrays)

    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_shape),
        transforms.Normalize([0.5], [0.5])
    ])

    # Prepare DataLoaders
    dataloaders = {}
    for phase in ['train', 'val', 'test']:
        images = []
        labels = []
        for class_label, splits in data_splits.items():
            images.extend(splits[phase])
            labels.extend([class_label] * len(splits[phase]))

        dataset = CustomDataset(images, labels, class_to_idx, transform=transform)
        dataloaders[phase] = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define the base models
    base_models = ['mobilenet_v2', 'resnet50']

    # Train the models
    criterion = nn.CrossEntropyLoss()
    histories = {}

    for model_name in base_models:
        model = create_model(model_name, num_classes, dropout_prob=0.5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)
        histories[model_name] = model

    # Evaluate the models
    for model_name, model in histories.items():
        print(f"Evaluating {model_name}")
        evaluate_model(model, dataloaders, list(image_arrays.keys()))


if __name__ == "__main__":
    main()
