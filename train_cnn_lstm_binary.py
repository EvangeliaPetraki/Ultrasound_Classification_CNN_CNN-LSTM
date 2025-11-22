import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
from datetime import datetime
# import cnn_lstm_dataloader
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
# import pydicom as dicom
# from pydicom.pixel_data_handlers import convert_color_space
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from torchsummary import summary
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import re
import cv2
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlbumentationsValDataset(Dataset):
    # def __init__(self, root_dir, transform=None, sequence_length=30, class_to_idx=None):
    def __init__(self, root_dir, transform=None, sequence_length=30):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        # self.class_to_idx = class_to_idx
        self.data = self._prepare_val_dataset()
    
    def _prepare_val_dataset(self):
        data = []
        for case_folder in os.listdir(self.root_dir):
            case_path = os.path.join(self.root_dir, case_folder)
            if not os.path.isdir(case_path):
                continue  # skip files

            try:
                class_label = int(case_folder[0])  # Get label from first character
            except ValueError:
                print(f"Skipping {case_folder}, cannot extract class label.")
                continue

            # class_id_prefix = case_folder[0]                    
            # if class_id_prefix not in ('0', '1'):
            #     print(f"Skipping {case_folder}, invalid prefix.")
            #     continue

            # try:
            #     class_names = list(self.class_to_idx.keys())
            #     class_label = self.class_to_idx[class_names[int(class_id_prefix)]]
            # except Exception as e:
            #     print(f"Skipping {case_folder}, cannot extract class label.")
            #     continue

            frames = sorted([
                os.path.join(case_path, fname)
                for fname in os.listdir(case_path)
                if fname.endswith('.png')
            ])

            if len(frames) >= self.sequence_length:
                for i in range(0, len(frames) - self.sequence_length + 1):
                    sequence = frames[i:i + self.sequence_length]
                    data.append((sequence, class_label))
            else:
                print(f"Skipping {case_folder}, not enough frames ({len(frames)} found).")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence_paths, label = self.data[idx]
        images = [cv2.imread(img_path) for img_path in sequence_paths]
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        if self.transform:
            images = [self.transform(image=img)['image'] for img in images]
        images = torch.stack(images)
        return images, label

class AlbumentationsDataset(Dataset):
    def __init__(self, folder_path, transform=None, included_classes=None):
        
        self.folder_path = folder_path
        self.transform = transform
        self.included_classes = included_classes
        self.data = self._prepare_dataset()
        self.class_to_idx, self.idx_to_class = self.get_class_mapping()


    def _prepare_dataset(self):
    # Create a dictionary to group frames by patient and class
        data_dict = {}
        
        # Iterate through files in the folder
        for class_name in os.listdir(self.folder_path):
            class_path = os.path.join(self.folder_path, class_name)
            if not os.path.isdir(class_path):
                continue

            if self.included_classes and class_name not in self.included_classes:
                continue

            for video_name in os.listdir(class_path):
                video_path = os.path.join(class_path, video_name)
                if not os.path.isdir(video_path):
                    continue

                frame_paths = sorted([os.path.join(video_path, frame) 
                                  for frame in os.listdir(video_path) if frame.endswith('.png')],
                                 key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

                if len(frame_paths) == 0:
                    continue

                data_dict[f"{class_name}_{video_name}"] = {
                'frames': frame_paths,
                'label': class_name
            }

        return list(data_dict.values())


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        patient_data = self.data[idx]
        frames = patient_data['frames']
        label = patient_data['label']
        
        # Load and transform frames
        processed_frames = []
        for frame_path in frames:
            frame = cv2.imread(frame_path)  # Reads the image as a numpy array

            if self.transform:
                frame = self.transform(image=frame)['image']
            processed_frames.append(frame)
        
        # Convert to tensor
        frames_tensor = torch.stack(processed_frames)
        
        # Create label tensor
        label_tensor = torch.tensor(self.class_to_idx[label])
        
        return frames_tensor, label_tensor
    

    def get_class_mapping(self):
        unique_classes = set(data['label'] for data in self.data)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        return self.class_to_idx, self.idx_to_class



class CNN_LSTM(nn.Module):
    def __init__(self, base_model , hidden_size, num_classes, lstm_layers):
        super(CNN_LSTM, self).__init__()

        # self.initial_layers = nn.Sequential(
        #     base_model.resnet.embedder.embedder.convolution,
        #     base_model.resnet.embedder.embedder.normalization,
        #     base_model.resnet.embedder.embedder.activation
        # )

        self.cnn = base_model
        self.cnn_features = base_model.resnet

        for param in self.cnn_features.parameters():
            param.requires_grad = False
        
        
        # for param in self.cnn.resnet.embedder.parameters():
        #     param.requires_grad=False
    
        # for param in self.cnn.resnet.encoder.stages[0].parameters():
        #     param.requires_grad=False
        # for param in self.cnn.resnet.encoder.stages[1].parameters():
        #     param.requires_grad=False
        # for param in self.cnn.resnet.encoder.stages[2].parameters():
        #     param.requires_grad=False
        for param in self.cnn.resnet.encoder.stages[3].parameters():
            param.requires_grad=True 

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (batch_size, channels, 1, 1)
        
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
                    nn.Linear(lstm_hidden_size *2, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    #nn.Linear(128, 64),  # Intermediate layer
                    #nn.ReLU(),
                    #nn.Dropout(0.5),    
                    nn.Linear(64, num_classes)
                )
        
    def forward(self, x):
        # Pass input through the pretrained feature extractor
        batch_size, num_frames, channels, height, width = x.shape

        x = x.view(-1, channels, height, width)

        cnn_output = self.cnn_features(x)
        x = cnn_output.last_hidden_state
        
        x = self.global_pool(x)  # Shape: (batch_size * num_frames, channels, 1, 1)
        x = x.view(batch_size, num_frames, -1)  # Reshape for LSTM input


        # Pass through the LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        # last_time_step = lstm_out[:, -1, :]
        pooled_output = torch.mean(lstm_out, dim=1)
        
        # Pass through the classification head
        # output = self.classifier(last_time_step)
        output = self.classifier(pooled_output)

        
        return output

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False     

def calculate_metrics(all_preds, all_targets, num_classes):


    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

   # class_mapping = {1: 0, 0: 1}  # Map original classes to new order
    
    reversed_preds = np.vectorize(class_mapping.get)(all_preds)
    reversed_targets = np.vectorize(class_mapping.get)(all_targets)

    reversed_preds = np.array(all_preds)
    reversed_targets = np.array(all_targets)

    precision = precision_score(reversed_targets, reversed_preds, average='weighted', zero_division=1)
    recall = recall_score(reversed_targets, reversed_preds, average='weighted', zero_division=1)
    f1 = f1_score(reversed_targets, reversed_preds, average='weighted')
    bal_accuracy = balanced_accuracy_score(reversed_targets, reversed_preds)

    # Specificity
    cm = confusion_matrix(reversed_targets, reversed_preds, labels=list(range(num_classes)))
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))  # True Negatives
    fp = cm.sum(axis=0) - np.diag(cm)  # False Positives
    specificity_per_class = tn / (tn + fp)
    specificity = np.mean(specificity_per_class) 

    mcc = matthews_corrcoef(reversed_targets, reversed_preds)

    try:
        auc_macro = roc_auc_score(reversed_targets, reversed_preds)
        auc_per_class = [auc_macro, auc_macro]  # Duplicate for consistent return shape
    
    except ValueError:
        auc_per_class = None  # Handle cases where AUC can't be calculated
        auc_macro = None

    return precision, recall, f1, bal_accuracy, specificity, cm, mcc, auc_per_class, auc_macro

print('We start at: ', datetime.now(), flush=True)

if __name__ == '__main__':
    SEED = 42
    set_seed(SEED)

    # Augmentations for train and validation
    trainTransform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        # A.Affine(shear=20, p=0.5),
        A.Affine(translate_percent=(0.05, 0.03), p=0.5),
        A.Affine(scale=(0.8, 1.0), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()  # Convert to PyTorch tensor
    ])

    valTransform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()  # Convert to PyTorch tensor
    ])


    parser = argparse.ArgumentParser(description="Fine-tune ResNet50")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr_resnet", type=float, default=1e-5, help="Learning rate for ResNet layers")
    parser.add_argument("--lr_lstm", type=float, default=1e-4, help="Learning rate for LSTM layers")
    parser.add_argument("--lr_classifier", type=float, default=1e-3, help="Learning rate for classifier")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lstm_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--lstm_hidden_size", type=int, default=512, help="Number of LSTM layers")
    parser.add_argument("--sch_patience", type=int, default=4, help="Scheduler Patience")

    args = parser.parse_args()

    # Access arguments
    BATCH_SIZE = args.batch_size
    CNN_LR = args.lr_resnet
    CLASSIFIER_LR = args.lr_classifier
    LSTM_LR = args.lr_lstm

    num_epochs = args.num_epochs
    sch_patience = args.sch_patience
    lstm_layers = args.lstm_layers
    lstm_hidden_size = args.lstm_hidden_size


    # Hyperparameters
    # num_epochs = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # BATCH_SIZE = 2
    # CNN_LR = 1e-5
    # LSTM_LR = 1e-4
    # CLASSIFIER_LR = 1e-3
    accumulation_steps = 1
    # lstm_hidden_size = 512
    # lstm_layers=1
    # sch_patience = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to your train and test datasets
    TRAIN_DATASET = '/home/f/fratzeska/E/LSTM/dataset/train'
    TEST_DATASET = '/home/f/fratzeska/E/LSTM/dataset/test'
    # VAL_DATASET = '/home/f/fratzeska/E/LSTM/dataset/val'

    # TRAIN_DATASET = '/home/f/fratzeska/E/LSTM/test_base_lstm/train'
    # TEST_DATASET = '/home/f/fratzeska/E/LSTM/test_base_lstm/test'
    VAL_DATASET = '/home/f/fratzeska/E/LSTM/test_base_lstm/val'
    # TRAIN_DATASET = 'C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/Databse/test_base_lstm/train'
    # TEST_DATASET =  'C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/Databse/test_base_lstm/test'
    # VAL_DATASET = 'C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/Databse/test_base_lstm/val'

    # TRAIN_DATASET = 'C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/project/.venv/Scripts/Transfer/database/train2'
    # TEST_DATASET = 'C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/project/.venv/Scripts/Transfer/database/test2'

    # TEST_DATASET = 'C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/Databse/Final Database_30/test'
    # TRAIN_DATASET = 'C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/Databse/Final Database_30/train'

    SEQUENCE_LENGTH = 30 

    included_classes = ["Normal Heart", "MVD with TCR"]

    # Create dataset loaders (assuming create_dataloaders handles splitting correctly)
    train_dataset = AlbumentationsDataset(TRAIN_DATASET, transform=trainTransform, included_classes=included_classes)
    # print('Training classes order: ', train_dataset.classes)

    test_dataset = AlbumentationsDataset(TEST_DATASET, transform=valTransform, included_classes=included_classes)
    # print('Testing classes order: ', test_dataset.classes)
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_to_idx)
    assert train_dataset.class_to_idx == test_dataset.class_to_idx
    val_dataset = AlbumentationsValDataset(VAL_DATASET, transform=valTransform)
    # val_dataset = AlbumentationsValDataset(VAL_DATASET, transform=valTransform, class_to_idx=class_to_idx)


    trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valLoader = DataLoader(val_dataset, batch_size= BATCH_SIZE, shuffle=False)


    print("Train class_to_idx:", train_dataset.class_to_idx)
    print("Val class_to_idx:  ", getattr(val_dataset, 'class_to_idx', 'N/A'))

    # Initialize model, loss function, optimizer, and scheduler
    processor = AutoImageProcessor.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-Resnet50Model")
    network_backbone = AutoModelForImageClassification.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-Resnet50Model")

    network_backbone.classifier = nn.Sequential(nn.Flatten(start_dim=1),  # Flatten the tensor
        nn.Linear(in_features=2048, out_features=1024),  # Intermediate feedforward layer with 512 units
        nn.ReLU(),  # Non-linear activation
        nn.Dropout(p=0.3),  # Dropout for regularization (optional)
        nn.Linear(in_features=1024, out_features=64),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=64, out_features=2, bias=True)  # Final output layer for 3 classes
    )

    network_backbone.load_state_dict(torch.load("/home/f/fratzeska/E/LSTM/H0-intermediateep25.pth", map_location=torch.device(device))) 


    # network_backbone.load_state_dict(torch.load("C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/experiments/2class/Hall of Fame/H0/H0-intermediateep25.pth", map_location=torch.device('cpu')))

    model = CNN_LSTM(network_backbone, lstm_hidden_size, num_classes, lstm_layers)

    cnn_parameters = []

    for name, param in model.named_parameters():
        if "encoder" in name and param.requires_grad:  # Parameters in the CNN
            cnn_parameters.append(param)

    lstm_parameters = []

    for name, param in model.named_parameters():
        if "lstm" in name:  # Parameters outside the CNN
            lstm_parameters.append(param)
    
    classifier_parameters =[]

    for name, param in model.named_parameters():
        if "classifier" in name:  # Parameters outside the CNN
            classifier_parameters.append(param)

    # optimizer = torch.optim.Adam([
    #     {"params": cnn_parameters, "lr": CNN_LR},  # Lower learning rate for the CNN
    #     {"params": lstm_parameters, "lr": LSTM_LR}, 
    #     {"params": classifier_parameters, "lr": CLASSIFIER_LR } # Higher learning rate for other parameters
    # ],
    # weight_decay=0.0001
    # )

    optimizer = torch.optim.Adam([
        {"params": lstm_parameters, "lr": LSTM_LR}, 
        {"params": classifier_parameters, "lr": CLASSIFIER_LR } # Higher learning rate for other parameters
    ],
    weight_decay=0.0001
    )

    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience =sch_patience, min_lr= 1e-7)
    loss_function = nn.CrossEntropyLoss()

    model= model.to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)

    print('Starting training', flush=True)
    print('Number of epochs = ', num_epochs, flush=True)
    print('CNN Learning Rate = ', CNN_LR, flush=True)
    print('Classifier Learning Rate = ', CLASSIFIER_LR, flush=True)
    print('LSTM Learning Rate = ', LSTM_LR, flush=True)
    print('Batch Size = ', BATCH_SIZE, flush=True)
    print('Loss Function = ', loss_function, flush=True)
    print('Optimizer = ', optimizer, flush=True)
    print('Scheduler = ', scheduler, flush=True)
    print('Scheduler Patience = ', sch_patience, flush=True)
    print('LSTM hidden size = ', lstm_hidden_size, flush=True)
    print('LSTM layers = ', lstm_layers, flush=True)
    print('Accumulation Steps = ', accumulation_steps, flush=True)


    # Training loop
    for epoch in range(num_epochs):

        print(f'Starting epoch {epoch + 1}', datetime.now(), flush=True)  
        # unfreeze_layers(model, epoch, unfreeze_schedule)  # Unfreeze progressively

        model.train()
        train_loss = 0.0
        all_preds = []
        all_targets = []

        for inputs, targets in trainLoader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss = loss / accumulation_steps
            loss.backward()

            optimizer.step()

            # if (i + 1) % accumulation_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()

            
            train_loss += loss.item()
            # writer.add_scalar('Loss/train', current_loss, epoch)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        # if (i + 1) % accumulation_steps != 0:
        #     optimizer.step()
        #     optimizer.zero_grad()
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, current learning rate: {current_lr}", flush=True)

        # Validation phase
        print('Starting evaluation at ', datetime.now(), flush=True)
        correct, total = 0, 0

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in testLoader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(inputs)
                valpreds = outputs.argmax(1)
                val_preds.extend(valpreds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

                loss = loss_function(outputs, targets)
                val_loss += loss.item()

                total += targets.size(0)
                correct += (outputs.argmax(1) == targets).sum().item()

        val_loss /= len(testLoader)  # Average validation loss
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}", flush=True)

        scheduler.step(val_loss)

        # Calculate metrics
        accuracy = 100.0 * correct / total
        precision, recall, f1, bal_accuracy, specificity, cm, mcc, auc_per_class, auc_macro = calculate_metrics(val_preds, val_targets, num_classes)

        print(f'Performance metrics for Frame Classification for epoch {epoch + 1}:', flush=True)
        print(f"Accuracy: {accuracy:.2f}%", flush=True)
        print(f"Precision: {precision:.2f}", flush=True)
        print(f"Recall: {recall:.2f}", flush=True)
        print(f"F1-Score: {f1:.2f}", flush=True)
        print(f"Balanced Accuracy: {bal_accuracy:.2f}", flush=True)
        print(f"Specificity: {specificity:.2f}", flush=True)
        print(f"Matthews Correlation Coefficient: {mcc:.2f}", flush=True)
        print(f"AUC macro: {auc_macro:.2f}", flush=True)

        print("Testing AUC per class:\n", flush=True)

        if auc_per_class is not None:
            for i, auc in enumerate(auc_per_class):
                print(f"AUC for Class {i}: {auc:.4f}", flush=True)

        print('--------------------------------', flush=True)

        print(f"Confusion Matrix:\n{cm}", flush=True)
        print('--------------------------------', flush=True)

        if (epoch+1)%5 ==0:
            interm_path ='./' + 'intermediate' + 'ep' + str(epoch+1) + '.pth' 
            print(f"Saving intermediate model: for epoch {epoch +1}", flush=True)
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), interm_path)
            else:
                torch.save(model.state_dict(), interm_path)

            print('STARTING INTERMEDIATE MODEL TESTING', datetime.now(), flush=True)

            test_correct, test_total = 0, 0

            model.eval()
            testing_loss = 0.0
            testing_preds = []
            testing_targets = []

            with torch.no_grad():
                for inputs, targets in valLoader:
                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)
                    outputs = model(inputs)
                    testpreds = outputs.argmax(1)
                    testing_preds.extend(testpreds.cpu().numpy())
                    testing_targets.extend(targets.cpu().numpy())

                    loss = loss_function(outputs, targets)
                    testing_loss += loss.item()

                    test_total += targets.size(0)
                    test_correct += (outputs.argmax(1) == targets).sum().item()

            testing_loss /= len(valLoader)  # Average validation loss
            print(f"Epoch {epoch+1}, Validation Loss: {testing_loss:.4f}", flush=True)

            # Calculate metrics
            testing_accuracy = 100.0 * test_correct / test_total
            test_precision, test_recall, test_f1, test_bal_accuracy, test_specificity, test_cm, test_mcc, test_auc_per_class, test_auc_macro = calculate_metrics(testing_preds, testing_targets, num_classes)

            print(f'Performance metrics for Frame Classification for epoch {epoch + 1}:', flush=True)
            print(f"Accuracy: {testing_accuracy:.2f}%", flush=True)
            print(f"Precision: {test_precision:.2f}", flush=True)
            print(f"Recall: {test_recall:.2f}", flush=True)
            print(f"F1-Score: {test_f1:.2f}", flush=True)
            print(f"Balanced Accuracy: {test_bal_accuracy:.2f}", flush=True)
            print(f"Specificity: {test_specificity:.2f}", flush=True)
            print(f"Matthews Correlation Coefficient: {test_mcc:.2f}", flush=True)
            if test_auc_macro is not None:
                print(f"AUC macro: {test_auc_macro:.2f}", flush=True)
            else:
                print("AUC macro: Not available (probably only one class in y_true or y_pred)", flush=True)
            # print(f"AUC macro: {test_auc_macro:.2f}", flush=True)

            print("Testing AUC per class:\n", flush=True)

            if auc_per_class is not None:
                for i, auc in enumerate(test_auc_per_class):
                    print(f"AUC for Class {i}: {auc:.4f}", flush=True)

            print('--------------------------------', flush=True)

            print(f"Confusion Matrix:\n{test_cm}", flush=True)
            print('--------------------------------', flush=True)

            

    # save_path ='./' + str(num_epochs) + 'ep' + str(CNN_LR) + str(LSTM_LR) + '.pth'
    # if torch.cuda.device_count() > 1:
    #     torch.save(model.module.state_dict(), save_path)
    # else:
    #     torch.save(model.state_dict(), save_path)

    print('Training complete at: ', datetime.now(), flush=True)
