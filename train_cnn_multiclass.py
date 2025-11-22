import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import create_dataloaders
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import argparse
import glob
from PIL import Image
import torch.nn.functional as F



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


def unfreeze_layers(model, epoch, schedule):
    """
    Unfreezes CNN layers based on the current epoch.
    schedule: Dict {epoch_number: number_of_layers_to_unfreeze}
    """

    if isinstance(model, nn.DataParallel):
        model = model.module

    if epoch in schedule:
        num_layers_to_unfreeze = schedule[epoch]
        cnn_layers = list(model.children())  # Get CNN layers
        
        for i in range(num_layers_to_unfreeze):
            for param in model.resnet.encoder.stages[-(i+1)].parameters():
                param.requires_grad = True
        print(f"Unfroze {num_layers_to_unfreeze} CNN layers at epoch {epoch}")

# Example unfreezing schedule
unfreeze_schedule = {5: 1, 10: 2, 15: 3, 20:4}  # Unfreeze 1 layer at epoch 5, 2 layers at 10, etc.  


def calculate_metrics(all_preds, all_targets, num_classes):
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # class_mapping = {2: 0, 1: 1, 0: 2}  # Map original classes to new order
    
    reversed_preds =all_preds
    reversed_targets =all_targets

    # reversed_preds = np.vectorize(class_mapping.get)(all_preds)
    # reversed_targets = np.vectorize(class_mapping.get)(all_targets)
  
    
    precision = precision_score(reversed_targets, reversed_preds, average='macro', zero_division=1)
    recall = recall_score(reversed_targets, reversed_preds, average='macro', zero_division=1)
    f1 = f1_score(reversed_targets, reversed_preds, average='macro')
    bal_accuracy = balanced_accuracy_score(reversed_targets, reversed_preds)

    # Specificity
    cm = confusion_matrix(reversed_targets, reversed_preds, labels=list(range(num_classes)))

    # tn, fp, fn, tp = cm.ravel()
    
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))  # True Negatives
    fp = cm.sum(axis=0) - np.diag(cm)  # False Positives

    specificity_per_class = tn / (tn + fp)
    specificity = np.mean(specificity_per_class)
    # specificity = tn / (tn + fp)
    
    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(reversed_targets, reversed_preds)

    # --- AUC calculation for multiclass 

    try:
        # Convert labels to one-hot encoding
        y_true_one_hot = label_binarize(reversed_targets, classes=np.arange(num_classes))
        y_pred_one_hot = label_binarize(reversed_preds, classes=np.arange(num_classes))

        # Compute multi-class AUC (OvR strategy)
        auc_per_class = roc_auc_score(y_true_one_hot, y_pred_one_hot, multi_class="ovr", average=None)
    
        # Compute macro-average AUC
        auc_macro = np.mean(auc_per_class)

    except ValueError:
        auc_per_class = None  # Handle cases where AUC can't be calculated
        auc_macro = None

    # # Area Under the Curve (AUC) Binary classification
    # try:
    #     auc = roc_auc_score(reversed_targets, reversed_preds)
    # except ValueError:
    #     auc = None  # Handle cases where AUC can't be calculated

    # -- Binary classification return statement:
    # return precision, recall, f1, bal_accuracy, specificity, cm, mcc, auc

    return precision, recall, f1, bal_accuracy, specificity, cm, mcc, auc_per_class, auc_macro

def voting(num_frames, MVD_TCR, MVD_NO_TCR, NORMAL, label, correct_sq, all_preds_sq, all_targets_sq, confidences, tcr_threshold):
    

        # Class vote counts: index 0 = Normal, 1 = MVD without TCR, 2 = MVD with TCR
    votes = [NORMAL, MVD_NO_TCR, MVD_TCR]

    # Get the index of the class with the most votes
    final_idx = np.argmax(votes)

    # Map index to class name
    class_names = ['Normal Heart', 'MVD without TCR', 'MVD with TCR']
    final_result = class_names[final_idx]
    
    # if NORMAL>MVD_TCR+MVD_NO_TCR:
    #     final_result = 'Normal Heart'
    #     final_idx = 2
    # elif MVD_TCR>= num_frames*tcr_threshold/100:
    #     final_result= 'MVD with TCR'
    #     final_idx=0
    # else:
    #     final_result = 'MVD without TCR'
    #     final_idx = 1

    # if MVD>= num_frames*0.5:
    #     final_result = 'MVD'
    #     final_idx = 0
    # else:
    #     final_result = 'Normal Heart'
    #     final_idx = 1

    # print('FInal Index: ',final_idx)
    all_preds_sq.append(final_idx)
    all_targets_sq.append([label])

    if final_idx == label:
        correct_sq+=1

    matching_frames = [f for f in confidences if f[2] == final_idx]

    if matching_frames:

        highest_confidence_frame = max(matching_frames, key=lambda x: x[0])

        highest_conf_index = confidences.index(highest_confidence_frame)

        top_5_frames = sorted(confidences, key=lambda x: x[0], reverse=True)[:5]

        # Get the neighboring frames (two before and two after)
        start_index = max(0, highest_conf_index - 2)  # Handle start of the list
        end_index = min(len(confidences), highest_conf_index + 3)  # Handle end of the list

        # Extract the neighboring frames
        neighboring_frames = confidences[start_index:highest_conf_index] + confidences[highest_conf_index+1:end_index]
         
    else: 
        highest_confidence_frame = (0, 0)  # Assign a default value (tuple)
        # highest_confidence_frame[0]==0, highest_confidence_frame[1]==0 
        neighboring_frames=[]
        top_5_frames=[]
    
    return MVD_TCR, MVD_NO_TCR, NORMAL, final_result, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, neighboring_frames, top_5_frames

def process_video(video_path, transform, label, sc_preds, sc_targets, sc_correct, sc_total, correct_sq, all_preds_sq, all_targets_sq):

    MVD_TCR=0
    MVD_NO_TCR=0
    NORMAL =0

    label = label
    correct_sq = correct_sq 
    all_preds_sq = all_preds_sq
    all_targets_sq =  all_targets_sq

    sc_preds = sc_preds
    sc_targets = sc_targets
    sc_correct= sc_correct
    sc_total=sc_total

    confidences = []

    frame_paths = sorted(glob.glob(os.path.join(video_path, '*.png')))  # Assuming frames are .png images

    num_frames = len(frame_paths)
    print(f'Number of Frames: {num_frames}')

    frame_predictions = []  
    j=0
    misclasssified=[]

    for frame_path in frame_paths:
        j += 1
        sc_total += 1

        frame = np.array(Image.open(frame_path))  # Open the frame
        augmented = transform(image=frame)
        transformed_frame = augmented["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = network(transformed_frame)
            logits = output.logits
            probabilities = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            predicted_class = output.logits.argmax(1)

            confidences.append((confidence.item(), j, prediction.item()))

            # sc_preds.append(predicted_class.tolist())
            # sc_targets.append([label])
            sc_targets.append(label)
            sc_preds.append(predicted_class.item())

            if predicted_class.item() == label:
                sc_correct += 1
            else:
                misclasssified.append(j)

            if predicted_class == 2:
                MVD_TCR += 1
            elif predicted_class == 1:
                MVD_NO_TCR += 1
            elif predicted_class==0:
                NORMAL+=1
            frame_predictions.append(predicted_class)

    if not frame_predictions:
        print(f"No predictions made for frames in {video_path}.")
        return None

    # Perform majority voting to classify the video
    MVD_TCR, MVD_NO_TCR, NORMAL, final_result, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, neighboring_frames, top_5_frames = voting(num_frames, MVD_TCR, MVD_NO_TCR, NORMAL, label, correct_sq, all_preds_sq, all_targets_sq, confidences, given_threshold)

    # video_predictions = majority_voting(frame_predictions)
    return MVD_TCR, MVD_NO_TCR, NORMAL, final_result, sc_preds, sc_targets, sc_correct, sc_total, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, neighboring_frames,  misclasssified,top_5_frames


def folder_testing(model_path):
    test_processor = AutoImageProcessor.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-Resnet50Model")
    test_network = AutoModelForImageClassification.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-Resnet50Model")

    test_network.classifier = nn.Sequential(nn.Flatten(start_dim=1),  # Flatten the tensor
        nn.Linear(in_features=2048, out_features=1024),  # Intermediate feedforward layer with 512 units
        nn.ReLU(),  # Non-linear activation
        nn.Dropout(p=0.3),  # Dropout for regularization (optional)
        nn.Linear(in_features=1024, out_features=64),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=64, out_features=3, bias=True)  # Final output layer for 3 classes
    )

    test_network.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    test_network = network.to(DEVICE)
    test_network.eval() 

    valTransform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(), 
    ])

    testing_all_preds = []
    testing_all_targets = []
    testing_correct, testing_total = 0, 0
    testing_correct_sq, testing_total_sq =0,0

    testing_all_preds_sq = []
    testing_all_targets_sq = []

    for video_path in glob.glob(FOLDER_TEST_DATASET):
        print(video_path, flush=True)
        testing_total_sq+=1

        file_name = os.path.split(video_path)[-1]
        print(file_name, flush=True)
        label = int(file_name.split('-')[0])

        testing_misclassified =[]

        testing_MVD_TCR, testing_MVD_NO_TCR, testing_NORMAL, testing_video_prediction, testing_all_preds, testing_all_targets, testing_correct, testing_total, testing_all_preds_sq, testing_all_targets_sq, testing_correct_sq, testing_highest_confidence_frame, testing_neighboring_frames, testing_misclassified, testing_top_5_frames = process_video(video_path, valTransform, label, testing_all_preds, testing_all_targets, testing_correct, testing_total, testing_correct_sq, testing_all_preds_sq, testing_all_targets_sq)

        if testing_video_prediction is not None:
            print(f"Predicted class for the video: {video_path} {testing_video_prediction}, while actual class = {label}", flush=True)
            print(f"Frames Classified as MVD WITHOUT RCT: {testing_MVD_NO_TCR} ", flush=True)
            print(f"Frames Classified as MVD WITH RCT: {testing_MVD_TCR} ", flush=True)
            print(f"Frames Classified as Normal: {testing_NORMAL} ", flush=True)

            # if testing_misclassified:
            #         print(f"Misclassified frames: ", flush=True)
            #         for index,frame in enumerate(testing_misclassified): 
            #             print(f"Frame {frame}", flush=True) 
            # else:
            #     print("No misclassified frames.", flush=True)

            print(f"Frame with highest confidence in the video: Frame {testing_highest_confidence_frame[1]} with confidence {testing_highest_confidence_frame[0]*100:.4f}%", flush=True)

            print(f"Following up:" )

            print("\nTop 5 Frames with Highest Confidence:")
            for rank, (conf, frame_num, pred) in enumerate(testing_top_5_frames, start=1):
                print(f"Rank {rank}: Frame {frame_num} with confidence {conf*100:.2f}%")

            # if testing_neighboring_frames:
            #     print("Neighboring frames with their confidences:", flush=True)
            #     for i, frame in enumerate(testing_neighboring_frames):
            #         print(f"Neighbor {i+1}: Frame {frame[1]} with confidence {frame[0]*100:.4f}%", flush=True)
            # else:
            #     print("No neighboring frames found.", flush=True)


    testing_accuracy_sq = 100.0 * testing_correct_sq / testing_total_sq
    testing_precision_sq, testing_recall_sq, testing_f1_sq, testing_bal_accuracy_sq, testing_specificity_sq, testing_cm_sq, testing_mcc_sq, testing_auc_per_class_sq, testing_auc_macro_sq = calculate_metrics(testing_all_preds_sq, testing_all_targets_sq, num_classes=3)

    print(f'TESTING Performance metrics for Sequel Classification for model:', flush=True)
    print(f"Accuracy: {testing_accuracy_sq:.2f}%", flush=True)
    print(f"Precision: {testing_precision_sq:.2f}", flush=True)
    print(f"Recall: {testing_recall_sq:.2f}", flush=True)
    print(f"F1-Score: {testing_f1_sq:.2f}", flush=True)
    print(f"Balanced Accuracy: {testing_bal_accuracy_sq:.2f}", flush=True)
    print(f"Specificity: {testing_specificity_sq:.2f}", flush=True)
    print(f"Confusion Matrix:\n{testing_cm_sq}", flush=True)
    print(f"Confusion Matrix:\n{testing_mcc_sq}", flush=True)
    print(f"Testing AUC macro:\n{testing_auc_macro_sq}", flush=True)

    print("Testing AUC per class:\n")

    if testing_auc_per_class_sq is not None:
        for i, auc in enumerate(testing_auc_per_class_sq):
            print(f"AUC for Class {i}: {auc:.4f}")

    print('--------------------------------', flush=True)

    print(f'TESTING Performance metrics for Frame Classification for model:', flush=True)

    testing_accuracy = 100.0 * testing_correct / testing_total
    testing_precision, testing_recall, testing_f1, testing_bal_accuracy, testing_specificity, testing_cm, testing_mcc, testing_auc_per_class, testing_auc_macro = calculate_metrics(testing_all_preds, testing_all_targets, num_classes=3)

    print(f'TESTING Performance metrics for Frame Classification for model', flush=True)
    print(f"Accuracy: {testing_accuracy:.2f}%", flush=True)
    print(f"Precision: {testing_precision:.2f}", flush=True)
    print(f"Recall: {testing_recall:.2f}", flush=True)
    print(f"F1-Score: {testing_f1:.2f}", flush=True)
    print(f"Balanced Accuracy: {testing_bal_accuracy:.2f}", flush=True)
    print(f"Specificity: {testing_specificity:.2f}", flush=True)
    print(f"Confusion Matrix:\n{testing_cm}", flush=True)
    print(f"Confusion Matrix:\n{testing_mcc}", flush=True)
    print(f"Testing AUC macro:\n{testing_auc_macro}", flush=True)

    print("Testing AUC per class:\n")

    if testing_auc_per_class is not None:
        for i, auc in enumerate(testing_auc_per_class):
            print(f"AUC for Class {i}: {auc:.4f}")

    print('--------------------------------', flush=True)


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
        ToTensorV2(),
    ])

    valTransform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Fine-tune ResNet50")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr_resnet", type=float, default=1e-5, help="Learning rate for ResNet layers")
    parser.add_argument("--lr_classifier", type=float, default=1e-3, help="Learning rate for classifier")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--sch_patience", type=int, default=4, help="Scheduler Patience")
    parser.add_argument("--tcr_threshold", type=int, default=10, help="MMVD with TCR threshold")

    args = parser.parse_args()

    # Access arguments
    BATCH_SIZE = args.batch_size
    LR_RESNET = args.lr_resnet
    LR_CLASSIFIER = args.lr_classifier
    num_epochs = args.num_epochs
    sch_patience = args.sch_patience
    given_threshold = args.tcr_threshold

    # Hyperparameters

    # num_epochs = 2
    # BATCH_SIZE = 16
    # LR_RESNET = 1e-5
    # LR_CLASSIFIER = 1e-3
    accumulation_steps = 1

    # Paths to your train and test datasets
    TRAIN_DATASET = '/home/f/fratzeska/E/dataset/train'
    TEST_DATASET = '/home/f/fratzeska/E/dataset/test'
    SEQUEL_SET = '/home/f/fratzeska/E/dataset/test 3class/*'
    FOLDER_TEST_DATASET= '/home/f/fratzeska/E/dataset/validation/validation 3class/*'

    # DATASET FOR TESTING
    #TRAIN_DATASET = '/home/f/fratzeska/E/dataset/test_base/train'
    #TEST_DATASET = '/home/f/fratzeska/E/dataset/test_base/test'
    #SEQUEL_SET = '/home/f/fratzeska/E/dataset/test_base/test 2class/*'
    #FOLDER_TEST_DATASET= '/home/f/fratzeska/E/dataset/test_base/val/*'

    # TRAIN_DATASET= "C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/Databse/test_base/train"
    # TEST_DATASET = "C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/Databse/test_base/test"
    # SEQUEL_SET = "C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/Databse/test_base/test 2class/*"
    # FOLDER_TEST_DATASET = "C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/Databse/test_base/val/*"

    train_dataset, _ = create_dataloaders.get_dataloader(TRAIN_DATASET, transforms=trainTransform, batchSize=BATCH_SIZE)
    print('Training classes order: ', train_dataset.classes)
    test_dataset, _ = create_dataloaders.get_dataloader(TEST_DATASET, transforms=valTransform, batchSize=BATCH_SIZE)
    print('Testing classes order: ', test_dataset.classes)

    trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, optimizer, and scheduler
    processor = AutoImageProcessor.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-Resnet50Model")
    network = AutoModelForImageClassification.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-Resnet50Model")

    for param in network.resnet.embedder.parameters():
        param.requires_grad=False
    for param in network.resnet.encoder.stages[0].parameters():
        param.requires_grad=False
    for param in network.resnet.encoder.stages[1].parameters():
        param.requires_grad=False
    for param in network.resnet.encoder.stages[2].parameters():
        param.requires_grad=False
    for param in network.resnet.encoder.stages[3].parameters():
        param.requires_grad=False            
    
    network.classifier = nn.Sequential(nn.Flatten(start_dim=1),  # Flatten the tensor
        nn.Linear(in_features=2048, out_features=1024),  # Intermediate feedforward layer with 512 units
        nn.ReLU(),  # Non-linear activation
        nn.Dropout(p=0.3),  # Dropout for regularization (optional)
        nn.Linear(in_features=1024, out_features=64),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=64, out_features=3, bias=True)  # Final output layer for 3 classes
    )

    for param in network.classifier.parameters():
        param.requires_grad = True

    resnet_parameters = []
    for name, param in network.named_parameters():
        if "encoder" in name and param.requires_grad:  # Parameters in the CNN
            resnet_parameters.append(param)

    classifier_parameters = []
    for name, param in network.named_parameters():
        if "classifier" in name:  # Parameters outside the CNN
            classifier_parameters.append(param)

    # optimizer = torch.optim.Adam([
    #     {"params": resnet_parameters, "lr": LR_RESNET},  # Lower learning rate for the CNN
    #     {"params": classifier_parameters, "lr": LR_CLASSIFIER}  # Higher learning rate for other parameters
    # ],
    # weight_decay=0.0001
    # )

    optimizer = torch.optim.Adam([
    {'params': network.resnet.encoder.stages[0].parameters(), 'lr': 5e-5},  # Freeze or minimal tuning
    {'params': network.resnet.encoder.stages[1].parameters(), 'lr': 1e-4},  # Slight fine-tuning
    {'params': network.resnet.encoder.stages[2].parameters(), 'lr': 5e-4},  # More adaptation
    {'params': network.resnet.encoder.stages[3].parameters(), 'lr': 1e-3},  # Closest to classifier
    {'params': network.classifier.parameters(), 'lr': 1e-3}  # Classifier needs largest LR
    ])

    # optimizer = torch.optim.SGD([
        # {"params": resnet_parameters, "lr": LR_RESNET},  # Lower learning rate for the CNN
        # {"params": classifier_parameters, "lr": LR_CLASSIFIER}  # Higher learning rate for other parameters], lr=..., momentum=0.9, weight_decay=0.01)
    # ],
    # weight_decay=0.01)


    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=sch_patience, min_lr=1e-6)
    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.BCEWithLogitsLoss()

    network = network.to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
        network = nn.DataParallel(network)


    print('Starting training')
    print('Number of epochs = ', num_epochs)
    print('Resnet Learning Rate = ', LR_RESNET)
    print('Classifier Learning Rate = ', LR_CLASSIFIER)
    print('Batch Size = ', BATCH_SIZE)
    print('Loss Function = ', loss_function)
    print('Optimizer = ', optimizer)
    print('Scheduler = ', scheduler)
    print('MMVD threshold = ', given_threshold)



    # Training loop
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}', datetime.now(), flush=True)

        unfreeze_layers(network, epoch, unfreeze_schedule)  # Unfreeze progressively

        current_loss = 0.0

        network.train()

        for i, (inputs, targets) in enumerate(trainLoader, 0):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # optimizer.zero_grad() if i % accumulation_steps == 0 else None

            # outputs = network(inputs)
            outputs = network(inputs, labels=targets)
            
            loss=outputs.loss
            if loss is None:
                raise ValueError("Model did not return a loss.")
            loss = loss.mean() if loss.ndim > 0 else loss

            # loss = loss_function(outputs.logits, targets) - auto einai lathos
            # loss = loss_function(outputs.logits[:, 1], targets.float())
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            current_loss += loss.item()
            # writer.add_scalar('Loss/train', current_loss, epoch)

            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500), flush=True)
                current_loss = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, current learning rate: {current_lr}", flush=True)

        # Validation phase
        print('Starting frame classification evaluation at ', datetime.now(), flush=True)
        correct, total = 0, 0
        all_preds = []
        all_targets = []
        network.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(testLoader, 0):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = network(inputs)
                preds = outputs.logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # loss = loss_function(outputs.logits, targets)
                loss = loss_function(outputs.logits[:, 1], targets.float())

                val_loss += loss.item()

                total += targets.size(0)
                correct += (outputs.logits.argmax(1) == targets).sum().item()

        val_loss /= len(testLoader)  # Average validation loss
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}", flush=True)

        scheduler.step(val_loss)

        # Calculate metrics
        accuracy = 100.0 * correct / total
        precision, recall, f1, bal_accuracy, specificity, cm, mcc, auc_per_class, auc_macro = calculate_metrics(all_preds, all_targets, num_classes=3)

        print(f'Performance metrics for Frame Classification for epoch {epoch + 1}:', flush=True)
        print(f"Accuracy: {accuracy:.2f}%", flush=True)
        print(f"Precision: {precision:.2f}", flush=True)
        print(f"Recall: {recall:.2f}", flush=True)
        print(f"F1-Score: {f1:.2f}", flush=True)
        print(f"Balanced Accuracy: {bal_accuracy:.2f}", flush=True)
        print(f"Specificity: {specificity:.2f}", flush=True)
        print(f"Confusion Matrix:\n{cm}", flush=True)
        print(f"Matthews Correlation Coefficient:\n{mcc}", flush=True)
        print(f"Area Under the Curve Macro:\n{auc_macro}", flush=True)

        print("Testing AUC per class:\n", flush=True)

        if auc_per_class is not None:
            for i, auc in enumerate(auc_per_class):
                print(f"AUC for Class {i}: {auc:.4f}", flush=True)

        
        print('--------------------------------', flush=True)

        print('Starting sequel classification evaluation at ', datetime.now(), flush=True)

        sc_preds = []
        sc_targets = []

        sc_correct, sc_total = 0, 0
        correct_sq, total_sq =0,0

        all_preds_sq = []
        all_targets_sq = []
        
        for video_path in glob.glob(SEQUEL_SET):
            print(video_path, flush=True)
            total_sq+=1

            file_name = os.path.split(video_path)[-1]
            print(file_name, flush=True)
            label = int(file_name.split('-')[0])

            sc_misclassified = []


            MVD_TCR, MVD_NO_TCR, NORMAL, video_prediction, sc_preds, sc_targets, sc_correct, sc_total, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, sc_neighboring_frames, sc_misclassified, top_5_frames = process_video(video_path, valTransform, label, sc_preds, sc_targets, sc_correct, sc_total, correct_sq, all_preds_sq, all_targets_sq)

            if video_prediction is not None:
                print(f"Predicted class for the video: {video_path} {video_prediction}, while actual class = {label}")
                print(f"Frames Classified as MVD with TCR: {MVD_TCR} ")
                print(f"Frames Classified as MVD without TCR: {MVD_NO_TCR} ")
                print(f"Frames Classified as Normal: {NORMAL} ")

                # if sc_misclassified:
                #     print(f"Misclassified frames: ", flush=True)
                #     for index,frame in enumerate(sc_misclassified): 
                #         print(f"Frame {frame}", flush=True) 
                # else:
                #     print("No misclassified frames.", flush=True)

                print(f"Frame with highest confidence in the video: Frame {highest_confidence_frame[1]} with confidence {highest_confidence_frame[0]*100:.4f}%", flush=True)

                if sc_neighboring_frames:
                    print("Neighboring frames with their confidences:")
                    for i, frame in enumerate(sc_neighboring_frames):
                        print(f"Neighbor {i+1}: Frame {frame[1]} with confidence {frame[0]*100:.4f}%")
                else:
                    print("No neighboring frames found.")

        accuracy_sq = 100.0 * correct_sq / total_sq
        precision_sq, recall_sq, f1_sq, bal_accuracy_sq, specificity_sq, cm_sq, mcc_sq, auc_per_class_sq, auc_macro_sq = calculate_metrics(all_preds_sq, all_targets_sq, num_classes=3)

        print(f'Performance metrics for Sequel Classification for epoch {epoch + 1}:', flush=True)
        print(f"Accuracy: {accuracy_sq:.2f}%", flush=True)
        print(f"Precision: {precision_sq:.2f}", flush=True)
        print(f"Recall: {recall_sq:.2f}", flush=True)
        print(f"F1-Score: {f1_sq:.2f}", flush=True)
        print(f"Balanced Accuracy: {bal_accuracy_sq:.2f}", flush=True)
        print(f"Specificity: {specificity_sq:.2f}", flush=True)
        print(f"Confusion Matrix:\n{cm_sq}", flush=True)
        print(f"Matthews Correlation Coefficient:\n{mcc_sq}", flush=True)
        print(f"Area Under the Curve macro:\n{auc_macro_sq}", flush=True)

        print("Testing AUC per class:\n", flush=True)

        if auc_per_class is not None:
            for i, auc in enumerate(auc_per_class):
                print(f"AUC for Class {i}: {auc:.4f}", flush=True)

    
        print('--------------------------------', flush=True)


        accuracy = 100.0 * sc_correct / sc_total
        sc_precision, sc_recall, sc_f1, sc_bal_accuracy, sc_specificity, sc_cm, sc_mcc, sc_auc_per_class, sc_auc_macro = calculate_metrics(sc_preds, sc_targets, num_classes=3)

        print(f'Performance metrics for Frame Classification for epoch {epoch + 1}:', flush=True)
        print(f"Accuracy: {accuracy:.2f}%", flush=True)
        print(f"Precision: {precision:.2f}", flush=True)
        print(f"Recall: {recall:.2f}", flush=True)
        print(f"F1-Score: {f1:.2f}", flush=True)
        print(f"Balanced Accuracy: {bal_accuracy:.2f}", flush=True)
        print(f"Specificity: {specificity:.2f}", flush=True)
        print(f"Confusion Matrix:\n{cm}", flush=True)
        print(f"Matthews Correlation Coefficient:\n{sc_mcc}", flush=True)
        print(f"Area Under the Curve macro:\n{sc_auc_macro}", flush=True)

        print("Testing AUC per class:\n", flush=True)

        if sc_auc_per_class is not None:
            for i, auc in enumerate(sc_auc_per_class):
                print(f"AUC for Class {i}: {auc:.4f}", flush=True)

        print('--------------------------------', flush=True)

        if (epoch+1)%10 ==0:
            interm_path ='./' + 'intermediate' + 'ep' + str(epoch+1) + '.pth' 
            print(f"Saving intermediate model: for epoch {epoch +1}", flush=True)
            if torch.cuda.device_count() > 1:
                torch.save(network.module.state_dict(), interm_path)
            else:
                torch.save(network.state_dict(), interm_path)

            print('STARTING INTERMEDIATE MODEL TESTING', datetime.now(), flush=True)

            folder_testing(interm_path)


    save_path ='./' + 'fine_tuned' + str(num_epochs) + 'ep' + str(LR_CLASSIFIER) + str(LR_RESNET)+ '.pth'
    if torch.cuda.device_count() > 1:
        torch.save(network.module.state_dict(), save_path)
    else:
        torch.save(network.state_dict(), save_path)
    print('Training complete at: ', datetime.now(), flush=True)
