import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, roc_auc_score, matthews_corrcoef
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

def calculate_metrics(all_preds, all_targets, num_classes):
    precision = precision_score(all_targets, all_preds, zero_division=1)
    recall = recall_score(all_targets, all_preds, zero_division=1)
    f1 = f1_score(all_targets, all_preds)
    bal_accuracy = balanced_accuracy_score(all_targets, all_preds)

    # Specificity
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    tn, fp, fn, tp = cm.ravel()
    # tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))  # True Negatives
    # fp = cm.sum(axis=0) - np.diag(cm)  # False Positives
    specificity = tn / (tn + fp)
    
    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(all_targets, all_preds)

    # Area Under the Curve (AUC)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = None  # Handle cases where AUC can't be calculated

    return precision, recall, f1, bal_accuracy, specificity, cm, mcc, auc

def voting(num_frames, MVD, NORMAL, label, correct_sq, all_preds_sq, all_targets_sq, confidences):
    if MVD>= num_frames*0.5:
        final_result = 'MVD'
        final_idx = 0
    else:
        final_result = 'Normal Heart'
        final_idx = 1

    # print('FInal Index: ',final_idx)
    all_preds_sq.append(final_idx)
    all_targets_sq.append([label])

    if final_idx == label:
        correct_sq+=1

    matching_frames = [f for f in confidences if f[2] == final_idx]
    if matching_frames:

        highest_confidence_frame = max(matching_frames, key=lambda x: x[0])

        highest_conf_index = confidences.index(highest_confidence_frame)

        # Get the neighboring frames (two before and two after)
        start_index = max(0, highest_conf_index - 2)  # Handle start of the list
        end_index = min(len(confidences), highest_conf_index + 3)  # Handle end of the list

        # Extract the neighboring frames
        neighboring_frames = confidences[start_index:highest_conf_index] + confidences[highest_conf_index+1:end_index]
         
    else: 
        highest_confidence_frame[0]==0, highest_confidence_frame[1]==0 
        neighboring_frames=[]
    
    return MVD, NORMAL, final_result, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, neighboring_frames

def process_video(video_path, transform, label, sc_preds, sc_targets, sc_correct, sc_total, correct_sq, all_preds_sq, all_targets_sq):

    MVD=0
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

            sc_preds.append(predicted_class.tolist())
            sc_targets.append([label])

            if predicted_class.item() == label:
                sc_correct += 1

            if predicted_class == 0:
                MVD += 1
            elif predicted_class == 1:
                NORMAL += 1
            frame_predictions.append(predicted_class)

    if not frame_predictions:
        print(f"No predictions made for frames in {video_path}.")
        return None

    # Perform majority voting to classify the video
    MVD, NORMAL, final_result, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, neighboring_frames = voting(num_frames, MVD, NORMAL, label, correct_sq, all_preds_sq, all_targets_sq, confidences)

    # video_predictions = majority_voting(frame_predictions)
    return MVD, NORMAL, final_result, sc_preds, sc_targets, sc_correct, sc_total, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, neighboring_frames


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
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--sch_patience", type=int, default=4, help="Scheduler Patience")
    args = parser.parse_args()

    # Access arguments
    BATCH_SIZE = args.batch_size
    LR_RESNET = args.lr_resnet
    LR_CLASSIFIER = args.lr_classifier
    num_epochs = args.num_epochs
    sch_patience = args.sch_patience
# Hyperparameters

    # num_epochs = 2
    # BATCH_SIZE = 16
    # LR_RESNET = 1e-5
    # LR_CLASSIFIER = 1e-3
    accumulation_steps = 1

    # Paths to your train and test datasets
    TRAIN_DATASET = '/home/f/fratzeska/E/dataset/whole_dataset/2class/train'
    TEST_DATASET = '/home/f/fratzeska/E/dataset/whole_dataset/2class/test'
    SEQUEL_SET = '/home/f/fratzeska/E/dataset/test 2class/*'

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
    # for param in network.resnet.encoder.stages[0].parameters():
        # param.requires_grad=False
    # for param in network.resnet.encoder.stages[1].parameters():
        # param.requires_grad=False
    # for param in network.resnet.encoder.stages[2].parameters():
        # param.requires_grad=False
    # for param in network.resnet.encoder.stages[3].parameters():
        # param.requires_grad=False            
    
    network.classifier = nn.Sequential(nn.Flatten(start_dim=1),  # Flatten the tensor
        nn.Linear(in_features=2048, out_features=1024),  # Intermediate feedforward layer with 512 units
        nn.ReLU(),  # Non-linear activation
        nn.Dropout(p=0.3),  # Dropout for regularization (optional)
        nn.Linear(in_features=1024, out_features=64),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=64, out_features=2, bias=True)  # Final output layer for 3 classes
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
    #    {"params": resnet_parameters, "lr": LR_RESNET},  # Lower learning rate for the CNN
     #   {"params": classifier_parameters, "lr": LR_CLASSIFIER}  # Higher learning rate for other parameters
    #],
    #weight_decay=0.0001
    #)

    optimizer = torch.optim.Adam([
        {'params': network.resnet.encoder.stages[0].parameters(), 'lr': 5e-6},  # Freeze or minimal tuning
        {'params': network.resnet.encoder.stages[1].parameters(), 'lr': 1e-5},  # Slight fine-tuning
        {'params': network.resnet.encoder.stages[2].parameters(), 'lr': 5e-5},  # More adaptation
        {'params': network.resnet.encoder.stages[3].parameters(), 'lr': 1e-4},  # Closest to classifier
        {'params': network.classifier.parameters(), 'lr': 1e-3}  # Classifier needs largest LR
    ])


    # optimizer = torch.optim.SGD([
        # {"params": resnet_parameters, "lr": LR_RESNET},  # Lower learning rate for the CNN
        # {"params": classifier_parameters, "lr": LR_CLASSIFIER}  # Higher learning rate for other parameters], lr=..., momentum=0.9, weight_decay=0.01)
    # ],
    # weight_decay=0.01)


    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=sch_patience)
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.BCEWithLogitsLoss()

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


    # Training loop
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}', datetime.now(), flush=True)
        current_loss = 0.0

        network.train()

        for i, (inputs, targets) in enumerate(trainLoader, 0):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad() if i % accumulation_steps == 0 else None

            outputs = network(inputs)
            # loss = loss_function(outputs.logits, targets)
            loss = loss_function(outputs.logits[:, 1], targets.float())
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
        precision, recall, f1, bal_accuracy, specificity, cm, mcc, auc = calculate_metrics(all_preds, all_targets, num_classes=2)

        print(f'Performance metrics for Frame Classification for epoch {epoch + 1}:', flush=True)
        print(f"Accuracy: {accuracy:.2f}%", flush=True)
        print(f"Precision: {precision:.2f}", flush=True)
        print(f"Recall: {recall:.2f}", flush=True)
        print(f"F1-Score: {f1:.2f}", flush=True)
        print(f"Balanced Accuracy: {bal_accuracy:.2f}", flush=True)
        print(f"Specificity: {specificity:.2f}", flush=True)
        print(f"Confusion Matrix:\n{cm}", flush=True)
        print(f"Matthews Correlation Coefficient:\n{mcc}", flush=True)
        print(f"Area Under the Curve:\n{auc}", flush=True)
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


            MVD, NORMAL, video_prediction, sc_preds, sc_targets, sc_correct, sc_total, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, neighboring_frames = process_video(video_path, valTransform, label, sc_preds, sc_targets, sc_correct, sc_total, correct_sq, all_preds_sq, all_targets_sq)

            if video_prediction is not None:
                print(f"Predicted class for the video: {video_path} {video_prediction}, while actual class = {label}")
                print(f"Frames Classified as MVD: {MVD} ")
                print(f"Frames Classified as Normal: {NORMAL} ")

                print(f"Frame with highest confidence in the video: Frame {highest_confidence_frame[1]} with confidence {highest_confidence_frame[0]*100:.4f}%")

                if neighboring_frames:
                    print("Neighboring frames with their confidences:")
                    for i, frame in enumerate(neighboring_frames):
                        print(f"Neighbor {i+1}: Frame {frame[1]} with confidence {frame[0]*100:.4f}%")
                else:
                    print("No neighboring frames found.")

        accuracy_sq = 100.0 * correct_sq / total_sq
        precision_sq, recall_sq, f1_sq, bal_accuracy_sq, specificity_sq, cm_sq, mcc_sq, auc_sq = calculate_metrics(all_preds_sq, all_targets_sq, num_classes=2)

        print(f'Performance metrics for Sequel Classification for epoch {epoch + 1}:', flush=True)
        print(f"Accuracy: {accuracy_sq:.2f}%", flush=True)
        print(f"Precision: {precision_sq:.2f}", flush=True)
        print(f"Recall: {recall_sq:.2f}", flush=True)
        print(f"F1-Score: {f1_sq:.2f}", flush=True)
        print(f"Balanced Accuracy: {bal_accuracy_sq:.2f}", flush=True)
        print(f"Specificity: {specificity_sq:.2f}", flush=True)
        print(f"Confusion Matrix:\n{cm_sq}", flush=True)
        print(f"Matthews Correlation Coefficient:\n{mcc_sq}", flush=True)
        print(f"Area Under the Curve:\n{auc_sq}", flush=True)
        print('--------------------------------', flush=True)


        accuracy = 100.0 * sc_correct / sc_total
        sc_precision, sc_recall, sc_f1, sc_bal_accuracy, sc_specificity, sc_cm, sc_mcc, sc_auc = calculate_metrics(sc_preds, sc_targets, num_classes=2)

        print(f'Performance metrics for Frame Classification for epoch {epoch + 1}:', flush=True)
        print(f"Accuracy: {accuracy:.2f}%", flush=True)
        print(f"Precision: {precision:.2f}", flush=True)
        print(f"Recall: {recall:.2f}", flush=True)
        print(f"F1-Score: {f1:.2f}", flush=True)
        print(f"Balanced Accuracy: {bal_accuracy:.2f}", flush=True)
        print(f"Specificity: {specificity:.2f}", flush=True)
        print(f"Confusion Matrix:\n{cm}", flush=True)
        print(f"Matthews Correlation Coefficient:\n{sc_mcc}", flush=True)
        print(f"Area Under the Curve:\n{sc_auc}", flush=True)
        print('--------------------------------', flush=True)
            
        if (epoch+1)%5 ==0:
            interm_path ='./' + 'intermediate' + 'ep' + str(epoch+1) + '.pth' 
            print(f"Saving intermediate model: for epoch {epoch +1}", flush=True)
            if torch.cuda.device_count() > 1:
                torch.save(network.module.state_dict(), interm_path)
            else:
                torch.save(network.state_dict(), interm_path)

    save_path ='./' + 'fine_tuned' + str(num_epochs) + 'ep' + str(LR_CLASSIFIER) + str(LR_RESNET)+ '.pth'
    if torch.cuda.device_count() > 1:
        torch.save(network.module.state_dict(), save_path)
    else:
        torch.save(network.state_dict(), save_path)
    print('Training complete at: ', datetime.now(), flush=True)
