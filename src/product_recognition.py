# product_recognition.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import argparse
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 224
LEARNING_RATE = 0.001
EPOCHS = 10


# Data Transforms
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# Load Dataset
def load_dataset(data_dir):
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[INFO] Loaded {len(train_dataset)} training and {len(val_dataset)} validation images.")

    return train_loader, val_loader, len(train_dataset.classes)

# Load Pretrained MobileNetV2 Model

def load_mobilenetv2(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(DEVICE)
    return model

# Train Function

def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, leave=True)
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_description(f"Training Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    return train_loss, train_acc


# Evaluate Function
def evaluate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc

# Save Model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


# Inference Function
def predict(model, class_names):
    model.eval()
    transform = get_transforms()[1]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        label = class_names[predicted.item()]
        cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Product Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Export to TorchScript for Mobile

def export_model(model, export_path):
    model.eval()
    example_input = torch.rand(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(export_path)
    print(f"[INFO] Model exported to {export_path}")


# Main Function
def main():
    parser = argparse.ArgumentParser(description="Product Recognition using MobileNetV2")
    parser.add_argument('--data', type=str, required=True, help="Path to dataset")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'export'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--export_path', type=str, default='product_recognition.pt')

    args = parser.parse_args()

    train_loader, val_loader, num_classes = load_dataset(args.data)
    model = load_mobilenetv2(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == 'train':
        for epoch in range(args.epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            print(f"Epoch {epoch + 1}/{args.epochs}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            save_model(model, "best_model.pth")

    elif args.mode == 'predict':
        model.load_state_dict(torch.load("best_model.pth"))
        class_names = train_loader.dataset.classes
        predict(model, class_names)

    elif args.mode == 'export':
        model.load_state_dict(torch.load("best_model.pth"))
        export_model(model, args.export_path)

if __name__ == "__main__":
    main()
