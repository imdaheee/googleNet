
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.googlenet import GoogLeNet


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

   
    DATASET_ROOT = "POC_Dataset"

    train_dir = os.path.join(DATASET_ROOT, "Training")
    test_dir = os.path.join(DATASET_ROOT, "Testing")

    
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform_train,
    )
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transform_test,
    )
    num_classes = len(train_dataset.classes)
    print(f"[INFO] Classes: {train_dataset.classes}")
    print(f"[INFO] Number of classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,      
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
    )

    
    model = GoogLeNet(num_classes=num_classes, use_aux=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )

    num_epochs = 10  
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        print(f"\n[Epoch {epoch+1}/{num_epochs}] Training...")
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            if isinstance(outputs, tuple):
                main_logits, aux1_logits, aux2_logits = outputs
                loss_main = criterion(main_logits, labels)
                loss_aux1 = criterion(aux1_logits, labels)
                loss_aux2 = criterion(aux2_logits, labels)
                loss = loss_main + 0.3 * (loss_aux1 + loss_aux2)
                logits = main_logits
            else:
                loss = criterion(outputs, labels)
                logits = outputs

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(logits, dim=1)
            running_total += labels.size(0)
            running_correct += (preds == labels).sum().item()

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total * 100.0
        print(f"[Train] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

       
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)  

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * images.size(0)

                _, preds = torch.max(logits, dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total * 100.0
        print(f"[Val]   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

   
    save_path = "googlenet_poc.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[INFO] Trained model saved to: {save_path}")

if __name__ == "__main__":
    main()