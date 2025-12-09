# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models.googlenet import GoogLeNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    DATASET_ROOT = "/Users/imdahee/Desktop/4-1/컴퓨터비전입문/googleNet/POC_Dataset"

    train_dir = os.path.join(DATASET_ROOT, "Training")
    test_dir = os.path.join(DATASET_ROOT, "Testing")

    # ------------------------------------
    # Transform 정의
    # ------------------------------------
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

   
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    num_classes = len(train_dataset.classes)
    print("[INFO] Classes:", train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    
    test_full = datasets.ImageFolder(test_dir, transform=transform_test)
    test_size = len(test_full)
    val_size = int(test_size * 0.8)  # 80% → validation
    test_size = test_size - val_size

    val_dataset, test_dataset = random_split(test_full, [val_size, test_size])

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    print(f"[INFO] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    
    model = GoogLeNet(num_classes=num_classes, use_aux=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4)

    num_epochs = 10

    # train
    best_val_acc = 0.0
    best_model_path = "best_googlenet.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0

        print(f"\n[Epoch {epoch+1}/{num_epochs}] Training...")

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            if isinstance(outputs, tuple):
                main_out, aux1, aux2 = outputs
                loss = (criterion(main_out, labels)
                        + 0.3 * criterion(aux1, labels)
                        + 0.3 * criterion(aux2, labels))
                logits = main_out
            else:
                loss = criterion(outputs, labels)
                logits = outputs

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(logits, dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total * 100.0
        print(f"[Train] Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

       
        # Validation
        val_loss, val_correct, val_total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                if isinstance(out, tuple): out = out[0]
                loss = criterion(out, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(out, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total * 100.0
        print(f"[Val]   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] New best model saved ({best_val_acc:.2f}%)")

    print("\n[INFO] Training Finished!")
    print(f"[INFO] Best Validation Acc: {best_val_acc:.2f}%")
    print(f"[INFO] Best model: {best_model_path}")

    
    print("\n[INFO] Testing on held-out dataset...")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_correct, test_total = 0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            if isinstance(out, tuple): out = out[0]
            _, preds = torch.max(out, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total * 100.0
    print(f"[TEST] Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
