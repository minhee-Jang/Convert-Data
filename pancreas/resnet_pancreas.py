import torch
import torch.nn as nn
import torch.optim as optim
#from torchvision.models import resnet50
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from resnet import resnet18, resnet50, resnet101
# 모델 인스턴스 생성

def get_transforms(img_size):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return {
        'train': train_transforms,
        'val': val_transforms,
    }


def trainandtest():
    
    
    data_transforms = get_transforms(img_size)
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])

    print(len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 데이터로더 설정
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet101(3)
    model = model.to(device)

    # model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = val_dataloader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(inputs.shape)

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

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            
            print(f'{epoch+1} / {num_epochs} epoch : {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 최고 정확도를 갖는 모델로 설정
    model.load_state_dict(best_model_wts)

    # # 모델 저장
    torch.save(model.state_dict(), "pancreatic_cancer_resnet_selected.pth")

    # 테스트 데이터로 예측 수행
    model.load_state_dict(torch.load("pancreatic_cancer_resnet_selected.pth"))
    model.eval()

    test_data_transforms = transforms.Compose([
        transforms.Resize(img_size),
        #transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_data_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_corrects = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_corrects += torch.sum(preds == labels)

    test_acc = test_corrects.double() / test_total
    print(f'Test Accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    data_dir = "D:/data/diffusion/current_pancreas/pancreas_final_filtered_crop"

    batch_size = 64
    img_size = 128
    num_classes = 3
    num_epochs = 100

    trainandtest()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define your model here, e.g., model = resnet50(3)
    # model = resnet50(3)  # Assuming you have a resnet50 architecture
    # model = model.to(device)

    # # Specify the path to the saved model
    # model_path = "pancreatic_cancer_resnet_selected.pth"

    # # Load and test the model
    # load_and_test_model(model, model_path, device)

    # #test()
