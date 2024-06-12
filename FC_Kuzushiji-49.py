import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from torchsummary import summary


# 读取数据
X = np.load("C:\\Users\\lcf14\\Desktop\\k49-train-imgs.npz")['arr_0']
X_test = np.load('C:\\Users\\lcf14\\Desktop\\k49-test-imgs.npz')['arr_0']

Y = np.load('C:\\Users\\lcf14\\Desktop\\k49-train-labels.npz')['arr_0']
Y_test = np.load('C:\\Users\\lcf14\\Desktop\\k49-test-labels.npz')['arr_0']

# 划分训练集和验证集 （评估模型性能；避免过拟合；选择最佳模型）
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

# 归一化数据
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train = X_train.astype('float32')

X_test /= 255
X_val /= 255
X_train /= 255

# 将 NumPy 数组转换为 PyTorch 张量
X_test = torch.Tensor(X_test)
X_val = torch.Tensor(X_val)
X_train = torch.Tensor(X_train)

Y_test = torch.Tensor(Y_test)
Y_val = torch.Tensor(Y_val)
Y_train = torch.Tensor(Y_train)
print(X_train.shape)
# 将图像数据调整为 MLP 所需的一维形状 [样本数, 特征数（高*宽）]
IMG_ROWS = X_train.shape[1]
IMG_COLS = X_train.shape[2]
NUM_FEATURES = IMG_ROWS * IMG_COLS

X_train = X_train.reshape((X_train.shape[0], NUM_FEATURES))
X_val = X_val.reshape((X_val.shape[0], NUM_FEATURES))
X_test = X_test.reshape((X_test.shape[0], NUM_FEATURES))

# 将标签转换为 LongTensor（长整型）类型，因为交叉熵损失函数需要这种数据类型（相对熵是比值，交叉熵是差值）
Y_test = Y_test.type(torch.LongTensor)
Y_val = Y_val.type(torch.LongTensor)
Y_train = Y_train.type(torch.LongTensor)

# 创建数据集和数据加载器 （用TensorDataset将数据封装成数据集对象，过数据加载类DataLoader来批量加载数据）
batch_size = 80

train_dataset = TensorDataset(X_train, Y_train)
validation_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 定义多层感知机(全连接层)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 1024)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 256)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(256, 128)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(128, 49)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.drop1(x)

        x = self.act2(self.fc2(x))
        x = self.drop2(x)

        x = self.act3(self.fc3(x))
        x = self.drop3(x)

        x = self.act4(self.fc4(x))
        x = self.drop4(x)

        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


# 定义模型训练流程
n_epochs = 20
train_loss_list = []
valid_loss_list = []
test_loss_list = []
train_acc_list = []
valid_acc_list = []
test_acc_list = []
train_counter = []
train_losses = []

# 使用加权损失函数
class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

model = Net().cuda()

# 定义一个交叉熵损失函数
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
# 随机梯度下降优化器（设置了学习率和动量参数）
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


for epoch in range(n_epochs):
    print('EPOCH {}:'.format(epoch + 1))


    def training(data):
        model.train()
        for inputs, labels in data:
            inputs, labels = inputs.cuda(), labels.cuda()
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_counter.append((epoch - 1) * len(data.dataset))


    def validation(data):
        model.eval()
        acc = 0
        count = 0
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in data:
                inputs, labels = inputs.cuda(), labels.cuda()
                y_pred = model(inputs)
                val_loss += loss_fn(y_pred, labels).item()
                acc += (torch.argmax(y_pred, 1) == labels).float().sum()
                count += len(labels)
                val_loss /= len(data.dataset)
            acc /= count
        #valid_loss_list.append(val_loss)
        valid_acc_list.append(acc.item() * 100)
        print("Validation accuracy: %.3f%%" % (acc * 100))


    def test(data):
        model.eval()

        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data:
                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

            test_loss /= len(data.dataset)
            test_accuracy = 100. * test_correct / total
        #test_loss_list.append(test_loss)
        test_acc_list.append(test_accuracy)
        print("Test accuracy: %.3f%%" % (test_accuracy))

    training(train_dataloader)
    validation(validation_dataloader)
    test(test_dataloader)
print("Complete!")



# 绘制学习曲线
plt.figure(figsize=(10, 6))
"""
# 绘制训练与验证损失
plt.subplot(1, 2, 1)
plt.plot([float(v) for v in valid_loss_list], label='Validation Loss', color='darkorange')
plt.plot([float(t) for t in test_loss_list], label='Test Loss', color='green')
plt.legend()
plt.title('Loss Evolution')
"""
# 绘制训练与验证准确率
#plt.subplot(1, 1, 1)
plt.plot([float(v) for v in valid_acc_list], label='Validation Accuracy', color='darkorange')
plt.plot([float(t) for t in test_acc_list], label='Test Accuracy', color='green')
plt.legend()
plt.title('Accuracy Evolution')

plt.show()