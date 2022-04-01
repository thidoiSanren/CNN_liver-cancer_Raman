import torch.nn as nn
from model_vgg import vgg
import torch.optim as optim
from datasets import TrainSets, ValidateSets
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    traindata = TrainSets()
    validatadata = ValidateSets()
    batchsize=128
    train_loader = Data.DataLoader(dataset=traindata, batch_size=batchsize, drop_last=False, shuffle=True, num_workers=4)
    validata_loader = Data.DataLoader(dataset=validatadata, shuffle=True, drop_last=False, num_workers=0)
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_acc = 0.0
    save_path = './VGG16.pth'
    train_num = len(traindata)
    val_num = len(validatadata)
    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=2, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    epochs = 15
    train_steps = len(train_loader)
    val_steps = len(validata_loader)
    for epoch in range(epochs):
        batch = 1
        train_bar = tqdm(train_loader)
        for step, (b_x, b_y) in enumerate(train_bar, start=0):
            net.train()
            # running_loss = 0.0
            b_x = torch.tensor(b_x, dtype=torch.float32)
            b_y = b_y.to(dtype=torch.float)
            b_y = Variable(b_y.long())
            b_x = torch.unsqueeze(b_x, dim=1)
            optimizer.zero_grad()
            output = net(b_x.to(device))
            loss = loss_function(output, b_y.to(device))
            loss.backward()
            optimizer.step()

            net.eval()
            acc = 0.0
            train_acc = 0.0
            valing_loss = 0.0
            with torch.no_grad():
                val_bar = tqdm(validata_loader)
                for step, (b_x, b_y) in enumerate(train_bar, start=0):
                    b_x = torch.tensor(b_x, dtype=torch.float32)
                    b_y = b_y.to(dtype=torch.float)
                    b_y = Variable(b_y.long())
                    b_x = torch.unsqueeze(b_x, dim=1)
                    optimizer.zero_grad()
                    output = net(b_x.to(device))
                    output_y = torch.max(output, dim=1)[1]
                    train_acc += torch.eq(output_y, b_y.to(device)).sum().item()

                for val_data in val_bar:
                    val_x, val_y = val_data
                    val_x = torch.tensor(val_x, dtype=torch.float32)
                    val_y = val_y.to(dtype=torch.float)
                    val_y = Variable(val_y.long())
                    val_x = torch.unsqueeze(val_x, dim=1)
                    outputs = net(val_x.to(device))
                    val_loss = loss_function(outputs, val_y.to(device))
                    valing_loss += val_loss.item()
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_y.to(device)).sum().item()
            train_loss_list.append(loss)
            train_acc_list.append((train_acc / train_num))
            val_accurate = acc / val_num
            val_loss_list.append(valing_loss / val_steps)
            val_acc_list.append(val_accurate)
            print('[epoch %d] batch: %.1f train_loss: %.3f  val_loss: %.3f train_accuracy: %.3f val_accuracy: %.3f' %
                  (epoch + 1, batch, loss, valing_loss / val_steps, train_acc / train_num, val_accurate))
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            batch += 1

    x1 = range(0, train_steps * epochs)
    x2 = range(0, train_steps * epochs)
    y1 = train_loss_list
    y2 = val_loss_list
    y3 = train_acc_list
    y4 = val_acc_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'b-o')
    plt.plot(x1, y2, 'g-*')
    plt.title(' loss vs. iteration')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y3, 'b-.')
    plt.plot(x2, y4, 'g-.')
    plt.xlabel('acc vs. iteration')
    plt.ylabel('acc')
    plt.show()
    plt.savefig("accuracy_loss.jpg")


if __name__ == '__main__':
    main()
