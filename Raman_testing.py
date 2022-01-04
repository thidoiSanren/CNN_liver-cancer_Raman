import torch
import torch.utils.data as Data
from Mydatasets import TestdataSets
from model_vgg import vgg
from tqdm import tqdm
from torch.autograd import Variable


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    testdata = TestdataSets()
    test_loader = Data.DataLoader(dataset=testdata, num_workers=4)

    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=2, init_weights=True)
    net.to(device)
    weights_path = "VGG16.pth"
    net.load_state_dict(torch.load(weights_path, map_location=device))

    net.eval()
    test_num = len(testdata)
    acc = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_x, test_y = test_data
            test_x = torch.tensor(test_x, dtype=torch.float32)
            test_y = test_y.to(dtype=torch.float)
            test_y = Variable(test_y.long())
            test_x = torch.unsqueeze(test_x, dim=1)
            outputs = net(test_x.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_y.to(device)).sum().item()

    test_accurate = acc / test_num
    print('test_accuracy: %.3f' % test_accurate)


if __name__ == '__main__':
    main()
