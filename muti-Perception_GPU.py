import torch as t
from torch import nn
from torch.autograd import Variable as V
from torch import optim
import torchvision as tv
import numpy as np
import visdom
import os


class Muti_Perceptron(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = t.nn.Linear(784,512)  # 第一层仿射
        self.fc2 = t.nn.Linear(512,128)  # 第二层仿射
        self.fc3 = t.nn.Linear(128,10)   # 第三层仿射

    def init_params(self):
        # 可以自己初始化,应用条件
        # 1.在nn的模型中不提供初始化,自己初始化
        # 2.得到足够先验知识,可以自己初始化,这样可以加快训练速度
        # 不推荐自己初始化
        self.fc1.weight.data.normal_(0, 5)
        self.fc1.bias.data.normal_(0, 1)
        self.fc2.weight.data.normal_(0, 5)
        self.fc2.bias.data.normal_(0, 1)
        self.fc3.weight.data.normal_(0, 5)
        self.fc3.bias.data.normal_(0, 1)

    def forward(self,x):
        x = x.view(-1, 28 * 28)  # 输入为28*28的图像
        x = t.nn.functional.relu(self.fc1(x))  # 第一层仿射 + 第一层线性变换
        x = t.nn.functional.relu(self.fc2(x))  # 第二层仿射 + 第二层线性变换
        x = self.fc3(x) # 第三层仿射
        x = t.nn.functional.softmax(x, dim=1)  # softmax函数输出
        return x


class perception_model():

    def __init__(self):
        self.mp = Muti_Perceptron().cuda()
        #self.mp.init_params()

    def get_train_data(self, batch_size=100):
        # 构造数据
        train_set = tv.datasets.MNIST("mnist/train", train=True,
                                       transform=tv.transforms.ToTensor(), download=True)
        # 一定要的打乱数据
        train_dataset = t.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        return train_dataset

    def get_test_data(self):
        # 构造数据
        test_set = tv.datasets.MNIST("mnist/test", train=False,
                                     transform=tv.transforms.ToTensor(), download=True)
        test_dataset = t.utils.data.DataLoader(test_set, batch_size=100)
        return test_dataset


    def AccuarcyCompute(self, outputs, labels):
        pred = outputs.cpu().data.numpy()  # 训练函数的输出结果,转化为CPU数值, 每一行代表一个样本的输出,是一个概率
        label = labels.cpu().data.numpy()  # 获取样本的label
        test_np = (np.argmax(pred, 1) == label)  # 获取最大的值的标签,进行比对,获取boolean矩阵
        test_np = np.float32(test_np)  # 转化为byte矩阵,1代表预测准确
        return np.mean(test_np)  # 求平均值(即一共都多少预测正确)


    def train(self, epoch=20, noise=False, filter=False, showImage=False):
        # 获取数据集
        train_dataset = self.get_train_data(batch_size=100)

        optimizer = optim.SGD(params=self.mp.parameters(),lr=0.01)
        optimizer.zero_grad()
        # 损失函数
        criterion = nn.CrossEntropyLoss().cuda()

        # 可视化
        vis = visdom.Visdom(env='loss')
        visimage = visdom.Visdom(env='image')
        step = list()
        step_loss = list()

        for x in range(epoch):
            for i, data in enumerate(train_dataset):
                optimizer.zero_grad()  # 梯度清零
                (inputs, labels) = data  # 将数据分为input和label
                beforegaussnoise = (inputs*255)
                if noise == True:
                    inputs = self.gaussNoise(inputs)  # add guass noise
                aftergaussnoise = (inputs*255)

                if filter==True:
                    self.guass_filter(inputs)  # guass filter
                aftergaussfliter = (inputs * 255)
                if showImage==True:
                    images = t.cat([beforegaussnoise, t.cat([aftergaussnoise, aftergaussfliter],dim=0)], dim=0)
                    visimage.images(images, win="image", opts=dict(title="images"))

                inputs = t.autograd.Variable(inputs).cuda()  # 变为variable类型
                labels = t.autograd.Variable(labels).cuda()  # 变为variable类型

                outputs = self.mp(inputs)  # 将数据输入神经网络,获取输出结果
                loss = criterion(outputs, labels)  # 计算损失函数
                loss.backward()  # 逆向传播backward计算梯度

                optimizer.step()  # 更新参数

            l = self.AccuarcyCompute(outputs, labels)
            step.append(len(step)*1)
            step_loss.append(1 - l)
        vis.line(X=step, Y=step_loss, win='loss', opts={'title': 'CrossEntropyLoss'})
        print('train done')

    # 保存模型
    def save_model(self):
        t.save(self.mp.state_dict(), 'mlp_model_GPU.pt')

    # 加载模型
    def load_model(self):
        self.mp.load_state_dict(t.load('mlp_model_GPU.pt'))
        self.mp.cuda()

    def guass_filter(self, inputs):
        for n in range(0, inputs.shape[0]):
            for i in range(0, inputs.shape[2]):
                for j in range(0, inputs.shape[3]):
                    if j == 0 and i == 0:
                        sub_matrix = inputs[n, 0, i:i + 2, j:j + 2]
                        value = sub_matrix.sum() / 4.0
                        inputs[n, 0, i, j] = value
                        continue
                    if j == 0 and i == 27:
                        sub_matrix = inputs[n, 0, i-1:i+1, j:j + 2]
                        value = sub_matrix.sum() / 4.0
                        inputs[n, 0, i, j] = value
                        continue
                    if j == 27 and i == 0:
                        sub_matrix = inputs[n, 0, i:i+2, j-1:j+1]
                        value = sub_matrix.sum() / 4.0
                        inputs[n, 0, i, j] = value
                        continue
                    if j == 27 and i == 27:
                        sub_matrix = inputs[n, 0, i-1:i+1, j - 1:j+1]
                        value = sub_matrix.sum() / 4.0
                        inputs[n, 0, i, j] = value
                        continue
                    if j == 0:
                        sub_matrix = inputs[n, 0, i - 1:i+2, j:j+2]
                        value = sub_matrix.sum() / 6.0
                        inputs[n, 0, i, j] = value
                        continue
                    if j == 27:
                        sub_matrix = inputs[n, 0, i - 1:i + 2, j-1:j+1]
                        value = sub_matrix.sum() / 6.0
                        inputs[n, 0, i, j] = value
                        continue
                    if i == 0:
                        sub_matrix = inputs[n, 0, i:i + 2, j-1:j + 2]
                        value = sub_matrix.sum() / 6.0
                        inputs[n, 0, i, j] = value
                        continue
                    if i == 27:
                        sub_matrix = inputs[n, 0, i - 1:i+1, j-1:j+2]
                        value = sub_matrix.sum() / 6.0
                        inputs[n, 0, i, j] = value
                        continue
                    if j - 1 >= 0 and i-1 >= 0 and j+1<=27 and i+1<=27:
                        sub_matrix = inputs[n, 0, i-1:i+2, j-1:j+2]
                        value = sub_matrix.sum()/9.0
                        inputs[n, 0, i, j] = value
                        continue

    def gaussNoise(self, inputs):
        noise = t.from_numpy(self.gen_noise(batch_size=inputs.shape[0]))  # generate noise
        inputs = t.add(inputs, noise.float())  # origin add noise
        one = t.ones_like(inputs)
        inputs = t.where(inputs > 1.0, one, inputs)
        return inputs

    def gen_noise(self, batch_size, mean=(14, 14), cov=[[14, 0], [0, 14]], size=(10, 10)):
        batch_noise = np.zeros((batch_size, 1, 28, 28))
        for k in range(0, batch_size):
            local = np.random.multivariate_normal(mean, cov, size)
            noise = np.zeros((1, 28, 28))
            for i in range(0, 10):
                for j in range(0, 10):
                    x_dim = max(0, min(int(local[i, j, 0]), 27))
                    y_dim = max(0, min(int(local[i, j, 1]), 27))
                    value = np.random.uniform(0, 1)
                    noise[0, x_dim, y_dim] = value
            batch_noise[k] = noise
        return batch_noise

    # 测试
    def test(self):
        self.load_model()
        self.mp.eval()
        test_dataset = self.get_test_data()
        accuarcy_list = []
        for i, (inputs, labels) in enumerate(test_dataset):
            inputs = t.autograd.Variable(inputs).cuda()
            labels = t.autograd.Variable(labels).cuda()
            outputs = self.mp(inputs)
            accuarcy_list.append(self.AccuarcyCompute(outputs, labels))
        print('测试集上的精确度:',sum(accuarcy_list) / len(accuarcy_list))
        self.mp.train()



if __name__=='__main__':
    pm = perception_model()
    pm.train(noise=False)
    pm.save_model()
    pm.test()

