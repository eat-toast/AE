import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# 미리 만들어둔 모델 불러오기
from AE_model import encoder, decoder

encoder = encoder()
decoder = decoder()

encoder.load_state_dict(torch.load('./encoder.pth'))
decoder.load_state_dict(torch.load('./decoder.pth'))

#  Hyper Parameter 설정
batch_size = 128


#  데이터 불러오기
dataset = MNIST('./data')
dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True)

#  라벨정보
label = dataset.targets
label = label.data.numpy()

###  만들어둔 모델을 이용해 MSE 계산하기  ###
input = dataset.data.view(dataset.data.size()[0], 784 )
input = input.type(torch.FloatTensor)
z = encoder(input)
output = decoder(z)

input = input.data.numpy()
output = output.data.numpy()

MSE = np.sum( (input[0] - output[0])**2, axis = 0)
RMSE = MSE**(0.5)


sns.distplot(RMSE)


###  Fake_image 넣고 계산  ###
def make_fake_img():
    img_size = 28
    n_fake_img = 60
    fake_img  = []
    for i in range(n_fake_img):
        fake_img.append( np.random.randn(img_size * img_size).reshape(1, img_size, img_size) )

    fake_img = torch.FloatTensor(fake_img)
    fake_img = fake_img.view(60, 784)
    fake_img = Variable(fake_img)

    fake_label = np.array([10] * n_fake_img)
    return fake_img, fake_label


fake_img , fake_label = make_fake_img()

fake_z = encoder(fake_img)
fake_output = decoder(fake_z)

fake_MSE = torch.sum( (fake_img - fake_output)**2, dim = 1).data
fake_RMSE = fake_MSE**(0.5)

sns.distplot(fake_RMSE)


total_RMSE = np.concatenate((RMSE, fake_RMSE))
sns.distplot(total_RMSE )
