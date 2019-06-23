'''
#  1. anomaly 데이터가 많으면 성능문제는 없을까?
'''

import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#  이미지를 저장할 폴더 생성
if not os.path.exists('./imgs/AE_exam_1_How_Many'):
    os.mkdir('./imgs/AE_exam_1_How_Many')


#  미리 만들어둔 모델 불러오기
from utils.AE_model import encoder, decoder
from utils.make_fake_img import make_fake_img

img_transform = transforms.Compose([
    transforms.ToTensor()
])

#  Hyper Parameter 설정
num_epochs = 10
batch_size = 2048
learning_rate = 1e-3


#  데이터 불러오기
dataset = MNIST('data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True)


#  모델 Optimizer 설정
criterion = nn.MSELoss()



Alpha = [0.0001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.99]

for alpha in Alpha:
    # fake_image 추가
    fake_img, _ = make_fake_img(alpha, cuda = False)

    #  모델 설정
    encoder_model = encoder()
    decoder_model = decoder()

    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    decoder_optimizer = torch.optim.Adam(decoder_model.parameters(), lr=learning_rate, weight_decay=1e-5)


    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data  # label 은 가져오지 않는다.
            img = img.view(img.size(0), -1)
            img = Variable(img)
            # ===================forward=====================
            latent_z = encoder_model(img)
            output = decoder_model(latent_z )
            # ===================backward====================
            loss = criterion(output, img)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}' .format(epoch + 1, num_epochs, float(loss.data) ))


        # ===================forward=====================
        latent_z = encoder_model(fake_img)
        output = decoder_model(latent_z )

        # ===================backward====================
        loss = criterion(output, fake_img)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        # ===================log========================
        print('##### fake epoch [{}/{}], loss:{:.4f} #### '.format(epoch + 1, num_epochs, float(loss.data) ))


    #  모델 저장
    torch.save(encoder_model.state_dict(), './weights/encoder_Alpha_' + str(alpha)+ '.pth')
    torch.save(decoder_model.state_dict(), './weights/decoder_Alpha_' + str(alpha)+ '.pth')

    ##################################################  시각화  ##################################################
    #  비정상 데이터
    latent_z = encoder_model(fake_img)
    output = decoder_model(latent_z)

    fake = (fake_img - output).data.cpu().numpy()
    fake = np.sum( fake**2, axis = 1)
    del latent_z
    del output

    #  정상 데이터
    img = dataloader.dataset.data
    img = img.view(img.size(0), -1)
    img= img.type('torch.FloatTensor')
    img = img / 255

    latent_z = encoder_model(img)
    output = decoder_model(latent_z)

    origin = (img - output).data.cpu().numpy()
    origin = np.sum( origin **2, axis = 1)

    # Method 1: on the same Axis
    sns.distplot(origin, color="skyblue", label="origin").set_title('alpha = '+str(alpha))
    sns.distplot(fake, color="red", label="fake")
    plt.legend()
    plt.savefig('./imgs/AE_exam_1_How_Many/'+str(alpha) + '_.png')
    plt.close('all')

    del latent_z
    del output
    del encoder_model
    del decoder_model
    del encoder_optimizer
    del decoder_optimizer
    del fake_img
    del img
    del origin