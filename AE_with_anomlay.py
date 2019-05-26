import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import numpy as np

# python - m visdom.server
import visdom

#  미리 만들어둔 모델 불러오기
from AE_model import encoder, decoder

#  이미지를 저장할 폴더 생성
if not os.path.exists('./AE_img'):
    os.mkdir('./AE_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


img_transform = transforms.Compose([
    transforms.ToTensor()
])

#  Hyper Parameter 설정
num_epochs = 100
batch_size = 128
learning_rate = 1e-3


#  맨 처음 한번만 다운로드 하기
# dataset = MNIST('./data', transform=img_transform, download=True)

#  데이터 불러오기
dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True)


#  모델 설정
encoder = encoder().cuda()
decoder = decoder().cuda()


#  모델 Optimizer 설정
criterion = nn.MSELoss()
encoder_optimizer = torch.optim.Adam( encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
decoder_optimizer = torch.optim.Adam( decoder.parameters(), lr=learning_rate, weight_decay=1e-5)


# fake_image 추가
# Mnist train 은 60,000개.  --> 이 중, 60개를 이상한 데이터로 추가하여 AE가 어떤 오류를 내 뱉는지 보자.

def make_fake_img():
    img_size = 28
    n_fake_img = 60
    fake_img  = []
    for i in range(n_fake_img):
        fake_img.append( np.random.randn(img_size * img_size).reshape(1, img_size, img_size) )

    fake_img = torch.FloatTensor(fake_img)
    fake_img = fake_img.view(n_fake_img, img_size * img_size)
    fake_img = Variable(fake_img).cuda()

    return fake_img


fake_img = make_fake_img()

vis = visdom.Visdom()
layout_1 = dict(title="Normal_data")
layout_2 = dict(title="anomaly_data")

normal = vis.line(Y = [0], X = [0], opts =layout_1 )
anomaly = vis.line(Y = [0], X = [0], opts =layout_2 )


for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data  # label 은 가져오지 않는다.
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        latent_z = encoder(img)
        output = decoder(latent_z )
        # ===================backward====================
        loss = criterion(output, img)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}' .format(epoch + 1, num_epochs, float(loss.data) ))
    vis.line(Y=[loss.data.cpu().numpy()], X=np.array([epoch]), win=normal, update='append')

    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './AE_img/output_image_{}.png'.format(epoch))

    # ===================forward=====================
    latent_z = encoder(fake_img)
    output = decoder(latent_z )

    # ===================backward====================
    loss = criterion(output, fake_img)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    # ===================log========================
    print('##### fake epoch [{}/{}], loss:{:.4f} #### '.format(epoch + 1, num_epochs, float(loss.data) ))
    vis.line(Y=[loss.data.cpu().numpy()], X=np.array([epoch]), win=anomaly, update='append')
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './AE_img/fake_image_{}.png'.format(epoch))

#  모델 저장
torch.save(encoder.state_dict(), './encoder.pth')
torch.save(decoder.state_dict(), './decoder.pth')

##################################################  시각화  ##################################################
#  비정상 데이터
latent_z = encoder(fake_img)
output = decoder(latent_z)

fake = (fake_img - output).data.cpu().numpy()
fake = np.sum( fake**2, axis = 1)

#  정상 데이터
img = dataloader.dataset.data
img = img.view(img.size(0), -1)
img= img.type('torch.cuda.FloatTensor')
img = img / 255

latent_z = encoder(img)
output = decoder(latent_z)

origin = (img - output).data.cpu().numpy()
origin = np.sum( origin **2, axis = 1)


import seaborn as sns
import matplotlib.pyplot as plt

# Method 1: on the same Axis
sns.distplot(origin, color="skyblue", label="origin")
sns.distplot(fake, color="red", label="fake")
plt.legend()