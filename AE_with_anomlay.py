'''
기본 골격은 AE + TSNE 파일과 같지만, visdom 을 이용해 loss 시각화 하는 부분을 추가 함.

Dist plot으로 anomaly 데이터와 정상데이터가 구분이 되는지 확인
'''


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import numpy as np

#  미리 만들어둔 모델 불러오기
from utils.AE_model import encoder, decoder
from utils.make_fake_img import make_fake_img


img_transform = transforms.Compose([
    transforms.ToTensor()
])

#  Hyper Parameter 설정
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

#  데이터 불러오기
dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True)


# fake_image 추가
# Mnist train 은 60,000개.  --> 이 중, 60개를 이상한 데이터로 추가하여 AE가 어떤 오류를 내 뱉는지 보자.
fake_img, _ = make_fake_img()

##################################################  시각화  ##################################################

#  모델 불러오기 - eval
#  사용하는 모델은 AE + TNSE 에서 학습한 weight를 이용
encoder = encoder().cuda().eval()
decoder = decoder().cuda().eval()
encoder.load_state_dict(torch.load('./weights/encoder_with_anomaly.pth'))
decoder.load_state_dict(torch.load('./weights/decoder_with_anomaly.pth'))

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


#  궁금한 사항들
#  1. anomaly 데이터가 많으면 성능문제는 없을까?
#  2. AE 모델은 원본 + alpha * 노이즈(noise)를 어디까지 구분할 수 있을까? (alpha 변형해보면서 실험)
#  3. 모델별 성능의 차이가 있을까?
