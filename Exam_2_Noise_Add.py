'''
#  2. AE 모델은 원본 + alpha * 노이즈(noise)를 어디까지 구분할 수 있을까? (alpha 변형해보면서 실험)
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
if not os.path.exists('./imgs/AE_exam_2_Noise_Add'):
    os.mkdir('./imgs/AE_exam_2_Noise_Add')


#  미리 만들어둔 모델 불러오기
from utils.AE_model import encoder, decoder

img_transform = transforms.Compose([
    transforms.ToTensor()
])

#  Hyper Parameter 설정
num_epochs = 10
batch_size = 2048
learning_rate = 1e-3


#  데이터 불러오기
dataset = MNIST('data', transform=img_transform)


fake_img = dataset.data[:600]
fake_img = fake_img.type('torch.FloatTensor')  / 255


norm_img = dataset.data[600:]
norm_img = norm_img.type('torch.FloatTensor') / 255
dataloader = DataLoader(norm_img, batch_size=batch_size , shuffle=True)


#  모델 Optimizer 설정
criterion = nn.MSELoss()

#  Noise 추가
def make_fake_img(fake_img, noise_ratio):
    img_size = 28
    n_fake_img = 600

    temp_img = fake_img + torch.FloatTensor( ( np.random.randn(n_fake_img, img_size , img_size) * noise_ratio ) )
    temp_img = temp_img.view(n_fake_img, img_size * img_size)
    temp_img = Variable(temp_img)

    return temp_img



Ratio = [0.0001, 0.01, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.99]

for ratio in Ratio:
    perturbation_img = make_fake_img(fake_img = fake_img, noise_ratio = ratio)

    #  모델 설정
    encoder_model = encoder()
    decoder_model = decoder()

    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    decoder_optimizer = torch.optim.Adam(decoder_model.parameters(), lr=learning_rate, weight_decay=1e-5)


    for epoch in range(num_epochs):
        for data in dataloader:
            img = data  # data loader이 바꼇다
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
        latent_z = encoder_model(perturbation_img)
        output = decoder_model(latent_z )

        # ===================backward====================
        loss = criterion(output, perturbation_img)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        # ===================log========================
        print('##### fake epoch [{}/{}], loss:{:.4f} #### '.format(epoch + 1, num_epochs, float(loss.data) ))


    #  모델 저장
    torch.save(encoder_model.state_dict(), './weights/encoder_Noise_' + str(ratio)+ '.pth')
    torch.save(decoder_model.state_dict(), './weights/decoder_Noise_' + str(ratio)+ '.pth')

    ##################################################  시각화  ##################################################
    #  비정상 데이터
    latent_z = encoder_model(perturbation_img)
    output = decoder_model(latent_z)

    fake = (perturbation_img - output).data.cpu().numpy()
    fake = np.sum( fake**2, axis = 1)
    del latent_z
    del output

    #  정상 데이터
    norm_img = norm_img.view(norm_img.size(0), -1)
    latent_z = encoder_model(norm_img)
    output = decoder_model(latent_z)

    origin = (norm_img - output).data.cpu().numpy()
    origin = np.sum( origin **2, axis = 1)

    # Method 1: on the same Axis
    sns.distplot(origin, color="skyblue", label="origin").set_title('noise ratio = '+str(ratio))
    sns.distplot(fake, color="red", label="fake")
    plt.legend()
    plt.savefig('./imgs/AE_exam_2_Noise_Add/'+str(ratio) + '_.png')
    plt.close()

    del latent_z
    del output
    del encoder_model
    del decoder_model
    del encoder_optimizer
    del decoder_optimizer
    del perturbation_img
    del img
    del origin


from torchvision.utils import save_image

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


for ratio in Ratio:
    encoder_model = encoder().eval()
    decoder_model = decoder().eval()

    encoder_model.load_state_dict(torch.load('./weights/encoder_Noise_' + str(ratio)+ '.pth'))
    decoder_model.load_state_dict(torch.load('./weights/decoder_Noise_' + str(ratio)+ '.pth'))

    #  비정상 데이터
    perturbation_img = make_fake_img(fake_img=fake_img, noise_ratio=ratio)
    perturbation_img = perturbation_img.cpu().data[:32]

    pic = perturbation_img.cpu().data
    pic = pic.view(pic.size(0), 1, 28, 28)

    save_image(pic, './imgs/AE_exam_2_Noise_Add/origin_image_{}.png'.format(ratio))


    del encoder_model
    del decoder_model