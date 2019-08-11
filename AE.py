#  라이브러리
import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image


# 미리 만들어둔 모델 불러오기
from utils.AE_model import encoder, decoder


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
num_epochs = 30
batch_size = 128
learning_rate = 1e-3


#  맨 처음 한번만 다운로드 하기
# dataset = MNIST('./data', transform=img_transform, download=True)

#  데이터 불러오기
dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True)


#  모델 설정
encoder = encoder().cuda().train()
decoder = decoder().cuda().train()


#  모델 Optimizer 설정
criterion = nn.MSELoss()
encoder_optimizer = torch.optim.Adam( encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
decoder_optimizer = torch.optim.Adam( decoder.parameters(), lr=learning_rate, weight_decay=1e-5)


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

    if epoch % 10 == 0:
        # pic = to_img(output.cpu().data)
        pic = output.cpu().data
        pic = pic.view(pic.size(0), 1, 28, 28)

        save_image(pic, './AE_img/output_image_{}.png'.format(epoch))

#  모델 저장
torch.save(encoder.state_dict(), './weights/encoder.pth')
torch.save(decoder.state_dict(), './weights/decoder.pth')