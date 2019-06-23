import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

lcmap = colors.ListedColormap(['silver', '#FF99FF', '#8000FF', '#0000FF', '#0080FF', '#58FAF4',
                               '#00FF00', '#FFFF00', '#FF8000', '#FF0000', 'darkgreen'])

# 미리 만들어둔 모델 불러오기
from AE_model import encoder, decoder

# 시각화
from sklearn.manifold import TSNE

if not os.path.exists('./AE+TSNE_img_LeakyReLU'):
    os.mkdir('./AE+TSNE_img_LeakyReLU')

img_transform = transforms.Compose([
    transforms.ToTensor()
])

#  Hyper Parameter 설정
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

#  데이터 불러오기
dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#  모델 설정
encoder = encoder().cuda()
decoder = decoder().cuda()

#  모델 Optimizer 설정
criterion = nn.MSELoss()
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=1e-5)


#  fake_image 추가
#  Mnist train 은 60,000개.  --> 이 중, 60개를 이상한 데이터로 추가하여 AE가 어떤 오류를 내 뱉는지 보자.


def make_fake_img():
    img_size = 28
    n_fake_img = 60
    fake_img = []
    for i in range(n_fake_img):
        fake_img.append(np.random.randn(img_size * img_size).reshape(1, img_size, img_size))

    fake_img = torch.FloatTensor(fake_img)
    fake_img = fake_img.view(60, 784)
    fake_img = Variable(fake_img).cuda()

    fake_label = np.array([10] * n_fake_img)
    return fake_img, fake_label


fake_img, fake_label = make_fake_img()

for epoch in range(num_epochs):
    for data in dataloader:
        # <<<<<<<<<<<<<<<<정상 데이터>>>>>>>>>>>>>>>>>>
        img, label = data  # label 은 가져오지 않는다.
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        latent_z = encoder(img)
        output = decoder(latent_z)
        # ===================backward====================
        loss = criterion(output, img)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, float(loss.data)))

    # <<<<<<<<<<<<<<<<비 정상 데이터>>>>>>>>>>>>>>>>>>
    # ===================forward=====================
    fake_latent_z = encoder(fake_img)
    output = decoder(fake_latent_z)

    # ===================backward====================
    loss = criterion(output, fake_img)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    # ===================log========================
    print('##### fake epoch [{}/{}], loss:{:.4f} #### '.format(epoch + 1, num_epochs, float(loss.data)))

    #  ===================Latent Space 확인========================
    #  6만개의 데이터가 너무 많아서, 1,000개의 sample 만 확인.
    sample_size = 1000
    x = dataset.data[:sample_size].view(sample_size, 784)
    x = x.type(torch.FloatTensor).cuda()
    label = dataset.targets[:sample_size]
    label = label.data.numpy()

    #  Latent Space
    z = encoder(x)
    fake_z = encoder(fake_img)

    #  change to numpy
    z = z.data.cpu().numpy()
    fake_z = fake_z.data.cpu().numpy()

    #  concat
    latent = np.concatenate((z, fake_z))
    latent_label = np.concatenate((label, fake_label))

    #  TSNE
    X_embedded = TSNE(n_components=2).fit_transform(latent)
    xs = X_embedded[:, 0]
    ys = X_embedded[:, 1]

    #  make figure
    fig, ax = plt.subplots()
    im = ax.scatter(xs, ys, cmap=lcmap, c=latent_label)
    fig.colorbar(im, ax=ax, ticks=range(11))
    ax.set_title('epoch:{}'.format(epoch))
    fig.savefig('./AE+TSNE_img_LeakyReLU/' + str(epoch) + '.jpg')
    plt.close('all')

torch.save(encoder.state_dict(), './encoder2.pth')
torch.save(decoder.state_dict(), './decoder2.pth')