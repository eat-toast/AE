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
# python - m visdom.server
import visdom

lcmap = colors.ListedColormap(['silver', '#FF99FF', '#8000FF', '#0000FF', '#0080FF', '#58FAF4',
                               '#00FF00', '#FFFF00', '#FF8000', '#FF0000', 'darkgreen'])


# 미리 만들어둔 모델 불러오기
from utils.AE_model import encoder, decoder
from utils.make_fake_img import make_fake_img

# 시각화
from sklearn.manifold import TSNE


if not os.path.exists('./imgs/AE+TSNE_img'):
    os.mkdir('./imgs/AE+TSNE_img')


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


#  모델 설정
encoder = encoder().cuda()
decoder = decoder().cuda()


#  모델 Optimizer 설정
criterion = nn.MSELoss()
encoder_optimizer = torch.optim.Adam( encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
decoder_optimizer = torch.optim.Adam( decoder.parameters(), lr=learning_rate, weight_decay=1e-5)



#  fake_image 추가
#  Mnist train 은 60,000개.  --> 이 중, 60개를 이상한 데이터로 추가하여 AE가 어떤 오류를 내 뱉는지 보자.

fake_img , fake_label = make_fake_img()

#  loss 시각화
vis = visdom.Visdom()
layout_1 = dict(title="Normal_data")
layout_2 = dict(title="anomaly_data")

normal = vis.line(Y = [0], X = [0], opts =layout_1 )
anomaly = vis.line(Y = [0], X = [0], opts =layout_2 )


for epoch in range(num_epochs):
    for data in dataloader:
        # <<<<<<<<<<<<<<<<정상 데이터>>>>>>>>>>>>>>>>>>
        img, label = data  #  label 은 가져오지 않는다.
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
    print('epoch [{}/{}], loss:{:.4f}' .format(epoch + 1, num_epochs, float(loss.data) ))
    vis.line(Y=[loss.data.cpu().numpy()], X=np.array([epoch]), win=normal, update='append')

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
    print('##### fake epoch [{}/{}], loss:{:.4f} #### '.format(epoch + 1, num_epochs, float(loss.data) ))
    vis.line(Y=[loss.data.cpu().numpy()], X=np.array([epoch]), win=anomaly, update='append')


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
    fig.savefig('./imgs/AE+TSNE_img/' +  str(epoch) + '.jpg')
    plt.close('all')

torch.save(encoder.state_dict(), './weights/encoder_with_anomaly.pth')
torch.save(decoder.state_dict(), './weights/decoder_with_anomaly.pth')