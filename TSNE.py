import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from sklearn.manifold import TSNE
from torchvision.datasets import MNIST

#  미리 만들어둔 모델 불러오기
from AE_model import encoder

#  모델 불러오기
encoder = encoder()
encoder.load_state_dict(torch.load('./encoder.pth'))
encoder.eval()


#  데이터 불러오기
img_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = MNIST('./data')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

batch = 1000
x = dataset.data[:batch ].view(batch, 784 )
x  = x.type(torch.FloatTensor)
label = dataset.targets[:batch]
label = label.data.numpy()
z = encoder(x)

img_size = 28
fake_img = np.random.randn(batch * img_size * img_size).reshape(batch, img_size, img_size)
fake_img = torch.FloatTensor(fake_img)
fake_img = fake_img.view(batch, 784)
fake_img = Variable(fake_img)

fake_z = encoder(fake_img)
fake_label = np.array( [10] * batch )

z = z.data.numpy()
fake_z = fake_z.data.numpy()

latent = np.concatenate((z, fake_z))
latent_label = np.concatenate( (label, fake_label ))

X_embedded = TSNE(n_components=2).fit_transform(latent)
xs = X_embedded[:,0]
ys = X_embedded[:,1]


lcmap = colors.ListedColormap(['#FFFFFF', '#FF99FF', '#8000FF', '#0000FF', '#0080FF', '#58FAF4',
                               '#00FF00', '#FFFF00', '#FF8000', '#FF0000', 'purple'])

fig, ax = plt.subplots()
im = ax.scatter(xs, ys, cmap=lcmap, c = latent_label )
fig.colorbar(im, ax=ax, ticks = range(11))

#  ax.set_title('epoch 100')
#  fig.savefig('churn_predict.jpg')

plt.show()