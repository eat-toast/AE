import torch
import numpy as np
from torch.autograd import Variable

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