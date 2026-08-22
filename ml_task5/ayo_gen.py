import torch
import argparse
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from dcgan import Generator  

parser = argparse.ArgumentParser()
parser.add_argument("--n_images", type=int, default=25)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--img_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=1)
opt = parser.parse_args()


generator = Generator()  


generator.load_state_dict(torch.load("generator_epoch_499.pth", map_location=torch.device('cpu')))


generator.eval()

z = Variable(torch.FloatTensor(np.random.normal(0, 1, (opt.n_images, opt.latent_dim))))

with torch.no_grad():
    gen_imgs = generator(z)

save_image(gen_imgs.data, 'generated.png', nrow=5, normalize=True)
print(f"Сохранено {opt.n_images} изображений в generated.png")