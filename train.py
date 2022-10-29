import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from Utils import sandwich_stack, apply_mask, attention_maps
from Refinement.UNet.unet_model import UNet

VOC_transforms_input = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 320)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])
VOC_transforms_target = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((220, 320))
                                    ])

VOC_seg_dataset = torchvision.datasets.VOCSegmentation(root= "./VOC", year = '2012', image_set = 'train', download = False, transform = VOC_transforms_input, target_transform = VOC_transforms_target)
VOC_seg_dataloader = DataLoader(VOC_seg_dataset, batch_size=8, shuffle=True)

attention_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').cuda()
patch_size = 8
num_heads=6

refiner = UNet(4, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
refiner = torch.nn.DataParallel(refiner)
refiner.to(device)

for x, y in VOC_seg_dataloader:
    x = x.cuda()
    attentions = attention_maps(x, attention_model, patch_size, num_heads)
    refiner_input = sandwich_stack(x, attentions)
    refined_attention = refiner(refiner_input)
    masked_image = apply_mask(x, refined_attention)

