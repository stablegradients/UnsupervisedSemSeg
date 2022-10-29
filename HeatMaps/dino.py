import torch, torchvision

def attention_maps(x, model, patch_size, num_heads):
    batch_size, _, h, w = x.shape
    print(x.shape)
    with torch.no_grad():
        attentions = model.get_last_selfattention(x)

    w_featmap = w // patch_size
    h_featmap = h // patch_size
    attentions = attentions[:, :, 0, 1:].reshape(batch_size, num_heads, h_featmap, w_featmap)
    attentions = torch.nn.functional.interpolate(attentions, scale_factor=patch_size, mode='bicubic')
    return attentions
