import torch

def sandwich_stack(image, attention):
    batch_size, _, height, width = image.shape
    batch_size, num_heads, height, width = attention.shape
    placeholder = torch.zeros((batch_size, num_heads * 4, height, width)).cuda()
    for i in range(num_heads):
        placeholder[:, 4*i + 0, :, :] = image[:, 0, :, :]
        placeholder[:, 4*i + 1, :, :] = image[:, 1, :, :]
        placeholder[:, 4*i + 2, :, :] = image[:, 2, :, :]
        placeholder[:, 4*i + 3, :, :] = attention[:, i, :, :]
    placeholder = placeholder.reshape(batch_size*num_heads, 4, height, width)
    return placeholder

def apply_mask(image, refined_attention):
    batch_size, _, height, width = image.shape
    refined_attention = refined_attention.repeat_interleave(3, dim=1)
    image = image.repeat(1,6,1,1)
    return (image * refined_attention).reshape(-1, 3, height, width)


def attention_maps(x, model, patch_size, num_heads):
    batch_size, _, h, w = x.shape
    with torch.no_grad():
        attentions = model.get_last_selfattention(x)
    w_featmap = w // patch_size
    h_featmap = h // patch_size
    attentions = attentions[:, :, 0, 1:].reshape(batch_size, num_heads, h_featmap, w_featmap)
    attentions = torch.nn.functional.interpolate(attentions, scale_factor=patch_size, mode='bicubic')
    return attentions