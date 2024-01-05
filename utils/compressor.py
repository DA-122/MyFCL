import torch
from convs.vqvae import VQVAE

    
def get_compressor_params(dataset_name):
    """_summary_

    Args:
        dataset_name (_type_): _description_

    Returns:
        tuple: _description_
    """
    name = dataset_name.lower()
    if name in ['cifar10', 'cifar100']:
        return (3, 64, 128, 32)
    elif name == "tinyimagenet":
        return (3, 128, 512, 64)
    else:
        assert 0

def get_compressor_model(dataset_name: str, pretrained = False):
    input_channels, embedding_dim, num_embeddings, img_size = get_compressor_params(dataset_name)
    model = VQVAE(input_channels, embedding_dim, num_embeddings, img_size= img_size)
    if pretrained:
        state_dict = torch.load('./checkpoint/vqvae_best_1.pt')
        model.load_state_dict(state_dict)
    return model
