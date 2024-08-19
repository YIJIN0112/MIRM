import torch.nn.functional as F
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from recbole.utils import init_logger, init_seed, set_color
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from logging import getLogger
# from recbole.quick_start import load_data_and_model

from ncl import NCL

# Train on dim=2 and get the plot.
data_list = ["ml-1m"]


filepath = 'saved/NCL-{0}.pth'.format(data_list[0])  
plotpath = 'plot/NCL-{0}.pdf'.format(data_list[0]) 

def load_data_and_model(model_file):
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = NCL(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data

config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file=filepath,
)

item_emb = model.item_embedding.weight.cpu().detach()
item_emb = F.normalize(item_emb, dim=1).numpy()
print(item_emb.shape)

plt.figure(figsize=(3, 3))

df = pd.DataFrame({
    'x': item_emb.T[0],
    'y': item_emb.T[1]
})

ax = sns.kdeplot(
    data=df, x='x', y='y',
    thresh=0, levels=300, cmap=sns.color_palette('light:b', as_cmap=True)
)

plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.savefig(plotpath, format='pdf', dpi=300)  
plt.show()