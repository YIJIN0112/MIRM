import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color

from model import NCL, SVD_MI
from trainer import NCLTrainer, SVD_MITrainer
from transformers import BertModel, BertTokenizer

import torch
from tqdm import tqdm
import os


def run_single_model(args):
    # configurations initialization
    if args.model=='SVD-MI':
        config = Config(
            model=SVD_MI,
            dataset=args.dataset, 
            config_file_list=args.config_file_list
        )
        config['svd'] = args.svd
        config['mi'] = args.mi
    else:
        config = Config(
            model=NCL,
            dataset=args.dataset, 
            config_file_list=args.config_file_list
        )
    config['gpu_id'] = "6"
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    if config['mi']>0:
        # preprocess BERT embedding
        file_path = args.dataset+'_emb.pt'
        if not os.path.exists(file_path):
            with open("dataset/ml-1m/ml-1m.item", 'r') as f:
                lines = f.readlines()
                lines = lines[1:]
                item_content = ['[PAD]']
                for line in lines:
                    columns = line.split('\t')
                    title = columns[1]  
                    description = columns[-1].strip() 
                    combined_text = title + " " + description  # Combine title and overview
                    item_content.append(combined_text)
                    # item_content.append(columns[1])

            # generate BERT model results
            id2word = dataset.field2id_token['movie_title']
            BERT = BertModel.from_pretrained('bert-base-uncased').cuda()
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            all_emb = []
            for item in tqdm(item_content):
                tokens = tokenizer.tokenize(item)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids_batch = torch.tensor([input_ids]).cuda()
                input_ids_batch = {'input_ids': input_ids_batch}
                with torch.no_grad():
                    outputs = BERT(**input_ids_batch)

                last_hidden_states = outputs.last_hidden_state[:, 0, :].mean(dim=0)
                all_emb.append(last_hidden_states)
            all_emb = torch.stack(all_emb, dim=0)
            torch.save(all_emb, file_path)
        else:
            all_emb = torch.load(file_path)

    # model loading and initialization
    if args.model=='SVD-MI':
        model = SVD_MI(config, train_data.dataset).to(config['device'])
    else:
        model = NCL(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    if args.model=='SVD-MI':
        if config['mi']>0:
            trainer = SVD_MITrainer(config, model, all_emb)
        else:
            trainer = SVD_MITrainer(config, model)
    else:
        trainer = NCLTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.test(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp', help='The datasets is ml-1m')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    parser.add_argument('--model', type=str, default='NCL', help='model name.')
    parser.add_argument('--svd', type=bool, default=False, help='Singular value decomposition (SVD).') 
    parser.add_argument('--mi', type=int, default=0, help='Number of multi-interests.')
    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        'properties/overall.yaml',
        'properties/NCL.yaml'
    ]
    if args.dataset in ['ml-1m']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    if args.config is not '':
        args.config_file_list.append(args.config)

    run_single_model(args)
