import argparse
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import itertools


def calculate_ranks_from_similarities(all_similarities, positive_relations):
    positive_relation_similarities = all_similarities[positive_relations]
    negative_relation_similarities = np.ma.array(all_similarities, mask=False)
    negative_relation_similarities.mask[positive_relations] = True
    ranks = list((negative_relation_similarities >= positive_relation_similarities[:, np.newaxis]).sum(axis=1) + 1)
    return ranks

def precision_at_k(preds, gts, k=1):
    p_k = 0.0
    for pred, gt in zip(preds, gts):
        p_k += (len(set(pred[:k]) & set(gt)) / k)
    p_k /= len(preds)
    return p_k

def mrr(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    scaled_rank_positions = np.ceil(rank_positions)
    return (1.0 / scaled_rank_positions).mean()

def example_f1(trues, preds):
    f1_list = []
    for t, p in zip(trues, preds):
        if len(t) + len(p) == 0:
            continue
        f1 = 2 * len(set(t) & set(p)) / (len(t) + len(p))
        f1_list.append(f1)
    return np.array(f1_list).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test BERT multi-label classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--model_pth', type=str, help='model checkpoint path')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load model
    labels_path = os.path.join(args.data_dir, args.dataset, 'train/training_data.pt')
    dummy_data = torch.load(labels_path)
    num_labels = dummy_data["labels"].shape[1]

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        problem_type="multi_label_classification"
    ).to(device)

    model.load_state_dict(torch.load(args.model_pth, map_location=device))
    model.eval()

    # Load test corpus
    corpus = {}
    with open(os.path.join(args.data_dir, args.dataset, 'test/corpus.txt')) as f:
        for line in f:
            i, t = line.strip().split('\t')
            corpus[i] = t

    # Tokenize test data
    texts = list(corpus.values())
    id_list = list(corpus.keys())

    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Run inference
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)

    predictions = np.concatenate(all_preds, axis=0)

    # Load ground truth
    gt_labels = {}
    with open(os.path.join(args.data_dir, args.dataset, 'test/doc2labels.txt')) as f:
        for line in f:
            i, t = line.strip().split('\t')
            gt_labels[i] = t.split(',')
    gt_labels = [list(map(int, gt_labels[i])) for i in id_list]

    all_ranks = []
    top_classes = []
    for pred, gt in zip(predictions, gt_labels):
        all_ranks.append(calculate_ranks_from_similarities(pred, gt))
        top_classes.append(np.argsort(-pred)[:3])  # top 3 predicted

    for k in [1, 2, 3]:
        print(f"Precision@{k}: {precision_at_k(top_classes, gt_labels, k):.4f}")
    print(f"MRR: {mrr(all_ranks):.4f}")
    print(f"Example F1: {example_f1(gt_labels, top_classes):.4f}")
