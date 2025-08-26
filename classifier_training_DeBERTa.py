from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeBERTa-v3 multi-label classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()

    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    # Load preprocessed data
    data_path = os.path.join(args.data_dir, args.dataset, 'train/training_data.pt')
    data_dict = torch.load(data_path)

    dataset = TensorDataset(
        data_dict["input_ids"],
        data_dict["attention_masks"],
        data_dict["labels"],
        data_dict["sample_mask"]
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Define model using AutoModel (important for DeBERTa-v3)
    num_labels = data_dict["labels"].shape[1]
    model = AutoModelForSequenceClassification.from_pretrained(
        'microsoft/deberta-v3-base',
        num_labels=num_labels,
        problem_type="multi_label_classification"
    ).to(device)

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def masked_loss(output, labels, mask):
        loss = loss_fn(output, labels)
        loss = loss * mask
        return loss.mean()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    # Training loop
    model.zero_grad()
    for e in range(args.epoch):
        print(f'\n🔁 Training epoch: {e + 1}/{args.epoch}')
        total_train_loss = 0
        model.train()

        for j, batch in enumerate(tqdm(data_loader)):
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device).float()
            sample_mask = batch[3].to(device).float()

            outputs = model(input_ids=input_ids, attention_mask=input_mask)
            logits = outputs.logits

            loss = masked_loss(logits, labels, sample_mask)
            total_train_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()

        avg_loss = total_train_loss / (j + 1)
        print(f'Epoch {e + 1} average loss: {avg_loss:.4f}')

    # Save model
    save_path = os.path.join(args.data_dir, args.dataset, 'train/model_DeBERTa.pt')
    torch.save(model.state_dict(), save_path)
    print(f'\n✅ Model saved to: {save_path}')
