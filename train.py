import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BILangDataset
from torch.utils.data import DataLoader, random_split
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
import time
import os
from dataset import casual_mask


# Config
MODELS_PATH = 'models'
name = 'english_2_telugu'
SEQ_LENGTH = 256
BATCH_SIZE = 8
lr = 0.00005
EPOCH = 20
NO_LAYERS = 6
D_MODEL = 512
model_filename = os.path.join(MODELS_PATH, f"{name}.pt")




def get_sentences(dataset_split, lang):
    for item in dataset_split:
        yield item[lang]




def tokenize(tokenizer_path, dataset, lang):
    tokenizer_path = Path(tokenizer_path)

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_sentences(dataset, lang), trainer=trainer)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds():
    dataset = load_dataset("HackHedron/English_Telugu_Parallel_Corpus")['train']
    dataset = dataset.select(range(2000))

    en_tokenizer = tokenize('tokenizer/shared_en_tokenizer.json', dataset, lang='english')
    te_tokenizer = tokenize('tokenizer/shared_te_tokenizer.json', dataset, lang='telugu')

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataset = BILangDataset(train_dataset, en_tokenizer, te_tokenizer, 'english', 'telugu', SEQ_LENGTH)
    test_dataset = BILangDataset(test_dataset, en_tokenizer, te_tokenizer, 'english', 'telugu', SEQ_LENGTH)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader, en_tokenizer, te_tokenizer

def build_model(src_vocab_size, tgt_vocab_size):
    model = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=SEQ_LENGTH,
        tgt_seq_len=SEQ_LENGTH,
        N=NO_LAYERS,
        d_model=D_MODEL
    )
    return model


def evaluate(model, dataloader, loss_fn, device, tokenizer_src, tokenizer_tgt):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            encoder_input = batch['encoder_input'].to(torch.long).to(device)
            decoder_input = batch['decoder_input'].to(torch.long).to(device)
            encoder_mask = batch['encoder_mask'].to(torch.long).to(device)
            decoder_mask = batch['decoder_mask'].to(torch.long).to(device)
            label = batch['label'].to(torch.long).to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            proj_output = model.projection(decoder_output)

            # Loss calculation
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_loss += loss.item()

            # Decode predictions (optional)
            pred_ids = torch.argmax(proj_output, dim=-1)  # [B, T]
            pred_text = tokenizer_tgt.decode(pred_ids[0].cpu().numpy())
            target_text = batch['tgt_text'][0]
            source_text = batch['src_text'][0]

            # Print results
            print("SOURCE     :", source_text)
            print("TARGET     :", target_text)
            print("PREDICTED  :", pred_text)
            print("-" * 60)

    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")
    return avg_loss



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader, en_tokenizer, te_tokenizer = get_ds()
    model = build_model(en_tokenizer.get_vocab_size(), te_tokenizer.get_vocab_size()).to(device)

    Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("runs", name))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=te_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1)

    global_step = 0

    for epoch in range(EPOCH):
        torch.cuda.empty_cache()
        model.train()
        start_time = time.time()
        epoch_loss = 0
        batch_count = 0

        for batch in train_loader:
            batch_count += 1

            encoder_input = batch['encoder_input'].to(torch.long).to(device)
            decoder_input = batch['decoder_input'].to(torch.long).to(device)
            encoder_mask = batch['encoder_mask'].to(torch.long).to(device)
            decoder_mask = batch['decoder_mask'].to(torch.long).to(device)
            label = batch['label'].to(torch.long).to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            proj_output = model.projection(decoder_output)

            loss = loss_fn(proj_output.view(-1, te_tokenizer.get_vocab_size()), label.view(-1))
            loss.backward()

            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if batch_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Batch: {batch_count}, Global Step: {global_step}, Loss: {loss.item():.4f}, Time: {elapsed:.2f}s")

        avg_train_loss = epoch_loss / batch_count
        val_loss = evaluate(model, test_loader, loss_fn, device, en_tokenizer, te_tokenizer)


        print(f"Epoch {epoch + 1}/{EPOCH} | Train Loss: {avg_train_loss:.4f}  | Val Loss: {val_loss:.4f}")
        # writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        # writer.add_scalar("Loss/Val", val_loss, epoch)

        # Save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    train()