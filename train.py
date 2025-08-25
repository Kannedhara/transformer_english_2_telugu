import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BILangDataset, casual_mask
from torch.utils.data import DataLoader
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

MODELS_PATH = 'models'
name = 'englist_2_telugu'
SEQ_LENGTH = 256
BATCH_SIZE=8
lr=0.0001
EPOCH=20
NO_LAYERS=6
model_filename='english_2_telugu'

def get_sentences(dataset_split, lang):
    for item in dataset_split:
        yield item[lang]

def tokenize(tokenizer_path, dataset, lang):
    tokenizer_path = Path(tokenizer_path)

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_sentences(dataset, lang), trainer=trainer)
        tokenizer_path = Path(tokenizer_path)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds():
    dataset=load_dataset("HackHedron/English_Telugu_Parallel_Corpus")
    dataset = dataset['train']


    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    
    train_en_tokenizer = tokenize('tokenizer/train_en_tokenizer.json', train_dataset, lang='english')
    train_te_tokenizer = tokenize('tokenizer/train_te_tokenizer.json', train_dataset, lang='telugu')
    test_en_tokenizer = tokenize('tokenizer/test_en_tokenizer.json', test_dataset, lang='english')
    test_te_tokenizer = tokenize('tokenizer/test_te_tokenizer.json', test_dataset, lang='telugu')
    

    train_dataset = BILangDataset(train_dataset, train_en_tokenizer, train_te_tokenizer, 'english', 'telugu', SEQ_LENGTH)
    test_dataset = BILangDataset(test_dataset, test_en_tokenizer, test_te_tokenizer, 'english', 'telugu', SEQ_LENGTH)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_data_loader, test_data_loader, train_en_tokenizer, train_te_tokenizer, test_en_tokenizer, test_te_tokenizer



def build_model(src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size=src_vocab_size,
                               tgt_vocab_size=tgt_vocab_size, 
                               src_seq_len=SEQ_LENGTH, 
                               tgt_seq_len=SEQ_LENGTH,
                               N=NO_LAYERS,
                               d_model=512)
    
    return model


def train():

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    train_dataset, test_dataset, train_en_tokenizer, train_te_tokenizer, test_en_tokenizer, test_te_tokenizer = get_ds()
    model = build_model(train_en_tokenizer.get_vocab_size(), train_te_tokenizer.get_vocab_size()).to(device)

    print("Using device: ", device)

    device = torch.device(device)


    Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_te_tokenizer.token_to_id('[PAD]'),     label_smoothing=0.1)


    initial_epoch = 0
    global_step = 0

    for epoch in range(EPOCH):
        torch.cuda.empty_cache()
        model.train()

        epoch_loss = 0
        batch_count = 0

        for batch in train_dataset:

            batch_count = batch_count + 1

            encoder_input = batch['encoder_input'].to(torch.long).to(device)
            decoder_input = batch['decoder_input'].to(torch.long).to(device)
            encoder_mask = batch['encoder_mask'].to(torch.long).to(device)
            decoder_mask = batch['decoder_mask'].to(torch.long).to(device)

            #print("starting encoder: ")
            #print("encoder_input size: ", encoder_input.shape)
            #print("decoder_input size: ", decoder_input.shape)
            #print("encoder_mask size", encoder_mask.shape)
            #print("decoder_mask size", decoder_mask.shape)

            encoder_output = model.encode(encoder_input, encoder_mask)
            #print("encoder_output shape is: ", encoder_output.shape)
            #print("starting decoder:")
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            proj_output = model.projection(decoder_output)


            label = batch['label'].to(torch.long).to(device)

            loss = loss_fn(proj_output.view(-1, train_te_tokenizer.get_vocab_size()), label.view(-1))

            print(f"Batch: {batch_count}, Global Step: {global_step}, Loss: {loss:.4f}")


            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch + 1} | Average Loss: {avg_loss:.4f}")

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)



if __name__ == '__main__':
    train()