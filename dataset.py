import torch
from torch.utils.data import Dataset


class BILangDataset(Dataset):

    def __init__(self, ds, src_tokens, tgt_tokens, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        self.src_lang  = src_lang
        self.tgt_lang =  tgt_lang
        self.seq_len = seq_len

        self.sos_token = src_tokens.token_to_id('[SOS]')
        self.eos_token = src_tokens.token_to_id('[EOS]')
        self.pad_token = src_tokens.token_to_id('[PAD]')



    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, index):
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair[self.src_lang]
        tgt_text = src_tgt_pair[self.tgt_lang]

        enc_input_tokens = self.src_tokens.encode(src_text).ids
        dec_input_tokens = self.tgt_tokens.encode(tgt_text).ids
        from train import SEQ_LENGTH

        enc_input_tokens = enc_input_tokens[:SEQ_LENGTH]
        dec_input_tokens = dec_input_tokens[:SEQ_LENGTH]

        enc_with_pdding_num = self.seq_len - len(enc_input_tokens) - 2
        dec_with_pdding_num = self.seq_len - len(dec_input_tokens) - 1

        # #print("len(enc_input_tokens):", len(enc_input_tokens))
        # #print("len(dec_input_tokens):", len(dec_input_tokens))
        # #print("enc_with_pdding_num: ", enc_with_pdding_num)
        # #print("dec_with_pdding_num:", dec_with_pdding_num)


        if enc_with_pdding_num < 0 or dec_with_pdding_num < 0:
            raise ValueError("sentense is too long")


        encoder_input = torch.cat([
    torch.tensor([self.sos_token], dtype=torch.long),
    torch.tensor(enc_input_tokens, dtype=torch.long),
    torch.tensor([self.eos_token], dtype=torch.long),
    torch.full((enc_with_pdding_num,), self.pad_token, dtype=torch.long)
])

        decoder_input = torch.cat([
    torch.tensor([self.sos_token], dtype=torch.long),
    torch.tensor(dec_input_tokens, dtype=torch.long),
    torch.full((dec_with_pdding_num,), self.pad_token, dtype=torch.long)
])

        label = torch.cat([
    torch.tensor(dec_input_tokens, dtype=torch.long),
    torch.tensor([self.eos_token], dtype=torch.long),
    torch.full((dec_with_pdding_num,), self.pad_token, dtype=torch.long)
])

        #print("decoder input size:", decoder_input.shape)

        return {
            "encoder_input": encoder_input.long(),
            "decoder_input": decoder_input.long(),
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            "label": label.to(torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def casual_mask(size):
    #print("input size of casual mask: ", size)
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    #print("casual mask shape is :", mask.shape)
    return mask == 0