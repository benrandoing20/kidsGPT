from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

def train_tokenizer(input_path, save_path="tokenizer/tokenizer.json", vocab_size=30_000):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<pad>", "<s_grade=0>"] + [f"<s_grade={i}>" for i in range(1,13)])
    tokenizer.train([input_path], trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.enable_truncation(max_length=1024)
    tokenizer.save(save_path)
    print("Saved tokenizer:", save_path)

if __name__ == "__main__":
    import sys
    train_tokenizer(sys.argv[1])
