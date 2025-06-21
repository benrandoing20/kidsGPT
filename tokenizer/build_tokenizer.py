from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

def train_tokenizer_bytelevel(input_path, save_path="tokenizer/tokenizer_bytelevel.json", vocab_size=30_000):
    """
    ByteLevel tokenizer - this will produce tokens with Ġ characters.
    The Ġ character (Unicode 0x0120) represents word boundaries (spaces).
    This is normal behavior for ByteLevel tokenization used by models like GPT-2.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    # ByteLevel pre-tokenizer adds Ġ characters for word boundaries
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<pad>", "<s_grade=0>"] + [f"<s_grade={i}>" for i in range(1,13)])
    tokenizer.train([input_path], trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.enable_truncation(max_length=1024)
    tokenizer.save(save_path)
    print("Saved ByteLevel tokenizer:", save_path)
    print("Note: Ġ characters in vocabulary are normal for ByteLevel tokenization")

if __name__ == "__main__":
    import sys
    train_tokenizer_bytelevel(sys.argv[1]) 