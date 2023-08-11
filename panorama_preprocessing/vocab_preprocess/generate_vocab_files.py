import sentencepiece as spm

spm.SentencePieceTrainer.Train(input="./split_vocab.txt",
                               model_prefix='r2r_vocab_2000',
                               vocab_size=2000,
                               pad_id=0,                
                               unk_id=1,
                               bos_id=2,
                               eos_id=3
                               )