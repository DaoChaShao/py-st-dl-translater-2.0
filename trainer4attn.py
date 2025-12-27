#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/16 21:42
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   trainer4attn.py
# @Desc     :   

from pathlib import Path
from torch import optim, nn

from src.configs.cfg_rnn import CONFIG4RNN
from src.configs.cfg_types import Tokens, SeqNets, SeqStrategies, SeqMergeMethods, AttnScorer
from src.configs.parser import set_argument_parser
from src.trainers.trainer4seq2seq_attn import TorchTrainer4SeqToSeq
from src.nets.seq2seq_attn_gru import SeqToSeqGRUWithAttn
from src.nets.seq2seq_attn_lstm import SeqToSeqLSTMWithAttn
from src.nets.seq2seq_attn_rnn import SeqToSeqRNNWithAttn
from src.utils.stats import load_json
from src.utils.PT import TorchRandomSeed

from pipeline.prepper import prepare_data


def main() -> None:
    """ Main Function """
    # Set up argument parser
    args = set_argument_parser()

    with TorchRandomSeed("Chinese to English (Seq2Seq) with attentions Translation"):
        # Get the dictionary
        dic_cn: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY_CN)
        dictionary_cn = load_json(dic_cn) if dic_cn.exists() else print("Dictionary file not found.")
        dic_en: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY_EN)
        dictionary_en = load_json(dic_en) if dic_en.exists() else print("Dictionary file not found.")
        # print(dictionary_cn[Tokens.PAD])
        # print(dictionary_en[Tokens.PAD])

        # Get the input size and number of classes
        vocab_size4cn: int = len(dictionary_cn)
        vocab_size4en: int = len(dictionary_en)
        print(vocab_size4cn, vocab_size4en)

        # Get the data
        train, valid = prepare_data()

        # Initialize model
        model = SeqToSeqGRUWithAttn(
            # model = SeqToSeqLSTMWithAttn(
            # model = SeqToSeqRNNWithAttn(
            vocab_size_src=vocab_size4cn,
            vocab_size_tgt=vocab_size4en,
            embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
            hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
            num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
            dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT_RATIO,
            bidirectional=False,
            accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
            PAD_SRC=dictionary_cn[Tokens.PAD],
            PAD_TGT=dictionary_en[Tokens.PAD],
            SOS=dictionary_cn[Tokens.SOS],
            EOS=dictionary_cn[Tokens.EOS],
            merge_method=SeqMergeMethods.CONCAT,
            teacher_forcing_ratio=0.5,
            use_attention=True,
            attn_scorer=AttnScorer.SCALED_DOT_PRODUCT
        )
        # Setup optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=args.alpha, weight_decay=CONFIG4RNN.HYPERPARAMETERS.DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss(ignore_index=dictionary_en[Tokens.PAD])
        model.summary()
        """
        ****************************************************************
        Model Summary for SeqToSeqGRUWithAttn
        ----------------------------------------------------------------
        - Source Vocabulary Size: 5235
        - Target Vocabulary Size: 3189
        - Embedding Dimension:    128
        - Hidden Size:            256
        - Number of Layers:       2
        - Dropout Rate:           0.5
        - Bidirectional:          True
        - Device:                 cpu
        - PAD Token (Source):     0
        - PAD Token (Target):     0
        - SOS Token:              2
        - EOS Token:              3
        ----------------------------------------------------------------
        Total parameters:         4,495,989
        Trainable parameters:     4,495,989
        Non-trainable parameters: 0
        ****************************************************************
        ****************************************************************
        Model Summary for SeqToSeqLSTMWithAttn
        ----------------------------------------------------------------
        - Source Vocabulary Size: 5235
        - Target Vocabulary Size: 3189
        - Embedding Dimension:    128
        - Hidden Size:            256
        - Number of Layers:       2
        - Dropout Rate:           0.5
        - Bidirectional:          True
        - Device:                 cpu
        - PAD Token (Source):     0
        - PAD Token (Target):     0
        - SOS Token:              2
        - EOS Token:              3
        ----------------------------------------------------------------
        Total parameters:         5,318,261
        Trainable parameters:     5,318,261
        Non-trainable parameters: 0
        ****************************************************************
        ****************************************************************
        Model Summary for SeqToSeqRNNWithAttn
        ----------------------------------------------------------------
        - Source Vocabulary Size: 5235
        - Target Vocabulary Size: 3189
        - Embedding Dimension:    128
        - Hidden Size:            256
        - Number of Layers:       2
        - Dropout Rate:           0.5
        - Bidirectional:          True
        - Device:                 cpu
        - PAD Token (Source):     0
        - PAD Token (Target):     0
        - SOS Token:              2
        - EOS Token:              3
        ----------------------------------------------------------------
        Total parameters:         2,851,445
        Trainable parameters:     2,851,445
        Non-trainable parameters: 0
        ****************************************************************
        """

        # Setup trainer
        trainer = TorchTrainer4SeqToSeq(
            vocab_size_tgt=vocab_size4en,
            model=model,
            optimiser=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            PAD=dictionary_en[Tokens.PAD],
            SOS=dictionary_en[Tokens.SOS],
            EOS=dictionary_en[Tokens.EOS],
            decode_strategy=SeqStrategies.BEAM_SEARCH,
            beam_width=CONFIG4RNN.PARAMETERS.BEAM_SIZE,
            accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
        )
        # Train the model
        trainer.fit(
            train_loader=train,
            valid_loader=valid,
            epochs=args.epochs,
            model_save_path=str(CONFIG4RNN.FILEPATHS.SAVED_NET),
            log_name=f"{SeqNets.GRU}-{SeqStrategies.BEAM_SEARCH}-{SeqMergeMethods.CONCAT}",
        )
        """
        ****************************************************************
        Training Log for SeqToSeqGRU with attentions
        ----------------------------------------------------------------
        "bid": true, "epoch": 51/100, "strategy": "beam", "merge": "concat", "bleu": 0.2368, "rouge": 0.6262, "attn": true, "score": "bahdanau"
        "bid": true, "epoch": 57/100, "strategy": "beam", "merge": "concat", "bleu": 0.1840, "rouge": 0.5721, "attn": true, "score": "dot"
        "bid": true, "epoch": 49/100, "strategy": "beam", "merge": "concat", "bleu": 0.2291, "rouge": 0.6174, "attn": true, "score": "scaled_dot"
        ****************************************************************
        ****************************************************************
        Training Log for SeqToSeqGRU with attention and without attention
        ----------------------------------------------------------------
        "bid": true, "epoch": 57/100, "strategy": "beam", "merge": "concat", "bleu": 0.1840, "rouge": 0.5721, "attn": true, "score": "dot"
        "bid": true, "epoch": 54/100, "strategy": "beam", "merge": "concat", "bleu": 0.1613, "rouge": 0.5433, "attn": false
        ****************************************************************
        ****************************************************************
        Training Log for SeqToSeqLSTM with attention and without attention
        ----------------------------------------------------------------
        "bid": true, "epoch": 63/100, "strategy": "beam", "merge": "concat", "bleu": 0.1782, "rouge": 0.5722, "attn": true, "score": "dot"
        "bid": true, "epoch": 64/100, "strategy": "beam", "merge": "concat", "bleu": 0.1442, "rouge": 0.5235, "attn": false
        ****************************************************************
        ****************************************************************
        Training Log for SeqToSeqRNN with attention and without attention
        ----------------------------------------------------------------
        "bid": true, "epoch": 16/100, "strategy": "beam", "merge": "concat", "bleu": 0.0808, "rouge": 0.4785, "attn": true, "score": "dot"
        "bid": true, "epoch": 15/100, "strategy": "beam", "merge": "concat", "bleu": 0.0938, "rouge": 0.4875, "attn": false
        ****************************************************************
        """


if __name__ == "__main__":
    main()
