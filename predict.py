
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import pandas as pd
import sys

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
#from data import TranslationDataset
from transformers import BertTokenizerFast, BertTokenizer
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel, BertLMHeadModel, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score

import sys
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import os


from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead, SequenceClassifierOutput
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import ModelOutput

from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
import warnings
from torch.profiler import profile, record_function, ProfilerActivity
import wandb

import argparse



from wandb_osh.hooks import TriggerWandbSyncHook

wandb.login()


torch.manual_seed(0)




def main():
    trigger_sync = TriggerWandbSyncHook()
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--test_dir",
        default=None,
        type=str,
        required=True,
        help="The test data dir. Should contain the .fasta files (or other data files) for the task.",
    )
    parser.add_argument(
        "--modelconfig",
        type=str,
        help="path to json including the config of the model" ,
    )
    parser.add_argument(
        "--load",
        default=None,
        type=str,
        help="path to the model pretrained to load" ,
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="path to save results" ,
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="batch_size" ,
    )



    args = parser.parse_args()

    with open(args.modelconfig, "r") as read_file:
        print("loading hyperparameter")
        modelconfig = json.load(read_file)
        from src.multiTrans import ED_BertForSequenceClassification, TCRDataset, BertLastPooler, unsupervised_auc, train_unsupervised, eval_unsupervised, MyMasking, ED_MultiTransformerModel


    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    test_path = args.test_dir
    train_path = args.train_dir



    tokenizer = AutoTokenizer.from_pretrained("aatok/")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<MIS>'})
        
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '<CLS>'})

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<EOS>'})

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<MASK>'})



    from tokenizers.processors import TemplateProcessing
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <EOS>",
        pair="<CLS> $A <MIS> $B:1 <EOS>:1",
        special_tokens=[
            ("<EOS>", 2),
            ("<CLS>", 3),
            ("<MIS>", 4),
        ],
    )

    mhctok = AutoTokenizer.from_pretrained("mhctok/")

    vocabsize = len(tokenizer._tokenizer.get_vocab())
    mhcvocabsize = len(mhctok._tokenizer.get_vocab())
    print("Loading models ..")
    # vocabsize = encparams["vocab_size"]
    max_length = 100
    encoder_config = BertConfig(vocab_size = vocabsize,
                        max_position_embeddings = max_length, # this shuold be some large value
                        num_attention_heads = modelconfig["num_attn_heads"],
                        num_hidden_layers = modelconfig["num_hidden_layers"],
                        hidden_size = modelconfig["hidden_size"],
                        type_vocab_size = 1,
                        pad_token_id =  tokenizer.pad_token_id)

    encoder_config.mhc_vocab_size  =mhcvocabsize

    encoderA = BertModel(config=encoder_config)
    encoderB = BertModel(config=encoder_config)
    encoderE = BertModel(config=encoder_config)

    max_length = 100
    decoder_config = BertConfig(vocab_size = vocabsize,
                        max_position_embeddings = max_length, # this shuold be some large value
                        num_attention_heads = modelconfig["num_attn_heads"],
                        num_hidden_layers = modelconfig["num_hidden_layers"],
                        hidden_size = modelconfig["hidden_size"],
                        type_vocab_size = 1,
                        is_decoder=True, 
                        pad_token_id =  tokenizer.pad_token_id)    # Very Important
    
    decoder_config.add_cross_attention=True

    decoderA = ED_BertForSequenceClassification(config=decoder_config) #BertForMaskedLM
    decoderA.pooler = BertLastPooler(config=decoder_config)
    decoderB = ED_BertForSequenceClassification(config=decoder_config) #BertForMaskedLM
    decoderB.pooler = BertLastPooler(config=decoder_config)
    decoderE = ED_BertForSequenceClassification(config=decoder_config) #BertForMaskedLM
    decoderE.pooler = BertLastPooler(config=decoder_config)
    # Define encoder decoder model
    model = ED_MultiTransformerModel(encoderA=encoderA,encoderB=encoderB,encoderE=encoderE, decoderA=decoderA, decoderB=decoderB, decoderE=decoderE)

    def count_parameters(mdl):
        return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    if args.load:
        checkpoint = torch.load(args.load+"/pytorch_model.bin")
        model.load_state_dict(checkpoint)
        print("loaded")
    model.to(device)
    target_peptidesFinal = pd.read_csv(test_path)["peptide"].unique()

    for target_peptide in target_peptidesFinal:
        results = pd.DataFrame(columns=["CDR3a", "CDR3b", "peptide", "rank"])
        datasetPetideSpecific= TCRDataset(test_path, tokenizer, device,target_peptide=target_peptide, mhctok=mhctok)
        print(target_peptide)
        scores = -1*np.array(get_logproba(datasetPetideSpecific, model, ignore_index =  tokenizer.pad_token_id))
        ranks = np.argsort(scores)[::-1]
        results["CDR3a"] = datasetPetideSpecific.alpha
        results["CDR3b"] = datasetPetideSpecific.beta
        results["peptide"] = target_peptide
        results["rank"] = ranks
        results.to_csv(args.save + target_peptide+".csv")


if __name__ == "__main__":
    main()