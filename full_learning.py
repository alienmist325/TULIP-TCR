
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

    # Required parameters
    parser.add_argument(
        "--train_dir",
        default=None,
        type=str,
        required=True,
        help="The train data dir. Should contain the .csv files (or other data files) for the task.",
    )
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
        "--save",
        default=None,
        type=str,
        help="path to save the model" ,
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="batch_size" ,
    )
    parser.add_argument(
        "--num_epochs",
        default=300,
        type=int,
        help="numbers of epochs" ,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="weight decay" ,
    )
    parser.add_argument(
        "--lr",
        default=0.0001,
        type=float,
        help="learning rate" ,
    )
    parser.add_argument(
        "--to_track",
        default=None,
        type=str,
        help="numpy array of the string "
    )
    parser.add_argument("--flex", action="store_true", help="Whether to run training.")
    parser.add_argument("--skipMiss", action="store_true", help="Whether to run training.")
    parser.add_argument("--mhcX", action="store_true", help="Whether to run training.")
    parser.add_argument("--below20", action="store_true", help="Whether to run training.")
    args = parser.parse_args()

    with open(args.modelconfig, "r") as read_file:
        print("loading hyperparameter")
        modelconfig = json.load(read_file)

    if args.flex:
        if args.skipMiss:
            from src.multiTransFlex2 import ED_BertForSequenceClassification, TCRDataset, BertLastPooler, unsupervised_auc, train_unsupervised, eval_unsupervised, MyMasking, ED_MultiTransformerModel
        else:
            from src.multiTransFlex import ED_BertForSequenceClassification, TCRDataset, BertLastPooler, unsupervised_auc, train_unsupervised, eval_unsupervised, MyMasking, ED_MultiTransformerModel
    else:
        if args.skipMiss:
            from src.multiTrans2 import ED_BertForSequenceClassification, TCRDataset, BertLastPooler, unsupervised_auc, train_unsupervised, eval_unsupervised, MyMasking, ED_MultiTransformerModel
        else:
            from src.multiTrans import ED_BertForSequenceClassification, TCRDataset, BertLastPooler, unsupervised_auc, train_unsupervised, eval_unsupervised, MyMasking, ED_MultiTransformerModel




    #     from src.multiTransFlex import ED_BertForSequenceClassification, TCRDataset, BertLastPooler, unsupervised_auc, train_unsupervised, eval_unsupervised, MyMasking, ED_MultiTransformerModel
    # elif args.flex2:
    #     print("flex2")
    #     from src.multiTransFlex2 import ED_BertForSequenceClassification, TCRDataset, BertLastPooler, unsupervised_auc, train_unsupervised, eval_unsupervised, MyMasking, ED_MultiTransformerModel
    # else:
    #     from src.multiTrans import ED_BertForSequenceClassification, TCRDataset, BertLastPooler, unsupervised_auc, train_unsupervised, eval_unsupervised, MyMasking, ED_MultiTransformerModel





    wandb.login()
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    do_reweight = False
    reweight_pc = 0.0
    test_path = args.test_dir
    train_path = args.train_dir

    if args.below20:
        train = pd.read_csv("../NetTCR/dataNew/Full_train_pretune_2.csv")
        peplist = train["peptide"].value_counts()
        peplistout = list(peplist[peplist<20].index)
        peplistin = list(peplist[peplist>19].index)
        test_path =  "../NetTCR/dataNew/Full_below20withneg_2.csv"
        train_path ="../NetTCR/dataNew/Full_train_pretune_mhcX_2.csv"
        datasetTrainFull = TCRDataset("../NetTCR/dataNew/Full_train_pretune_mhcX_2.csv", tokenizer, device,excluded_peptide=peplistout, mhctok=mhctok)
        target_peptidesFinal = peplistout

    # mhcX = False
    # hideMHC = True
    tokenizer = AutoTokenizer.from_pretrained("aatok/")#lightonai/RITA_l")#/content/drive/MyDrive/phd/TCREp/")
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

    print(f'The model has {count_parameters(model):,} trainable parameters')

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    if args.load:
        checkpoint = torch.load(args.load+"/pytorch_model.bin")
        model.load_state_dict(checkpoint)
        print("loaded")

    if do_reweight:
        model.set_reweight()
    else:
        model.reweight=False


    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("pti",tokenizer.pad_token_id)
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id,  reduction='sum')


    datasetTrainFull = TCRDataset(train_path, tokenizer, device, mhctok=mhctok)
    train_dataloaderFull = torch.utils.data.DataLoader(dataset=datasetTrainFull, batch_size=args.batch_size, shuffle=True, collate_fn=datasetTrainFull.all2allmhc_collate_function) 
    datasetValidFinal = TCRDataset(test_path, tokenizer, device,target_binder=False, mhctok=mhctok)
    valid_dataloaderFinal = torch.utils.data.DataLoader(dataset=datasetValidFinal, batch_size=args.batch_size, shuffle=True, collate_fn=datasetValidFinal.all2allmhc_collate_function) 


    #valid_dataloaderFull = torch.utils.data.DataLoader(dataset=datasetValidFull, batch_size=1, shuffle=True, collate_fn=datasetValidFull.all2allmhc_collate_function) 


    masker = MyMasking(tokenizer, mlm_probability = 0.15)


    config_dict = {
        'num_attn_heads':  modelconfig["num_attn_heads"],
        'batch_size':  args.batch_size,
        'num_hidden_layers':  modelconfig["num_hidden_layers"],
        'hidden_size':  modelconfig["hidden_size"],
        'dropout':  0.1,
        'lr':  args.lr,
        'weight_decay': args.weight_decay,
        "do_reweight": do_reweight,
        "reweight_pc":reweight_pc,
        "mhcX": args.mhcX,
        "flex":args.flex,
        "skipMiss":args.skipMiss,
        "below20":args.below20,
    }

    wandb.init(project="pm53", entity="barthelemymp", config=config_dict)
    trigger_sync() 
    wandb.config.update(config_dict) 
    trigger_sync() 

    if args.to_track:
        target_peptidesFinal = np.load(args.to_track)
    else:
        target_peptidesFinal = ['GILGFVFTL',
        'NLVPMVATV',
        'GLCTLVAML',
        'ELAGIGILTV',
        'YLQPRTFLL',
        'LLWNGPMAV',
        'CINGVCWTV',
        'FLYALALLL',
        'LLYDANYFL',
        'SLFNTVATLY',
        'ALWEIQQVV',
        'RTLNAWVKV',
        'RLPGVLPRA',
        'LLFGYPVYV',
        'LLDFVRFMGV',
        'FLASKIGRLV',
        'FLPRVFSAV',
        'ALSKGVHFV',
        'FLHVTYVPA',
        'KLVAMGINAV',
        'KSVNITFEL',
        'VMVELVAEL',
        'CLAVHECFV',
        'FLLNKEMYL',
        'RLARLALVL',
        'AAGIGILTV',
        'AMQTMLFTM',
        'YLNDHLEPWI',
        'RIMTWLDMV',
        'MMILSDDAV',
        'FIAGLIAIV',
        'KVLEYVIKV',
        'FLNRFTTTL',
        'YIDIGDYTV',
        'RLGPVQNEV',
        'ALYGFVPVL',
        'KLSYGIATV',
        'EAAGIGILTV',
        'CVNGSCFTV',
        'YMPYFFTLL',
        'ILGFVFTLT',
        'WLDMVDTSL',
        'WLLWPVTLA',
        'RMFPNAPYL',
        'TLLFLMSFT',
        'HLMSFPQSA',
        'KVYPIILRL',
        'QVILLNKHI',
        'VFLVLLPLV',
        'ALLPGLPAA',
        'KTWGQYWQV',
        'SWMESEFRV',
        'TILTSLLVL',
        'NLHPDSATL',
        'KLNEEIAII',
        'WLTNIFGTV',
        'LITGRLQSL',
        'TLMNVITLV',
        'KLNIKLLGV',
        'SLPGVFCGV',
        'SLENVAFNV',
        'FLALCADSI',
        'FLIGCNYLG',
        'YTVSCLPFT',
        'SACVLAAEC',
        'ALYYPSARI',
        'VLLGVKLFGV',
        'SVLYYQNNV',
        'YLGGMSYYC',
        'SLLMPILTL',
        'YMRSLKVPA',
        'QMAPISAMV',
        'KLSALGINAV',
        'TLEYMDWLV',
        'IQPGQTFSV',
        'SMMILSDDA',
        'RLYLDAYNM',
        'KLYGLDWAEL',
        'SGGGETALA',
        'KLPDDFTGCV',
        'HMTEVVRHC',
        'FLAHIQWMV',
        'NVLTLVYKV',
        'MLNPNYEDL',
        'FLNGSCGSV',
        'LLLTILTSL',
        'KLIEYTDFA',
        'MQVESDDYI',
        'ALNTPKDHI',
        'ILAYCNKTV',
        'KIILFLALI',
        'SLLSVLLSM',
        'RTIKVFTTV',
        'KDNVILLNK',
        'KLWAQCVQL',
        'MLDLQPETT',
        'TALALLLLD',
        'LMWLSYFIA',
        'LLLEWLAMA',
        'KLPDDFMGC',
        'KLFIRQEEV',
        'YVWKSYVHV',
        'FLPGVYSVI',
        'LPDDFMGCV',
        'NLIDSYFVV',
        'GLALYYPSA',
        'VLNDILSRL',
        'TLIGDCATV',
        'KQIYKTPPI',
        'KQLSSNFGA',
        'NLNCCSVPV',
        'IMDQVPFSV',
        'KLQCVDLHV',
        'CLEASFNYL',
        'KLQFTSLEI',
        'YLNTLTLAV',
        'LLFNKVTLA',
        'FVAAIFYLI',
        'RLCAYCCNI',
        'KLKDCVMYA',
        'ALDPHSGHFV',
        'QVVSDIDYV',
        'YLNSTNVTI',
        'LMGHFAWWT',
        'AIFYLITPV',
        'SLINTLNDL',
        'AVIKTLQPV',
        'DLFMRIFTI',
        'VLWAHGFEL',
        'KLFEFLVYGV',
        'RLFARTRSM',
        'YADVFHLYL',
        'IMLCCMTSC',
        'FLLPSLATV',
        'SMWALVISV',
        'FTVLCLTPV',
        'FVVFLLVTL',
        'DRLNQLESK',
        'FLCLFLLPS',
        'SLLMWITQV',
        'YLLEMLWRL',
        'TMCDIRQLL',
        'DQVILLNKH',
        'YVDNSSLTI',
        'KMDYFSGQL',
        'IKLDDKDPQ',
        'LIDFYLCFL',
        'YGFQPTNGV']

    target_peptidesFinal = pd.read_csv(test_path)["peptide"].unique()

    if args.below20:
        test_path =  "../NetTCR/dataNew/Full_below20withneg_2.csv"
        train_path ="../NetTCR/dataNew/Full_train_pretune_mhcX_2.csv"
        datasetTrainFull = TCRDataset("../NetTCR/dataNew/Full_train_pretune_mhcX_2.csv", tokenizer, device,excluded_peptide=peplistout, mhctok=mhctok)
        train_dataloaderFull = torch.utils.data.DataLoader(dataset=datasetTrainFull, batch_size=args.batch_size, shuffle=True, collate_fn=datasetTrainFull.all2allmhc_collate_function) 
        target_peptidesFinal = peplistout
        datasetValidFinal = TCRDataset(test_path, tokenizer, device,target_binder=False, mhctok=mhctok)
        valid_dataloaderFinal = torch.utils.data.DataLoader(dataset=datasetValidFinal, batch_size=args.batch_size, shuffle=True, collate_fn=datasetValidFinal.all2allmhc_collate_function) 



    for epoch in range(args.num_epochs+1):
        if epoch%20==0:
            aucelist = []
            aucalist = []
            aucblist = []
            for target_peptide in target_peptidesFinal:
                datasetPetideSpecific= TCRDataset(test_path, tokenizer, device,target_peptide=target_peptide, mhctok=mhctok)
                dataloaderPetideSpecific = torch.utils.data.DataLoader(dataset=datasetPetideSpecific, batch_size=1, shuffle=True, collate_fn=datasetValidFinal.all2allmhc_collate_function) 
                print(target_peptide)
                auca, aucb, auce = unsupervised_auc(model, dataloaderPetideSpecific, tokenizer.pad_token_id)
                wandb.log({target_peptide+"_a":auca, target_peptide+"_b":aucb,target_peptide+"_e":auce, "epochT":epoch})
                trigger_sync() 
                aucelist.append(auce)
                aucalist.append(auca)
                aucblist.append(aucb)
            wandb.log({"avg_e":np.mean(aucelist), "avg_a":np.mean(aucalist),"avg_b":np.mean(aucblist), "epochT":epoch})
            trigger_sync() 

        print("Starting epoch", epoch+1)
        epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE = train_unsupervised(model, optimizer, masker, train_dataloaderFull, criterion)
        print(epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE)
        wandb.log({"epoch_lm_lossAu": epoch_lm_lossA, "epoch_lm_lossBu":epoch_lm_lossB ,"epoch_lm_lossEu":epoch_lm_lossE ,"epoch_mlm_lossAu":epoch_mlm_lossA ,"epoch_mlm_lossBu":epoch_mlm_lossB ,"epoch_mlm_lossEu":epoch_mlm_lossE, "epochT":epoch})
        trigger_sync() 
        if epoch%10==0:
            with  torch.no_grad():
                epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE = eval_unsupervised(model, masker, valid_dataloaderFinal, criterion)
                print(epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE)
                auca, aucb, auce = unsupervised_auc(model, valid_dataloaderFinal, tokenizer.pad_token_id)
                wandb.log({"epoch_lm_lossAu_val": epoch_lm_lossA, "epoch_lm_lossBu_val":epoch_lm_lossB ,"epoch_lm_lossEu_val":epoch_lm_lossE ,"epoch_mlm_lossAu_val":epoch_mlm_lossA ,"epoch_mlm_lossBu_val":epoch_mlm_lossB ,"epoch_mlm_lossEu_val":epoch_mlm_lossE,"auca":auca, "aucb":aucb, "auce":auce, "epochT":epoch})
                trigger_sync() 
                if epoch%10==0:
                    if args.save:
                        model.save_pretrained(args.save + str(epoch))



if __name__ == "__main__":
    main()
