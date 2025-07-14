import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

from rnn_model import GRUDecoder
from evaluate_model_helpers import *

# argument parser for command line arguments
parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model on the copy task dataset (phoneme-level only, no Redis required).')
parser.add_argument('--model_path', type=str, default='../data/t15_pretrained_rnn_baseline',
                    help='Path to the pretrained model directory (relative to the current working directory).')
parser.add_argument('--data_dir', type=str, default='../data/hdf5_data_final',
                    help='Path to the dataset directory (relative to the current working directory).')
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                    help='Evaluation type: "val" for validation set, "test" for test set. '
                         'If "test", ground truth is not available.')
parser.add_argument('--gpu_number', type=int, default=1,
                    help='GPU number to use for RNN model inference. Set to -1 to use CPU.')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory for results. If None, uses model_path.')
args = parser.parse_args()

# paths to model and data directories
# Note: these paths are relative to the current working directory
model_path = args.model_path
data_dir = args.data_dir
output_dir = args.output_dir if args.output_dir else model_path

# define evaluation type
eval_type = args.eval_type  # can be 'val' or 'test'. if 'test', ground truth is not available

# load model args
model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))

# set up gpu device
gpu_number = args.gpu_number
if torch.cuda.is_available() and gpu_number >= 0:
    if gpu_number >= torch.cuda.device_count():
        raise ValueError(f'GPU number {gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}')
    device = f'cuda:{gpu_number}'
    device = torch.device(device)
    print(f'Using {device} for model inference.')
else:
    if gpu_number >= 0:
        print(f'GPU number {gpu_number} requested but not available.')
    print('Using CPU for model inference.')
    device = torch.device('cpu')

# define model
model = GRUDecoder(
    neural_dim = model_args['model']['n_input_features'],
    n_units = model_args['model']['n_units'], 
    n_days = len(model_args['dataset']['sessions']),
    n_classes = model_args['dataset']['n_classes'],
    rnn_dropout = model_args['model']['rnn_dropout'],
    input_dropout = model_args['model']['input_network']['input_layer_dropout'],
    n_layers = model_args['model']['n_layers'],
    patch_size = model_args['model']['patch_size'],
    patch_stride = model_args['model']['patch_stride'],
)

# load model weights
checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), weights_only=False)
# rename keys to not start with "module." (happens if model was saved with DataParallel)
for key in list(checkpoint['model_state_dict'].keys()):
    checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
    checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
model.load_state_dict(checkpoint['model_state_dict'])  

# add model to device
model.to(device) 

# set model to eval mode
model.eval()

# load data for each session
test_data = {}
total_test_trials = 0
for session in model_args['dataset']['sessions']:
    files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]
    if f'data_{eval_type}.hdf5' in files:
        eval_file = os.path.join(data_dir, session, f'data_{eval_type}.hdf5')

        data = load_h5py_file(eval_file)
        test_data[session] = data

        total_test_trials += len(test_data[session]["neural_features"])
        print(f'Loaded {len(test_data[session]["neural_features"])} {eval_type} trials for session {session}.')
print(f'Total number of {eval_type} trials: {total_test_trials}')
print()

# put neural data through the pretrained model to get phoneme predictions (logits)
with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
    for session, data in test_data.items():

        data['logits'] = []
        data['pred_seq'] = []
        input_layer = model_args['dataset']['sessions'].index(session)
        
        for trial in range(len(data['neural_features'])):
            # get neural input for the trial
            neural_input = data['neural_features'][trial]

            # add batch dimension
            neural_input = np.expand_dims(neural_input, axis=0)

            # convert to torch tensor
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)

            # run decoding step
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data['logits'].append(logits)

            pbar.update(1)
pbar.close()

# convert logits to phoneme sequences and print them out
phoneme_results = {
    'session': [],
    'block': [],
    'trial': [],
    'true_sentence': [],
    'pred_phonemes': [],
    'true_phonemes': [],
}

for session, data in test_data.items():
    data['pred_seq'] = []
    for trial in range(len(data['logits'])):
        logits = data['logits'][trial][0]
        pred_seq = np.argmax(logits, axis=-1)
        # remove blanks (0)
        pred_seq = [int(p) for p in pred_seq if p != 0]
        # remove consecutive duplicates
        pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
        # convert to phonemes
        pred_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq]
        # add to data
        data['pred_seq'].append(pred_seq)

        # store results
        block_num = data['block_num'][trial]
        trial_num = data['trial_num'][trial]
        
        phoneme_results['session'].append(session)
        phoneme_results['block'].append(block_num)
        phoneme_results['trial'].append(trial_num)
        phoneme_results['pred_phonemes'].append(" ".join(pred_seq))
        
        if eval_type == 'val':
            sentence_label = data['sentence_label'][trial]
            true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
            true_seq = [LOGIT_TO_PHONEME[p] for p in true_seq]
            phoneme_results['true_sentence'].append(sentence_label)
            phoneme_results['true_phonemes'].append(" ".join(true_seq))
        else:
            phoneme_results['true_sentence'].append(None)
            phoneme_results['true_phonemes'].append(None)

        # print out the predicted sequences
        print(f'Session: {session}, Block: {block_num}, Trial: {trial_num}')
        if eval_type == 'val':
            print(f'Sentence label:      {sentence_label}')
            print(f'True phonemes:       {" ".join(true_seq)}')
        print(f'Predicted phonemes:  {" ".join(pred_seq)}')
        print()

# if using the validation set, calculate phoneme error rate (PER)
if eval_type == 'val':
    total_true_phonemes = 0
    total_phoneme_errors = 0

    phoneme_results['phoneme_errors'] = []
    phoneme_results['num_phonemes'] = []

    for i in range(len(phoneme_results['pred_phonemes'])):
        true_phonemes = phoneme_results['true_phonemes'][i].split()
        pred_phonemes = phoneme_results['pred_phonemes'][i].split()
        ed = editdistance.eval(true_phonemes, pred_phonemes)

        total_true_phonemes += len(true_phonemes)
        total_phoneme_errors += ed

        phoneme_results['phoneme_errors'].append(ed)
        phoneme_results['num_phonemes'].append(len(true_phonemes))

        print(f'{phoneme_results["session"][i]} - Block {phoneme_results["block"][i]}, Trial {phoneme_results["trial"][i]}')
        print(f'True phonemes:       {" ".join(true_phonemes)}')
        print(f'Predicted phonemes:  {" ".join(pred_phonemes)}')
        print(f'PER: {ed} / {len(true_phonemes)} = {ed / len(true_phonemes):.2f}%')
        print()

    print(f'Total true phonemes: {total_true_phonemes}')
    print(f'Total phoneme errors: {total_phoneme_errors}')
    print(f'Aggregate Phoneme Error Rate (PER): {100 * total_phoneme_errors / total_true_phonemes:.2f}%')

# write predicted phonemes to a csv file
output_file = os.path.join(output_dir, f'baseline_rnn_{eval_type}_predicted_phonemes_{time.strftime("%Y%m%d_%H%M%S")}.csv')
df_out = pd.DataFrame(phoneme_results)
df_out.to_csv(output_file, index=False)
print(f'Results saved to: {output_file}')

# also save a summary file with just the predictions for competition format
competition_output_file = os.path.join(output_dir, f'baseline_rnn_{eval_type}_competition_format_{time.strftime("%Y%m%d_%H%M%S")}.csv')
ids = [i for i in range(len(phoneme_results['pred_phonemes']))]
df_competition = pd.DataFrame({'id': ids, 'text': phoneme_results['pred_phonemes']})
df_competition.to_csv(competition_output_file, index=False)
print(f'Competition format results saved to: {competition_output_file}')

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print("Note: This evaluation only provides phoneme-level predictions.")
print("For word-level predictions, you would need to:")
print("1. Install and start Redis server")
print("2. Run the language model with: python language_model/language-model-standalone.py")
print("3. Use the original evaluate_model.py script")
print("="*80) 