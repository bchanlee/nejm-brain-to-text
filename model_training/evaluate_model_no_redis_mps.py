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
parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model on the copy task dataset (phoneme-level only, using Apple MPS if available, no Redis required).')
parser.add_argument('--model_path', type=str, default='../data/t15_pretrained_rnn_baseline',
                    help='Path to the pretrained model directory (relative to the current working directory).')
parser.add_argument('--data_dir', type=str, default='../data/hdf5_data_final',
                    help='Path to the dataset directory (relative to the current working directory).')
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                    help='Evaluation type: "val" for validation set, "test" for test set. '
                         'If "test", ground truth is not available.')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory for results. If None, uses model_path.')
args = parser.parse_args()

# paths to model and data directories
model_path = args.model_path
data_dir = args.data_dir
output_dir = args.output_dir if args.output_dir else model_path

eval_type = args.eval_type  # can be 'val' or 'test'. if 'test', ground truth is not available

model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args_5_sessions.yaml'))

# Set up device: Prefer MPS if available, else CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
    print('Using Apple MPS (Metal Performance Shaders) for model inference.')
else:
    device = torch.device('cpu')
    print('Using CPU for model inference.')

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

checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), map_location=device, weights_only=False)

# Handle DataParallel state dict (remove prefixes)
state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('_orig_mod.'):
        new_key = key[10:]  # Remove '_orig_mod.' prefix
    elif key.startswith('module.'):
        new_key = key[7:]   # Remove 'module.' prefix
    else:
        new_key = key
    new_state_dict[new_key] = value

# Filter state dict to only include weights for the sessions we're using
filtered_state_dict = {}
current_sessions = model_args['dataset']['sessions']
for key, value in new_state_dict.items():
    if key.startswith('day_weights.') or key.startswith('day_biases.'):
        # Extract the day index from the key
        day_idx = int(key.split('.')[1])
        # Only include if this day index corresponds to one of our sessions
        if day_idx < len(current_sessions):
            filtered_state_dict[key] = value
    else:
        # Include all non-day-specific weights
        filtered_state_dict[key] = value

model.load_state_dict(filtered_state_dict)  

model.to(device) 
model.eval()

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

with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
    for session, data in test_data.items():
        data['logits'] = []
        data['pred_seq'] = []
        input_layer = model_args['dataset']['sessions'].index(session)
        for trial in range(len(data['neural_features'])):
            neural_input = data['neural_features'][trial]
            neural_input = np.expand_dims(neural_input, axis=0)
            # MPS does not support bfloat16, so use float32
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data['logits'].append(logits)
            pbar.update(1)
pbar.close()

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
        pred_seq = [int(p) for p in pred_seq if p != 0]
        pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
        pred_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq]
        data['pred_seq'].append(pred_seq)
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
        print(f'Session: {session}, Block: {block_num}, Trial: {trial_num}')
        if eval_type == 'val':
            print(f'Sentence label:      {sentence_label}')
            print(f'True phonemes:       {" ".join(true_seq)}')
        print(f'Predicted phonemes:  {" ".join(pred_seq)}')
        print()

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

output_file = os.path.join(output_dir, f'baseline_rnn_{eval_type}_predicted_phonemes_{time.strftime("%Y%m%d_%H%M%S")}.csv')
df_out = pd.DataFrame(phoneme_results)
df_out.to_csv(output_file, index=False)
print(f'Results saved to: {output_file}')

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