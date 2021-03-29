import os
import argparse
import subprocess
import json
import sys
import ast


def main():
    print('extracting arguments')
    hp_list = []
    hyperparameters = sys.argv[1:]
    parser = argparse.ArgumentParser()
    aug_arguments = []
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    sm_hps = ast.literal_eval(os.environ.get('SM_HPS'))
    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--vocab', type=str, default=os.environ.get('SM_CHANNEL_VOCABULARY'))
    #parser.add_argument('--dev', type=str, default=os.environ.get('SM_CHANNEL_DEV'))
    #parser.add_argument('--alphabet-config-path', type=str)
    parser.add_argument('--train_file', type=str, default='train.csv')
    parser.add_argument('--test_file', type=str, default='test.csv')
    parser.add_argument('--vocab_file', type=str, default='alphabet.txt')
    #parser.add_argument('--n-hidden', type=str, default='16') 
    parser.add_argument('--test_batch_size', type=str, default='16') 
    #parser.add_argument('--drop-source-layers', type=str, default='1')
    parser.add_argument('--train_batch_size', type=str, default='16') 
    #parser.add_argument('--learning-rate', type=str, default='0.0001')
    #parser.add_argument('--dropout-rate', type=str, default='0.10')
    parser.add_argument('--epochs', type=str, default='100')
    parser.add_argument('--augmentation', default=str)
    #parser.add_argument('--spec-augmentation', type=bool)
    
    
    # Data, model, and output directories
    
    
    
    args, _ = parser.parse_known_args()
    train_file = os.path.join(args.train, args.train_file)
    test_file = os.path.join(args.test, args.test_file)
    print(args.vocab_file)
    vocab_file = os.path.join(args.vocab, args.vocab_file)
    print("Using Vocabulary file: {}".format(vocab_file))
    print(args.augmentation)
    train_cmd = ["python", "/opt/ml/code/DeepSpeech.py",
        "--alphabet_config_path", vocab_file, 
        "--export_dir","/opt/ml/model",
        "--show_progressbar",
        "--automatic_mixed_precision", "True",
        "--train_files", train_file,
        "--test_files", test_file ,                 
        "--epochs", args.epochs,
        "--train_batch_size", args.train_batch_size,
        "--test_batch_size", args.test_batch_size, 
        "--train_cudnn", "True",
        "--checkpoint_dir", "/opt/ml/input/data/checkpoint"
    ]
    if args.augmentation is not None:
        for line in ast.literal_eval(os.environ.get('SM_HP_AUGMENTATION')):
            aug_arguments.append("--augment")
            aug_arguments.append(str(line))
        del sm_hps['augmentation']
        train_cmd = train_cmd + aug_arguments

        for key, value in sm_hps.items():
            k = '--'+str(key)
            hp_list.append(k)
            v = str(value)
            hp_list.append(v)

    
    redirect_output = ['1>&2']
    print('running subprocess: {}'.format(' '.join(train_cmd + hp_list)))
    process = subprocess.Popen(train_cmd + hp_list , stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)    
    stdout = []
    while True:
        line = process.stdout.readline()
        #stdout.append(line)
        print(line),
        if line == '' and process.poll() != None:
            break
    #print(process.stdout())
    
    
    if process.stderr != None:
        print('process error:')
        print(process.stderr.decode('utf-8'))
    

if __name__ == '__main__':
    main()