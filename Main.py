import argparse
import configparser
import os

import Training
import Evaluation

def check_action(value):
    if value != "training" and value != "evaluation":
        raise argparse.ArgumentTypeError("Invalid action argument")
    return value

def check_config_file(value):
    if os.path.exists(value):
        if value.endswith('.ini'):
            return value
    raise argparse.ArgumentTypeError("Invalid file path")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action="store", help='Action to perform: one between training and evaluation', dest="action", type=check_action, required=True)
    parser.add_argument('-f', action="store", help='Configuration file path', dest="file", type=check_config_file, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    config = configparser.ConfigParser()
    config.read(args.file)

    if (args.action == 'training'):
        Training.loss_type = config['LOSSFUNCTION']['LossType']; Training.window_size = int(config['LOSSFUNCTION']['WindowSize'])
        Training.scales = int(config['LOSSFUNCTION']['Scales']); Training.orients = int(config['LOSSFUNCTION']['Orientations'])
        
        Training.n_patches = int(int(config['TRAINING']['NPatches']) / 5); Training.patch_size = int(config['TRAINING']['PatchSize'])
        Training.batch_size = int(config['TRAINING']['BatchSize']); Training.epoch = int(config['TRAINING']['Epochs'])
        Training.lr = float(config['TRAINING']['LearningRate']); Training.decay_step = int(config['TRAINING']['DecayStep'])
        Training.decay_fac = float(config['TRAINING']['DecayFactor']); Training.save_period = int(config['TRAINING']['SavePeriod'])
        
        Training.train()

    else:
        Evaluation.weights_file = config['EVALUATION']['WeightsFile']; Evaluation.anomaly_metrics = config['EVALUATION']['AnomalyMetrics']
        Evaluation.ae_patch_size = int(config['EVALUATION']['PatchSize']); Training.ae_stride = int(config['EVALUATION']['Stride'])
        Evaluation.ae_batch_splits = int(config['EVALUATION']['BatchSplits']); Training.invert_reconstruction = bool(config['EVALUATION']['InvertReconstruction'])
        Training.fpr_value = float(config['EVALUATION']['ThresholdFPR']); eval_type = int(config['EVALUATION']['EvaluationType'])

        tresh = Evaluation.validation()
        if (eval_type == 0):
            Evaluation.evaluation_complete(tresh)
        elif (eval_type >  0 and eval_type < 41):
            Evaluation.evaluation(str(eval_type).zfill(2), tresh, True)
        else:
            raise argparse.ArgumentTypeError("Evaluation type")