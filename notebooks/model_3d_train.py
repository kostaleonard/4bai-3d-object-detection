import argparse
from dataset import *
from model_3d_bounding_box import *

BUCKET_NAME = 'sagemaker-4bai-project'
S3_TRAIN_IMAGE_DIR = ('s3://{0}/KITTI/data_object_image_2/training/'
                      'image_2/').format(BUCKET_NAME)
S3_LABEL_DIR = 's3://{0}/KITTI/training/label_2/'.format(BUCKET_NAME)
HISTORY_FIG_OUTFILE='model_3d'


def parse_args() -> argparse.Namespace:
    """Returns the command line arguments.
    :return: The command line arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description='Trains the 3D model.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--model_checkpoint_filename', type=str)
    return parser.parse_args()


def train_model(args: argparse.Namespace) -> None:
    """Trains the model using the command line arguments.
    :param args: The command line arguments.
    """
    partition = get_train_only_kitti_partition_s3(S3_TRAIN_IMAGE_DIR)
    model = get_model_3d_deepbox()
    train_args = DEFAULT_TRAIN_ARGS
    if args.epochs:
        train_args['epochs'] = args.epochs
    if args.batch_size:
        train_args['batch_size'] = args.batch_size
    if args.model_checkpoint_filename:
        train_args['model_checkpoint_filename'] = \
            args.model_checkpoint_filename
    history = train_s3(model, partition, S3_TRAIN_IMAGE_DIR, S3_LABEL_DIR)
    plot_history(history, outfile=HISTORY_FIG_OUTFILE, show=False)


if __name__ == '__main__':
    args = parse_args()
    train_model(args)
