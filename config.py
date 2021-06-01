config_local = {    # Data
    'STL10_TRAIN': './dataset/stl10/train_split',
    'STL10_TEST': './dataset/stl10/test_split',
    'SVHN_EXTRA': './dataset/svhn/extra',
    'SVHN_TEST': './dataset/svhn/test',
    'CIFAR10_RAW': './dataset/cifar10_raw',
    'BIASED_CIFAR10_TRAIN': './dataset/biased_cifar10/train',
    'BIASED_CIFAR10_TEST': './dataset_biased_cifar10/test',

    # Checkpoints
    'STL10_WEIGHTS': './ckpt/learner_0510_v100.pt',
    'SVHN_WEIGHTS': None,  # populate
    'CIFAR10_WEIGHTS': None,
}

config_cluster = {
    # Data
    'STL10_TRAIN': '/scratch/users/avento/datasets/raw_files/stl10_raw/train_imgs',
    'STL10_TEST': '/scratch/users/avento/datasets/raw_files/stl10_raw/test_imgs',
    'SVHN_EXTRA': '/scratch/users/avento/datasets/raw_files/svhn_raw/extra_imgs',
    'SVHN_TEST': '/scratch/users/avento/datasets/raw_files/svhn_raw/test_imgs',
    'BIASED_CIFAR10_TRAIN': '/scratch/users/avento/datasets/biased_cifar10/fine_tune_images',
    'BIASED_CIFAR10_TEST': '/scratch/users/avento/datasets/biased_cifar10/test_images',

    # Checkpoints
    'STL10_WEIGHTS': '/scratch/users/avento/model_weights/learner_0510_v100.pt',
    'SVHN_WEIGHTS': '/scratch/users/avento/model_weights/learning_svhn_0529_v100.pt',  
    'CIFAR10_WEIGHTS': '/scratch/users/avento/model_weights/learning_cifar_0530_v100.pt',
}
