import argparse


def parseArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_classes', help="Number of classes", type=int, default=10)
	parser.add_argument('--n_filters', type=int, default=128)
	parser.add_argument('--filter_multiplier', type=float, default=0.5)
	parser.add_argument('--filter_size', type=int, default=3)
	parser.add_argument('-l', '--var_n_filters', nargs='+', type=int, required=False)
	parser.add_argument('--l_rate', type=float, default=0.001)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--optimizer', type=str, required=False)
	parser.add_argument('--activation', type=str, default='leakyrelu')
	parser.add_argument('--loss', type=str, default='categorical_crossentropy')
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--initializer', type=str, default='orthogonal')
	parser.add_argument('--data_augmentation', type=str, default=False)
	parser.add_argument('--denselayer_size', type=int, default=128)
	parser.add_argument('--batch_norm', type=str, default=True)
	parser.add_argument('--train_model', type=str, default=True)
	args = parser.parse_args()

	return args
