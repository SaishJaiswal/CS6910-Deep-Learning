import argparse


def parseArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_classes', help="Number of classes", type=int, default=10)
	parser.add_argument('--n_layers', type=int, default=4)
	parser.add_argument('-l', '--layer_sizes', nargs='+', type=int, required=True)
	parser.add_argument('--l_rate', type=float, default=0.01)
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--optimizer', type=str, required=True)
	parser.add_argument('--activation', type=str, default='sigmoid')
	parser.add_argument('--loss', type=str, default='cross_entropy')
	parser.add_argument('--output_activation', type=str, default='softmax')
	parser.add_argument('--batch_size', type=int, default=1)
	args = parser.parse_args()

	return args
