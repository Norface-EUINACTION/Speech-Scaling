import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--countries", default=None, type=str, help="Add country")

def main(args: argparse.Namespace) -> None:
	args = [args.countries]

	print ("Starting scaling....")
	for arg in args:
		print('Scaling for ' + arg)
		os.system("python run_scaler_mps.py {} multi no".format(arg))
		print('Scaling Finished for ' + arg + '\n')
		print('-----------------------------------------------------------------------------')

	print('Normalizing Scores')
	for arg in args:
		os.system("python run_scaler_mps.py {} multi yes".format(arg))

	print('Finished!!! ')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)