import argparse


# Need all of this to deal with the filesystem
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Folder for all the input stuff")
parser.add_argument("--output_dir", type=str, required=True, help="Folder for all the output stuff")


def main(args):
	# Example for using the directory
    with open(f"{args.input_dir}/input_file.txt", 'r') as f:
        print(f.read())


	# Move model to GPU
	#assert torch.cuda.is_available()==True, "No GPU available. Check your Slurm configuration"
	#model = model.cuda()

	# Example for outputting
    with open(f"{args.output_dir}/output_file.txt", 'w') as f:
        f.write('frog')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("done!")
