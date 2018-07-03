from train import PreTrain, Train
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--pretrain", help="option for pre-Training", action="store_true")
	parser.add_argument("--train", help="option for Training", action="store_true")
	parser.add_argument("--test", help="option for Test with Image ex) python main.py --test -i image_name", action="store_true")
	parser.add_argument("-i", "--image", help="input Image for Testing")
	args = parser.parse_args()
	
	if args.pretrain:
		pretrain = PreTrain()
		pretrain.training()

	elif args.train:
		print("train")

	elif args.test:
		if(args.image):
			print(args.image)
		else:
			print("please input image with -i or --image")
		


	#train = Train()
	#train.training()
	return 1

if __name__=="__main__":
	main()