from train import PreTrain, Train
from predict import Test
import argparse
import cv2

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--pretrain", help="option for pre-Training", action="store_true")
	parser.add_argument("--train", help="option for Training", action="store_true")
	parser.add_argument("--test", help="option for Test with Image ex) python main.py --test -i image_name", action="store_true")
	parser.add_argument("-i", "--image", help="input Image for Testing")
	args = parser.parse_args()

	pretrain = PreTrain()
	pretrain.training()

	if args.pretrain:
		pretrain = PreTrain()
		pretrain.training()

	elif args.train:
		train = Train()
		train.training()

	elif args.test:
		if(args.image):
			img = cv2.resize(cv2.imread(args.image), (448, 448))
			test = Test()
			test.predict(img)
		else:
			print("please input image with -i or --image")
		
	return 1

if __name__=="__main__":
	main()