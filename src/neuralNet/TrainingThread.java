package neuralNet;

public class TrainingThread extends Thread{
/*** Setup requirements for multithreading the training process **************/	
	MLPNet net;
	double[][] trainSet;
	int numSamples;
	int epochs;
	char[] classes;
	
	public TrainingThread(MLPNet net, double[][] trainSet, int numSamples, int epochs, char[] classes){
		this.net = net;
		this.trainSet = trainSet;
		this.numSamples = numSamples;
		this.epochs = epochs;
		this.classes = classes;
	}
	public void run(){
		this.net.train(this.trainSet, this.numSamples, this.epochs, this.classes);
	}
}
