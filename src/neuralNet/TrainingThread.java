package neuralNet;

// Implements a training thread 
public class TrainingThread extends Thread{
/*** Setup requirements for multithreading the training process **************/	
	MLPNet net;
	double[][] trainSet;
	int numSamples;
	int epochs;
	char[] classes;
	
	// Constructor
	public TrainingThread(MLPNet net, double[][] trainSet, int numSamples, int epochs, char[] classes){
		this.net = net;
		this.trainSet = trainSet;
		this.numSamples = numSamples;
		this.epochs = epochs;
		this.classes = classes;
	}
	// Run function activates the thread
	public void run(){
		// Train the network
		this.net.train(this.trainSet, this.numSamples, this.epochs, this.classes);
	}
}
