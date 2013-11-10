package neuralNet;

public class Network {
	protected int numInputs;
	protected int numLayers;
	protected int numOutputs;
	protected int numNodes;
	protected char classifier;
	
	protected double rate;
	
	protected Neuron[] output;
	protected Neuron[][] hidden;
	
	protected ITrainingStrategy trainingStrategy;

	public void setTrainingStrategy(ITrainingStrategy iTrainingStrategy) {
		// TODO Auto-generated method stub
		
	}

	public void train(double[][] trainSet, int numSamples, int epochs,
			char[] classes) {
		// TODO Auto-generated method stub
		
	}

	public int process(double[][] trainSet, int index) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	public double test(double[][] trainSet, int numSamples, char[] classes){
		return 0.0;
	}
}
