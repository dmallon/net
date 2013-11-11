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

	/**
	 * 
	 * @param iTrainingStrategy
	 */
	public void setTrainingStrategy(ITrainingStrategy iTrainingStrategy) {}

	/**
	 * 
	 * @param trainSet
	 * @param numSamples
	 * @param epochs
	 * @param classes
	 */
	public void train(double[][] trainSet, int numSamples, int epochs, char[] classes) {}

	/**
	 * 
	 * @param trainSet
	 * @param index
	 * @return
	 */
	public int process(double[][] trainSet, int index) {
		return 0;
	}
	
	/**
	 * 
	 * @param trainSet
	 * @param numSamples
	 * @param classes
	 * @return
	 */
	public double test(double[][] trainSet, int numSamples, char[] classes){
		return 0.0;
	}
}
