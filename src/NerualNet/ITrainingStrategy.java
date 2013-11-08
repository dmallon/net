package NerualNet;

public interface ITrainingStrategy {

	void train(double[][] trainSet, int numSamples, int epochs, char[] classes,
			int numInputs, int numLayers, int numNodes, int numOutputs,
			Neuron[][] hidden, Neuron[] output, char classifier, double rate);
}
