package neuralNet;

public interface ITrainingStrategy {

	void train(Network net, double[][] trainSet, int numSamples, int epochs, char[] classes);
}
