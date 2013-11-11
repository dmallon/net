package neuralNet;

public interface ITrainingStrategy {

	void train(MLPNet net, double[][] trainSet, int numSamples, int epochs, char[] classes);
}
