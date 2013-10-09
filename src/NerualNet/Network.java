package NerualNet;

interface Network {
	void train(double[][] trainSet, int numSamples, int epochs);
	double process(double[][] inputs, int numSamples);
}
