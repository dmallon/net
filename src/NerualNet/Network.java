package NerualNet;

interface Network {
	void train(double[][] trainSet, int numSamples, int epochs, char[] classes);
	char process(double[][] inputs, int numSamples, char classes);
}
