package NerualNet;


public class MLPNet extends Network{
	private int numInputs;
	private int numLayers;
	private int numOutputs;
	private int numNodes;
	private char classifier;
	
	private double rate;
	
	private Neuron[] output;
	private Neuron[][] hidden;
	
	private ITrainingStrategy trainingStrategy;
	
	
	
	MLPNet(int inputs, int layers, int outputs, double rate, char classifier){	
		this.numInputs = inputs;
		this.numLayers = layers;
		this.numOutputs = outputs;
		this.numNodes = 2 * inputs;
		this.rate = rate;
		this.classifier = classifier;
		
		this.hidden = new Neuron[this.numNodes][this.numLayers];
		this.output = new Neuron[this.numOutputs];
				
		for (int i = 0; i < this.numLayers; i++){
			for (int j = 0; j < this.numNodes; j++){
				if(i == 0)
					this.hidden[j][i] = new Neuron(numInputs, 2);
				else
					this.hidden[j][i] = new Neuron(numNodes, 2);
			}
		}
		for (int j = 0; j < this.numOutputs; j++){
			this.output[j] = new Neuron(numNodes, 1);
		}
	}

	
	/**
	 * 
	 */
	public void train(double[][] trainSet, int numSamples, int epochs, char[] classes){
		this.trainingStrategy.train(trainSet, numSamples, epochs, classes, numInputs, numLayers, numNodes, numOutputs, hidden, output, classifier, rate);
	}
	
	
	/**
	 * 
	 */
	public int process(double[][] set, int index){
		double[] firstIn = new double[this.numInputs];
		double[] inputs = new double[this.numNodes];
		double[] outputs = new double[this.numNodes];
		double[] finalOut = new double[this.numOutputs];
		double out = 0.0;
		
		for (int j = 0; j < this.numInputs; j++){
			firstIn[j] = set[j][index];
		}
		
		// Hidden layers
		for (int j = 0; j < this.numLayers; j++){
			for (int k = 0; k < this.numNodes; k++){
				if(j == 0)
					this.hidden[k][j].activate(firstIn);
				else
					this.hidden[k][j].activate(inputs);
				outputs[k] = this.hidden[k][j].getOutput();
			}						
			for (int k = 0; k < this.numNodes; k++){
				inputs[k] = outputs[k];
			}
		}
		
		// Output layer (and calculate output delta)
		for (int j = 0; j < this.numOutputs; j++){
			this.output[j].activate(inputs);
			finalOut[j] = this.output[j].getOutput();
			System.out.println(finalOut[j]);
			if(finalOut[j] > 0.65)
				out = 1.0;
			else
				out = 0.0;					
		}
	
		if(out == 1.0)
			return (this.classifier);
		else
			return '!';
	}

	public ITrainingStrategy getTrainingStrategy() {
		return trainingStrategy;
	}

	public void setTrainingStrategy(ITrainingStrategy trainingStrategy) {
		this.trainingStrategy = trainingStrategy;
	}
	
}
