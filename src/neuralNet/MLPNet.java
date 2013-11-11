package neuralNet;


public class MLPNet{	
	protected int numInputs;
	protected int numLayers;
	protected int numOutputs;
	protected int numNodes;
	protected char classifier;
	
	protected double rate;
	
	protected Neuron[] output;
	protected Neuron[][] hidden;
	
	protected ITrainingStrategy trainingStrategy;
	
	MLPNet(int inputs, int layers, int outputs, double rate, char classifier){	
		this.numInputs = inputs;
		this.numLayers = layers;
		this.numOutputs = outputs;
		this.numNodes = inputs;
		this.rate = rate;
		this.classifier = classifier;
		this.trainingStrategy = null;
		
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
		this.trainingStrategy.train(this, trainSet, numSamples, epochs, classes);
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
	
	public double test(double[][] trainSet, int numSamples, char[] classes){
		double[] firstIn = new double[this.numInputs];
		double[] inputs = new double[this.numNodes];
		double[] outputs = new double[this.numNodes];
		double[] finalOut = new double[this.numOutputs];
		int incorrect = 0;
		double fitness = 0.0;
		double out = 0.0;
		
		for (int i = 0; i < numSamples; i++){
			for (int j = 0; j < numInputs; j++){
				firstIn[j] = trainSet[j][i];
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
				if(finalOut[j] > 0.75)
					out = 1.0;
				else
					out = 0.0;					
			}
			if(out == 0.0 && this.classifier == classes[i])
				incorrect++;
			else if (out == 1.0 && this.classifier != classes[i])
				incorrect++;		
		}
		fitness = 1.0 - ((double)incorrect)/((double)numSamples);
		return fitness;
	}
	
	public void setWeights(double[] chrom){
		int c = 0;
		for (int i = 0; i < this.numLayers; i++){
			for (int j = 0; j < this.numNodes; j++){
				for (int k = 0; k < this.numInputs; k++){
					this.hidden[j][i].setWeight(k, chrom[c]);
					c++;
				}
			}
		}
		for (int i = 0; i < this.numInputs; i++){
			this.output[0].setWeight(i, chrom[c]);
		}
	}

	public ITrainingStrategy getTrainingStrategy() {
		return trainingStrategy;
	}

	public void setTrainingStrategy(ITrainingStrategy trainingStrategy) {
		this.trainingStrategy = trainingStrategy;
	}
	
}
