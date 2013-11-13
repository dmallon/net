package neuralNet;

import java.util.Random;

// Implements training strategy for Differential Evolution
public class DETrainingStrategy implements ITrainingStrategy {
	private int pop_size = 100;
	private int chrom_len;
	private double initRange = 0.3;
	private double[][] pop;
	private double[] fit;
	private double[] trial_vect;
	private double[] xPrime;
	
	private double mut_rate = 0.3;
	private double cross_prob= 0.5;
	private double fitPrime;
	
	private Random rnd = new Random();
	
	// Constructor
	public DETrainingStrategy(){
		
	}
	
	// MLPNN training function using DE
	@Override
	public void train(MLPNet net, double[][] trainSet, int numSamples, int epochs, char[] classes) 
	{	
		// Initialize population and chromosome sizes
		chrom_len = (net.numLayers * net.numNodes + net.numOutputs) * net.numInputs;		
		pop = new double[pop_size][chrom_len];
		fit = new double[pop_size];
		trial_vect = new double[chrom_len];
		xPrime = new double[chrom_len];
		
		// Fill population with random weights
		for (int i = 0; i < pop_size; i++){
			for (int j = 0; j < chrom_len; j++){
				pop[i][j] = -initRange + (2 * initRange * rnd.nextDouble());
			}
		}
		
		// Populate initial fitness array
		for (int i = 0; i < pop_size; i++){
			fit[i] = this.fitness(net, pop[i], trainSet, numSamples, classes);
		}
		
		// Begin DE algorithm
		int t = 0;
		while (t < epochs){
			for (int i = 0; i < pop_size; i++){
				this.mutate();
				this.crossover(pop[i]);
				fitPrime = this.fitness(net, xPrime, trainSet, numSamples, classes);
				// Keep the most fit of out of the current chrom and the training vect
				if( fitPrime > fit[i]){
					System.arraycopy(xPrime, 0, pop[i], 0, chrom_len);
					fit[i] = fitPrime;
				}				
			}
			t++;
		}
		// Get the final most-fit chromosome to feed to the network
		int max = 0;
		for (int i = 1; i < pop_size; i++){
			if(fit[i] > fit[max])
				max = i;
		}
		net.setWeights(pop[max]);		
	}
	
	// Returns the fitness of the current chromosome when applied to the MLPNN and tested
	private double fitness(MLPNet net, double[] chrom, double[][] trainSet, int numSamples, char[] classes){		
		// Apply the chrom to the network
		net.setWeights(chrom);
		// Run through the test set to generate error
		return net.test(trainSet, numSamples, classes);
	}
	
	// Mutation operator for DE
	private void mutate(){
		int i2, i3;
		double[] x1 = new double[chrom_len];
		
		// Create a random trial vector
		for (int i = 0; i < chrom_len; i++){
			x1[i] = (rnd.nextDouble() * 0.6) - 0.3;
		}
		
		// Select two random vectors from population that are not equal
		do{
			i2 = rnd.nextInt(pop_size);
			i3 = rnd.nextInt(pop_size);
		}while(i2 == i3);
		
		// Mutate the trial vector
		for (int i = 0; i < chrom_len; i++){
			trial_vect[i] = x1[i] + mut_rate*(pop[i2][i] - pop[i3][i]);
		}		
	}
	
	// Crossover operator
	private void crossover(double[] chrom){
		// Apply uniform crossover with probability mask
		for (int i = 0; i < chrom_len; i++){
			if(rnd.nextDouble() > cross_prob){
				xPrime[i] = trial_vect[i];
			}
			else{
				xPrime[i] = chrom[i];
			}
		}
	}

}
