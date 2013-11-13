package neuralNet;

import java.util.Arrays;
import java.util.Random;

public class GATrainingStrategy implements ITrainingStrategy {

	private int pop_size = 30;
	private Chromosome[] pop;
	
	private double mut_rate = 0.5;
	private double cross_prob= 0.5;
	private double fitPrime;
	private Chromosome bestParents[];
	private Random rnd = new Random();
	public GATrainingStrategy(){
		
	}
	
	@Override
	public void train(MLPNet net, double[][] trainSet, int numSamples, int epochs, char[] classes) 
	{	
		// Initialize population and chromosome sizes
		Chromosome.size = (net.numLayers * net.numNodes + net.numOutputs) * net.numInputs;
		pop = new Chromosome[this.pop_size];
		
		// Fill population with random weights
		for (int i = 0; i < pop_size; i++){
			pop[i] = new Chromosome();
		}
		
		// Calculate fitness of each chromosome
		for (int i = 0; i < pop_size; i++){
			pop[i].setFitness(this.fitness(net, pop[i].chromosome, trainSet, numSamples, classes));
		}
		
		
		// Begin GA algorithm
		int t = 0;
		while (t < epochs){
			for (int i = 0; i < pop_size; i++){
				
				// Select two best-fit parents
				Arrays.sort(pop);
				bestParents = new Chromosome[]{this.pop[0], this.pop[1]};
				
				// Perform crossover and mutation on the two best-fit parents
				Chromosome child = bestParents[0].crossover(bestParents[1]).mutate();
				
				pop[i].setFitness(this.fitness(net, pop[i].chromosome, trainSet, numSamples, classes));
				
				// Replace worst fit with new child
				pop[pop.length-1] = child;
			}
			t++;
		}
		
		// Apply best-fit as final net weights
		Arrays.sort(pop);
		net.setWeights(pop[0].chromosome);		
	}
	
	private double fitness(MLPNet net, double[] chrom, double[][] trainSet, int numSamples, char[] classes){		
		net.setWeights(chrom);
		return net.test(trainSet, numSamples, classes);
	}
	
}
