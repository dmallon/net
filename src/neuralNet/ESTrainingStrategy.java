package neuralNet;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class ESTrainingStrategy implements ITrainingStrategy {

	private int pop_size = 30;
	private Chromosome[] pop;
	private int offspring_size = 15;
	private Chromosome[] offspring;
	private double mut_rate = 0.5;
	private double cross_prob= 0.5;
	private double fitPrime;
	private Random rnd = new Random();
	
	public ESTrainingStrategy(){
		 offspring = new Chromosome[offspring_size];
	}
	
	@Override
	public void train(MLPNet net, double[][] trainSet, int numSamples, int epochs, char[] classes) 
	{	
		// Initialize population and chromosome size
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
		
		
		// Begin ES algorithm
		int t = 0;
		while (t < epochs){
				
				// Generate offspring
				for (int i = 0; i < offspring_size; i++) {
					// Select a random parent
					Chromosome selectedParent = pop[rnd.nextInt(pop_size)];
					
					// Pick one chromosome from 
					Chromosome child = selectedParent.mutateStrategyParams().mutate();
					
					offspring[i] = child;
					
				}
				
				Chromosome[] combinedParentsAndOffspring = new Chromosome[offspring_size + pop_size];
				
				System.arraycopy(offspring, 0, combinedParentsAndOffspring, 0, offspring_size);
				System.arraycopy(pop, 0, combinedParentsAndOffspring, offspring_size, pop_size);
				
				
				// Calculate fitness of each chromosome of combined sub-population 
				for (int i = 0; i < combinedParentsAndOffspring.length; i++) {
					combinedParentsAndOffspring[i].setFitness(this.fitness(net, combinedParentsAndOffspring[i].chromosome, trainSet, numSamples, classes));
				}
				
				// Sort by fitness
				Arrays.sort(combinedParentsAndOffspring);
				
				// Take best fit for new population
				for (int i = 0; i < pop_size; i++) {
					pop[i] = combinedParentsAndOffspring[i];
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
