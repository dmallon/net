package neuralNet;

import java.util.Arrays;
import java.util.Random;

public class Chromosome implements Comparable<Chromosome>{
	
	double[] chromosome;
	double mutateProb = 0.5;
	double crossOverProb = 0.5;
	double initRange = 0.3;
	double maxMutateStepSize = 0.25;
	double maxMutateStrategyParamsSize = .001;
	int crossOverPoint;
	static int size;
	private double fitness;
	
	
	public double getFitness() {
		return fitness;
	}


	public void setFitness(double fitness) {
		this.fitness = fitness;
	}


	public Chromosome(){
		
		this.chromosome = new double[size];
		this.crossOverPoint = (int) size/2;
		
		// Initialize chromosome with random values
		Random r = new Random();
		for (int i = 0; i < chromosome.length; i++) {
			this.chromosome[i] = -initRange + (2 * initRange) * r.nextDouble();
		}
	}
	
	
	/**
	 * This crossover should be used with Genetic Algorithm to produce a recombined child from two parents.
	 * @param mates The other parent to be used as crossover mate.
	 * @return Chromosome The child chromosome with crossover applied.
	 */
	public Chromosome crossover(Chromosome mate){
		
		Chromosome child = new Chromosome();
		System.arraycopy(this.chromosome, 0, child.chromosome, 0, size);
		
		for (int i = 0, j = size; i < size -1; i++, j++) {
			child.chromosome[i] = mate.chromosome[i];
			child.chromosome[j] = this.chromosome[j];
		}
		
		return child;
	}
	
	
	/**
	 * This crossover method should be used for Evolutionary Strategies Algorithm to produce a recombined child from multiple parents.
	 * @param mates The parent chromosomes to be used in the crossover to produce the child.
	 * @return Chromosome The child chromosome with crossover applied.
	 */
	public static Chromosome crossover(Chromosome[] mates){
		
		Chromosome child = new Chromosome();
		
		for (int i = 0; i < mates.length; i++) {
			int start = size/mates.length * i;
			int end = start + size/mates.length;
			
			double[] slice = Arrays.copyOfRange(mates[i].chromosome, start, end);
		
			System.arraycopy(slice, 0, child.chromosome, start, slice.length);
		}
		
		return child;
	}


	/**
	 * 
	 * @return Chromosome The mutated chromosome
	 */
	public Chromosome mutate(){
		
		Chromosome mutant = new Chromosome();
		System.arraycopy(this.chromosome, 0, mutant.chromosome, 0, size);
		
		int[] indices = new int[(int) (mutant.chromosome.length * mutateProb)];
		int oldI = -1;
		Random r = new Random();
		
		for (int i = 0; i < indices.length; i++) {
			
			indices[i] = r.nextInt(mutant.chromosome.length);
			
			if(oldI != indices[i]){
				mutant.chromosome[indices[i]] += -maxMutateStepSize + (2 * maxMutateStepSize) * r.nextDouble();
			}
			else{
				i--;
				continue;
			}
			
		    oldI = indices[i];
		}
		
		return mutant;
	}
	
	
	public Chromosome mutateStrategyParams(){
		Random r = new Random();
		this.crossOverProb += -maxMutateStrategyParamsSize + (2 * maxMutateStrategyParamsSize) * r.nextDouble();
		this.mutateProb += -maxMutateStrategyParamsSize + (2 * maxMutateStrategyParamsSize) * r.nextDouble();
		
		return this;
	}
	
	
	public void print(){
		
		for (int i = 0; i < chromosome.length; i++) {
			System.out.print(chromosome[i] + " ");
		}
		System.out.println();
	}



	@Override
	public int compareTo(Chromosome c) {
		return this.fitness > c.fitness ? -1
			 : this.fitness < c.fitness ? 1
		     : 0;
	}
	
}
