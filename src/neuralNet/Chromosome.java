package neuralNet;

import java.util.Random;

public class Chromosome implements Comparable<Chromosome>{
	
	double[] chromosome;
	double mutateProb = 0.5;
	double crossOverProb = 0.5;
	double initRange = 0.3;
	double maxMutateStepSize = 0.5;
	int crossOverPoint;
	int size;
	private double fitness;
	
	
	public double getFitness() {
		return fitness;
	}


	public void setFitness(double fitness) {
		this.fitness = fitness;
	}


	public Chromosome(int size){
		this.size = size;
		
		this.chromosome = new double[size];
		this.crossOverPoint = (int) size/2;
		
		// Initialize chromosome with random values
		Random r = new Random();
		for (int i = 0; i < chromosome.length; i++) {
			this.chromosome[i] = -initRange + (2 * initRange) * r.nextDouble();
		}
	}
	
	
	/**
	 * 
	 * @param chromosome The chromosome to be used in conjunction for crossover 
	 * @return Chromosome The child chromosome with crossover applied.
	 */
	public Chromosome crossover(Chromosome chromosome){
		
		Chromosome child = new Chromosome(this.size);
		System.arraycopy(this.chromosome, 0, child.chromosome, 0, this.size);
		
		for (int i = 0, j = this.size/2; i < this.size/2 -1; i++, j++) {
			child.chromosome[i] = chromosome.chromosome[i];
			child.chromosome[j] = this.chromosome[j];
		}
		
		return child;
	}

	/**
	 * 
	 * @return Chromosome The mutated chromosome
	 */
	public Chromosome mutate(){
		
		Chromosome mutant = new Chromosome(this.size);
		System.arraycopy(this.chromosome, 0, mutant.chromosome, 0, this.size);
		
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
	
	
	public void print(){
		
		for (int i = 0; i < chromosome.length; i++) {
			System.out.print(chromosome[i] + " ");
		}
		System.out.println();
	}



	@Override
	public int compareTo(Chromosome c) {
		return this.fitness > c.fitness ? 1
			 : this.fitness < c.fitness ? -1
		     : 0;
	}
	
}
