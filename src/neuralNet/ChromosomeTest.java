package neuralNet;

import static org.junit.Assert.*;

import org.junit.Test;

public class ChromosomeTest {

	@Test
	public void test() {
		
		Chromosome.size = 30;
		Chromosome c = new Chromosome();
		Chromosome c2 = new Chromosome();
		Chromosome c3 = new Chromosome();
		
		Chromosome[] crossover_array = {c, c2, c3};
		Chromosome child = Chromosome.crossover(crossover_array, Chromosome.size);
		
		c.print();
		c2.print();
		c3.print();
		
		child.print();
		
	}

}
