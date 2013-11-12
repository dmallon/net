package neuralNet;

import static org.junit.Assert.*;

import org.junit.Test;

public class ChromosomeTest {

	@Test
	public void test() {
		Chromosome c = new Chromosome(20);
		c.print();
		Chromosome c2 = new Chromosome(20);
		c2.print();
		c.crossover(c2).print();
	}

}
