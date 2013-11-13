package neuralNet;

// Thread implementation for network testing
public class TestThread extends Thread{
/*** Setup requirements for multithreading the testing process **************/	
	MLPNet[] nets;
	double[][] trainSet;
	char[] expected;
	int index;
	int numClasses;
	int[] results;
	
	// Thread constructor
	public TestThread(MLPNet[] nets, double[][] trainSet, char[] expected, int index, int numClasses){
		this.nets = nets;
		this.trainSet = trainSet;
		this.index = index;
		this.numClasses = numClasses;
		this.results = new int[3];
		this.expected = expected;
	}
	// Run function activates the thread
	public void run(){
		int out;
		int responses = this.numClasses;
		// Run the test set through the network and cacluate error
		for(int i = 0; i < numClasses; i++){
			out = this.nets[i].process(this.trainSet, this.index);
			if(out != '!'){
				if(out == expected[this.index]){
					//System.out.print(out + " ");
					this.results[0]++;
				}
				else{						
					//System.out.print("Error(" + out + ") ");
					this.results[1]++;
				}
			}
			else
				responses--;
		}
		if(responses == 0){
			//System.out.print("Failed to classify");
			this.results[2]++;
		}
			
	}
	// Return the results from the finished thread
	public int[] getResult(){
		return this.results;
	}
}
