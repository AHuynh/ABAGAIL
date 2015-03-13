package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
*	Implementation of backprop, RHC, SA, and GA to find optimal weights to 
*	a neural network classifying heart disease. A lot of the code came 
*	from AbaloneTest by Hannah Lau and NNClassificationTest by Andrew Guillory
*
*	@author 	Mohamed El Banani
*	@version 	1.0
*/

public class WineTest {
	//get data instances
	private static Instance[] allInstances = initializeInstances();
    
    //Split instances into a test set and a training set
	private static Instance[] instances = Arrays.copyOfRange(allInstances, 0,196);
	private static Instance[] testInstances = Arrays.copyOfRange(allInstances, 197,296);

	//neural network properties (output layer corresponds to number of classes)
	private static int inputLayer = 11, hiddenLayer = 10 , outputLayer = 7; 

	//training properties
	private static int[] trainingItterations = {10, 50, 100, 200, 500, 1000, 2000, 5000};
    private static int[] numRepetitions = {10, 10, 10};
    private static int[] maxEpochs      = {10, 50, 100, 200, 500, 1000, 2000, 5000};
    private static double learningRate  = 0.5;
    private static double momentum      = 0.2;
    private static ErrorMeasure measure = new SumOfSquaresError();

	//Results to report (training time and evaluation)
    private static double[][] time = new double[3][];
    private static double[][] eval = new double[3][];


    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private static DataSet set = new DataSet(instances);
    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA ", "GA "};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args)
	{
        int numRuns = trainingItterations.length;        
        for (int z = 0; z < numRuns; z++
		{
            String header = "";
            results = "";
            header = "-----------------------------------------------------------\n";
            header += "-----------------------------------------------------------\n";
            header += "WineTest \n--------- \n\nDataset: Wine Quality White\n";
            header += "Input Layer: " + inputLayer+ "\nHidden Layer: " + hiddenLayer;
            header += "\nOutput Layer: " + outputLayer+ "\n\n";
            header += "Number of Iterations: (itterationsXrepititions) \n";
            header += "Randomized Hill Climbing: " + trainingItterations[z] +"x"+ numRepetitions[0];
            header += "\n";
            header += "Simulated Annealing:      " + trainingItterations[z] +"x" + numRepetitions[1];
            header += "\n";
            header += "Genetic Algorithm:        " + trainingItterations[z] +"x" + numRepetitions[2];
            header += "\n";
            header += "BackProp-RPROP (max):     " + maxEpochs[z] +"x" + numRepetitions[2];
            header += "\n";
            System.out.println(header);
    	   

            //int[] numEval = new int[3];
            for(int i = 0; i < oa.length; i++)
			{
				networks[i] = factory.createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});
				nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
			}

			oa[0] = new RandomizedHillClimbing(nnop[0]);
			oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
			oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

			// for(int i = 0; i < 2; i++) {
			for(int i = 0; i < oa.length; i++)
			{
				double tCorrect = 0, tIncorrect = 0, correct = 0, incorrect = 0;
				double start = 0 , end = 0, trainingTime = 0, testingTime = 0;
				int[][] confusion, confTrain;

				confusion = new int[outputLayer][outputLayer];
				confTrain = new int[outputLayer][outputLayer];
				start = System.nanoTime();
				System.out.print("Running: ");
				System.out.print(oaNames[i]);

				for (int k = 0; k < numRepetitions[i]; k++)
				{
					int loadingFactor = (int) numRepetitions[i]/10; 
					if ((k+1) % loadingFactor == 0)
						System.out.print("*");
						
					start = System.nanoTime();
					train(oa[i], networks[i], oaNames[i], z); //trainer.train();
					end = System.nanoTime();
					trainingTime += end - start;

					Instance optimalInstance = oa[i].getOptimal();
					networks[i].setWeights(optimalInstance.getData());

					int predicted, actual;
					start = System.nanoTime();
					for (int j = 0; j < testInstances.length; j++)
					{
						networks[i].setInputValues(testInstances[j].getData());
						networks[i].run();

						actual = testInstances[j].getLabel().getData().argMax();
						predicted = networks[i].getOutputValues().argMax();
						double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

						confusion[actual][predicted]++;
					}

					for (int j = 0; j < instances.length; j++)
					{
						networks[i].setInputValues(instances[j].getData());
						networks[i].run();
						actual = instances[j].getLabel().getData().argMax();
						predicted = networks[i].getOutputValues().argMax();
						double trash = Math.abs(predicted - actual) < 0.5 ? tCorrect++ : tIncorrect++;

						confTrain[actual][predicted]++;
					}

					end = System.nanoTime();
					testingTime += end - start;
				}

				trainingTime /= 10E6;
				testingTime /= 10E6;
				trainingTime /= numRepetitions[i];
				testingTime /= numRepetitions[i];


				results +=  "\nResults for " + oaNames[i] + ": \n\n"  + "Average Training Time (millisec): " + trainingTime + "\nAverage Testing Time (millisec): " + testingTime+ "\nTesting Accuracy:\nPercent correctly classified: " + df.format(correct/(correct+incorrect)*100);

				results += "\n\nTesting Confusion Matrix:\n";
				results += printConfusion(confusion);

				results += "\n\nTraining Accuracy:\nPercent correctly classified: " + df.format(tCorrect/(tCorrect+tIncorrect)*100);
				results += "\nTraining Confusion Matrix: \n";
				results += printConfusion(confTrain);

				System.out.println();
			}

            int nncorrect = 0;
            int nnincorrect = 0;
            int nntIncorrect = 0;
            int nntCorrect = 0;
            int[][] nnconfTrain = new int[7][7];
            int[][] nnconfusion = new int[7][7];
            int actual, predicted;
            WeightUpdateRule[] wU = {new RPROPUpdateRule()};
            String[] wUNames = {"RPROPUpdateRule"};
            for (int i = 0; i < wU.length; i++)
			{
                double start = 0, end = 0, trainingTime = 0, testingTime = 0;
                double numItterations = 0;
                
                System.out.print("Running: BackPropagation with ");
                System.out.println(wUNames[i]);

                for (int k = 0; k < numRepetitions[i+1]; k++)
				{
                    start = System.nanoTime();    
                    //Create a backprop neural network 
                    BackPropagationNetwork nnNetwork = factory.createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});

                    ConvergenceTrainer trainer = new ConvergenceTrainer(
                     new BatchBackPropagationTrainer(set, nnNetwork,
                         new SumOfSquaresError(),wU[i]),1E-10, maxEpochs[z]);
                    trainer.train();
                    end = System.nanoTime();
                    trainingTime += end - start;
                    numItterations += trainer.getIterations();

                    start = System.nanoTime();
                    for (int j = 0; j < testInstances.length; j++)
					{
                        nnNetwork.setInputValues(testInstances[j].getData());
                        nnNetwork.run();

                        actual = testInstances[j].getLabel().getData().argMax();
                        predicted = nnNetwork.getOutputValues().argMax();
                        double trash = Math.abs(predicted - actual) < 0.5 ? nncorrect++ : nnincorrect++;

                        nnconfusion[actual][predicted]++;
                    }

                    for( int j = 0; j < instances.length; j++)
					{
                        nnNetwork.setInputValues(instances[j].getData());
                        nnNetwork.run();

                        actual = instances[j].getLabel().getData().argMax();

                        predicted = nnNetwork.getOutputValues().argMax();
                        double trash = Math.abs(predicted - actual) < 0.5 ? nntCorrect++ : nntIncorrect++;

                        nnconfTrain[actual][predicted]++;
                    }
                    end = System.nanoTime();
                    testingTime += end - start;
                }

                trainingTime /= 10E6;
                testingTime /= 10E6;
                trainingTime /= numRepetitions[i+1];
                testingTime /= numRepetitions[i+1];
                numItterations /= numRepetitions[i+1];

                results +=  "\nResults for BackPropagation with (" + wUNames[i] + "): \n\n"   + " Average Training Time (millisec): " + trainingTime + "\nAverage Testing Time (millisec): " + testingTime+ "\nAverage Itteartions : " + numItterations + "\nTesting Accuracy:\nPercent correctly classified: " + df.format((100*nncorrect)/(nncorrect+nnincorrect));
                results += "\n\n testing Confusion \n";
                results += printConfusion(nnconfusion);
                results += "\n\nTraining Accuracy:\nPercent correctly classified: " + df.format((100*nntCorrect)/(nntCorrect+nntIncorrect));;
                results += "\n Training Confusion \n";
                results += printConfusion(nnconfTrain);
            }
            System.out.println(results);
			
			String fileOut = "src/opt/test/newResults_" + System.currentTimeMillis() + ".txt";
			
            try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(fileOut, true))))
			{
                out.println(header);
                out.println(results);
            } catch (IOException e) {
				e.printStackTrace();
            } 
        } 
    }

    private static Instance[] initializeInstances()
	{
        double[][][] attributes = new double[4898][][];

        try
		{
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/wine.txt")));

            for (int i = 0; i < attributes.length; i++)
			{
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
				attributes[i][0] = new double[10]; // 10 attributes
				attributes[i][1] = new double[1];

				for(int j = 0; j < 10; j++)
					attributes[i][0][j] = Double.parseDouble(scan.next());

				attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];
        for (int i = 0; i < instances.length; i++)
		{
            instances[i] = new Instance(attributes[i][0]);
            // classifications are 3, 4, 5, 6, 7, 8, or 9
            double classification = attributes[i][1][0];
            double[] label = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            if (classification == 9) {
                label[6] = 1;
            } else if (classification >= 8) {
            	label[5] = 1;
            } else if (classification >= 7) {
                label[4] = 1;
            } else if (classification >= 6) {
                label[3] = 1;
            } else if (classification >= 5) {
                label[2] = 1;
            } else if (classification >= 4) {
                label[1] = 1;
            } else {
            	label[0] = 1;
            }
            instances[i].setLabel(new Instance(label));
        }

        return instances;
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int index)
	{
        for (int i = 0; i < trainingItterations[index]; i++)
		{
            oa.train();
            double error = 0;
            for (int j = 0; j < instances.length; j++)
			{
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel();
                Instance example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(network.getOutputValues()));
                error += measure.value(output, example);
            }
        }
    }

    private static String printConfusion(int[][] conf)
	{
        String out = "";
    	for (int i = 0; i < conf.length; i++)
		{
        	for (int j = 0; j < conf[0].length; j++)
			{
        		if (i == j) {
    				out += "(";
    			}
        		out  += conf[i][j];
        		if (i == j) {
        			out += ")";
                }
                out  += ",\t";
            }
            out += "\n";
        }
        return out;
    }

}