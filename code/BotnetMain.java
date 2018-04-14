package bot;

import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;



public class BotnetMain {
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try {

			//here we open/load the arff file
			DataSource data = new DataSource("data.arff");
			//here we get the actual dataset
			Instances dataset = data.getDataSet();				
			//here we tell the dataset that the class is in the last element/field
			dataset.setClassIndex(dataset.numAttributes() - 1);
			//here we just printout the number of features/attributes
			System.out.println(dataset.numAttributes());

			int seed = 1;  // the seed for randomizing the data
			int folds = 10;
			Random rand = new Random(seed);	
			double acc = 0.0;
			//here we just want to print the model name(J48, KNN..etc.) and Accuracy value for each model
			System.out.println("Model,Accuracy");

			//create several models, run 10 fold cross validation and get Accuracy metric
			J48 tree = new J48();
			//Initialise evaluation with the dataset
			Evaluation eval = new Evaluation(dataset);
			//this is where we apply 10 fold cross-validation
			eval.crossValidateModel(tree, dataset, folds, rand);
			//get the accuracy, here we used double because we need the accurate number ex. 92.21
			acc = eval.pctCorrect();
			System.out.println("J48,"+acc);
			//System.out.println("=======================================");

			NaiveBayes nb = new NaiveBayes();
			//this is where we apply 10 fold cross-validation
			eval.crossValidateModel(nb, dataset, folds, rand);
			//get the accuracy, here we used double because we need the accurate number ex. 92.21
			acc = eval.pctCorrect();
			System.out.println("NaiveBayes,"+acc);
			//System.out.println("=======================================");

			IBk knn = new IBk(5);
			//this is where we apply 10 fold cross-validation
			eval.crossValidateModel(knn, dataset, folds, rand);
			//get the accuracy, here we used double because we need the accurate number ex. 92.21
			acc = eval.pctCorrect();
			System.out.println("KNN,"+acc);
			//System.out.println("=======================================");

			RandomForest rf = new RandomForest();
			//this is where we apply 10 fold cross-validation
			eval.crossValidateModel(rf, dataset, folds, rand);
			//get the accuracy, here we used double because we need the accurate number ex. 92.21
			acc = eval.pctCorrect();
			System.out.println("RandomForest,"+acc);
			//System.out.println("=======================================");

			SMO smo = new SMO(); //SMO(Sequential Minimal Optimisation)
			//this is where we apply 10 fold cross-validation
			eval.crossValidateModel(smo, dataset, folds, rand);
			//get the accuracy, here we used double because we need the accurate number ex. 92.21
			acc = eval.pctCorrect();
			System.out.println("SMO,"+acc);
			//System.out.println("=======================================");
		} catch (IOException e) {
			e.printStackTrace();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
}
