//package bot;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.TextDirectoryLoader;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.WordsFromFile;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;
//import weka.core.converters.ConverterUtils.DataSource;

public class Preprocess {

	public static void main(String[] args) {
		// convert the directory into a dataset
		TextDirectoryLoader loader = new TextDirectoryLoader();
		try {
			//here each text file is transformed into one string in the dataset
			//each string has a class (BotNet or Normal)
			//the dir should contain two subdirs, one contains files with Normal code
			//the other contains files with Botnet code
			loader.setDirectory(new File("Path/To/SourceCodeDirs/"));
			Instances rawData = loader.getDataSet();

			//Make a filter
			StringToWordVector filter = new StringToWordVector();

			//Make a tokenizer
			WordTokenizer wt = new WordTokenizer();
			String delimiters = " \r\t\n.,;:\'\"()?!-><#$\\%&*+/@^_=[]{}|`~0123456789";
			wt.setDelimiters(delimiters);
			filter.setTokenizer(wt);
			//Inform filter about dataset
			filter.setInputFormat(rawData);

			//number of words to keep     
			filter.setWordsToKeep(5000);

			//apply TF-IDF transform
			filter.setIDFTransform(true);
			filter.setTFTransform(true);

			//use stemming
			LovinsStemmer stemmer = new LovinsStemmer();
			filter.setStemmer(stemmer);

			//filter.setLowerCaseTokens(true);

			//use stopwords list to remove stop words
			WordsFromFile stopWords = new WordsFromFile();
			stopWords.setStopwords(new File("stopwords.txt"));
			filter.setStopwordsHandler(stopWords);

			//here is where we apply the filter
			Instances dataFiltered = Filter.useFilter(rawData, filter);

			//move class label to last index - the filter to use is Reorder
			Reorder reorder = new Reorder();
			reorder.setAttributeIndices("2-last,1");
			reorder.setInputFormat(dataFiltered);
			dataFiltered = Filter.useFilter(dataFiltered, reorder);

			//set class index to the last attribute
			//here we tell the dataset that the class is the last element/field/column
			dataFiltered.setClassIndex(dataFiltered.numAttributes() - 1);

			//save the dataset as ARFF file
			ArffSaver saver = new ArffSaver();
			//saver.setInstances(rawData);
			saver.setInstances(dataFiltered);
			saver.setFile(new File("data.arff"));
			saver.writeBatch();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
