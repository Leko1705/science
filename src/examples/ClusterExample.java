package examples;

import science.cluster.Classifier;
import science.cluster.KMeans;

import java.util.Arrays;

public class ClusterExample {

    public static void main(String[] args) {
        Classifier classifier = new KMeans(2, 2);

        // add data to classify
        classifier.addData(new double[]{0, 5});
        classifier.addData(new double[]{5, 0});

        // group all the added data
        classifier.cluster();

        // classify a new unknown data set
        double[] classified = new double[]{0, 5};
        int clazz = classifier.classify(classified);

        System.out.println("class-ID for " + Arrays.toString(classified) + ": " + clazz);
    }

}
