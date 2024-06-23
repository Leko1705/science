package examples;

import science.cluster.Classifier;
import science.cluster.KMeans;

import java.util.Arrays;

public class ClusterExample {

    private static final int DATA_SIZE = 2;
    private static final int K_CENTROID_AMOUNT = 2;

    public static void main(String[] args) {
        Classifier classifier = new KMeans(DATA_SIZE, K_CENTROID_AMOUNT);

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
