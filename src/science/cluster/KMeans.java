package science.cluster;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class KMeans implements Classifier {

    private final int size;
    private final ArrayList<double[]> dataSet;
    private final double[][] centroids;

    public KMeans(int dataSize, int K){
        size = dataSize;
        dataSet = new ArrayList<>();
        Random random = new Random();
        centroids = new double[K][];

        for (int i = 0; i < centroids.length; i++) {
            double[] s = new double[size];
            ArrayList<Double> cache = new ArrayList<>();
            for (int j = 0; j < s.length; j++) {
                double r = random.nextInt(101);
                while (cache.contains(r)) {
                    r = random.nextInt(101);
                }
                s[j] = r;
                cache.add(r);
            }
            centroids[i] = s;
        }

    }

    public void addData(double[] dataset){
        if (dataset.length != size)
            throw new IllegalArgumentException("dataset length does not match with defined Set siz");
        this.dataSet.add(dataset);
    }

    @Override
    public void removeData(double[] dataset) {
        for (int i = 0; i < dataSet.size(); i++) {
            double[] point = dataSet.get(i);
            if (Arrays.equals(dataset, point)) {
                dataSet.remove(i);
                return;
            }
        }
    }

    public List<double[]> getData(){
        return dataSet;
    }

    public void cluster(){
        cluster(null);
    }

    @SuppressWarnings("unchecked")
    public void cluster(Integer maxIterations){

        if (dataSet.isEmpty()) return;

        int iteration = 0;

        while (true) {
            ArrayList<double[]>[] centroidAssign = new ArrayList[centroids.length];
            for (int i = 0; i < centroidAssign.length; i++){
                centroidAssign[i] = new ArrayList<>();
            }

            // calculate nearest centroids
            for (double[] set : dataSet) {
                int nearestCentroid = calculateNearestCentroid(set);
                centroidAssign[nearestCentroid].add(set);
            }

            // calculate each mean centroid
            double[][] cache = Arrays.copyOfRange(centroids, 0, centroids.length);
            for (int i = 0; i < centroids.length; i++) {
                if (!centroidAssign[i].isEmpty()) centroids[i] = calculateMeanCentroid(centroidAssign[i]);
            }

            if (Arrays.deepEquals(cache, centroids)) break;

            iteration += 1;

            if (maxIterations != null && iteration == maxIterations) break;
        }

    }

    public int classify(double[] dataset){
        if (dataset.length != size)
            throw new IllegalArgumentException("dataset length does not match with defined Set siz");
        return calculateNearestCentroid(dataset);
    }

    private int calculateNearestCentroid(double[] dataset){
        double min = Double.POSITIVE_INFINITY;
        int c = -1;
        for (int i = 0; i < centroids.length; i++){
            double distance = euclideanDistance(centroids[i], dataset);
            if (distance < min){
                min = distance;
                c = i;
            }
        }
        return c;
    }

    private static double euclideanDistance(double[] v1, double[] v2){
        double sum = 0.0;
        for (int i = 0; i < v1.length; i++){
            sum += Math.pow(v1[i] - v2[i], 2);
        }
        return Math.sqrt(sum);
    }

    private static double[] calculateMeanCentroid(ArrayList<double[]> data){
        double[] result = new double[data.get(0).length];
        for (double[] point : data) {
            for (int j = 0; j < result.length; j++) {
                result[j] += point[j];
            }
        }
        for (int i = 0; i < result.length; i++)
            result[i] /= data.size();
        return result;
    }


}
