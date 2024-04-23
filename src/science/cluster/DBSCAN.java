package science.cluster;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DBSCAN implements Classifier {

    private final int size;
    private int minPts;
    private double maxDistance;
    private final ArrayList<Point> points;

    public DBSCAN(int dataSize, double maxDistance, int minPts){
        this.size = dataSize;
        this.maxDistance = maxDistance;
        this.minPts = minPts;
        points = new ArrayList<>();
    }

    public void setMaxDistance(double maxDistance) {
        this.maxDistance = maxDistance;
    }

    public void setMinPts(int minPts) {
        this.minPts = minPts;
    }


    public void addData(double[] dataset){
        if (dataset.length != size)
            throw new IllegalArgumentException("dataset length does not match with defined Set size");
        this.points.add(new Point(dataset));
    }

    @Override
    public void removeData(double[] dataset) {
        for (int i = 0; i < points.size(); i++) {
            Point point = points.get(i);
            if (Arrays.equals(dataset, point.value)) {
                points.remove(i);
                return;
            }
        }
    }

    @Override
    public List<double[]> getData() {
        List<double[]> data = new ArrayList<>();
        for (Point point : points) {
            data.add(point.value);
        }
        return data;
    }


    public void cluster(){

        for (Point point : points){
            point.setCluter(null);
            point.setMarked(false);
        }


        int nextCluster = -1;
        for (Point point : points){

            if (!point.isMarked()){
                point.setMarked(true);
                ArrayList<Point> neighbours = getNeighbours(point);

                if (neighbours.size() >= minPts){
                    nextCluster++;
                    point.setCluter(nextCluster);

                    for (int i = 0; i < neighbours.size(); i++){
                        if (!neighbours.get(i).isMarked()){
                            neighbours.get(i).setMarked(true);
                            ArrayList<Point> subNeighbours = getNeighbours(neighbours.get(i));

                            if (subNeighbours.size() >= minPts)
                                neighbours.addAll(subNeighbours);
                        }

                        if (neighbours.get(i).getCluter() == null){
                            neighbours.get(i).setCluter(nextCluster);
                        }
                    }
                }
            }
        }

        for (Point point : points){
            if (point.getCluter() == null)
                point.setCluter(++nextCluster);
        }

    }

    public int classify(double[] data){
        for (Point point : points){
            if (Arrays.equals(data, point.getValue())){
                return point.getCluter();
            }
        }
        return -1;
    }

    private ArrayList<Point> getNeighbours(Point point){
        double[] v1 = point.getValue();
        ArrayList<Point> neighbours = new ArrayList<>();
        for (Point point1 : points){
            double[] v2 = point1.getValue();
            double distance = euclideanDistance(v1, v2);
            if (distance <= maxDistance)
                neighbours.add(point1);
        }
        return neighbours;
    }


    private static double euclideanDistance(double[] v1, double[] v2){
        double sum = 0.0;
        for (int i = 0; i < v1.length; i++){
            sum += Math.pow(v1[i] - v2[i], 2);
        }
        return Math.sqrt(sum);
    }



    private static class Point{
        private final double[] value;
        private Integer cluter;
        private boolean marked;
        public Point(double[] value){
            this.value = value;
        }
        public Integer getCluter() {
            return cluter;
        }
        public void setCluter(Integer cluter) {
            this.cluter = cluter;
        }
        public boolean isMarked() {
            return marked;
        }
        public void setMarked(boolean marked) {
            this.marked = marked;
        }
        public double[] getValue() {
            return value;
        }
    }
}
