package science.cluster;

import java.util.Collection;

public interface Classifier {

    void cluster();

    int classify(double[] dataset);

    void addData(double[] dataset);

    void removeData(double[] dataset);

    Collection<double[]> getData();

}
