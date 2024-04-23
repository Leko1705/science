package science.nn.functional;

import java.util.ArrayList;

public class Softmax implements Function{

    private final ArrayList<Double> all = new ArrayList<>();
    private double sum;

    @Override
    public double eval(double x) {
        return Math.exp(x)/sum;
    }

    @Override
    public double gradient(double x) {
        double v = Math.exp(x);
        double subSum = 0.0;
        for (double d : all)
            if (d != v)
                subSum += d;
        all.clear();
        return (v*subSum)/(Math.pow(this.sum, 2));
    }

    public void add(double value){
        sum += Math.exp(value);
        all.add(Math.exp(value));
    }
}
