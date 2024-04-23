package science.nn.loss;

import java.util.Objects;

public abstract class Loss {

    private double[] output;

    private double[] target;

    private int absoluteCalculations = 0;

    public Loss(double[] output, double[] target) {
        this.output = output;
        this.target = target;
        checkArrays();
    }

    public void setOutput(double[] output) {
        this.output = output;
        checkArrays();
    }

    public void setTarget(double[] target) {
        this.target = target;
        checkArrays();
    }

    public double calculate(){
        absoluteCalculations++;
        return calculate(output, target);
    }

    public double gradientAt(int idx){
        return gradientAt(output[idx], target[idx]);
    }

    protected int getAbsoluteCalculations() {
        return absoluteCalculations;
    }

    protected abstract double calculate(double[] output, double[] target);

    protected abstract double gradientAt(double output, double target);

    private void checkArrays(){
        Objects.requireNonNull(output);
        Objects.requireNonNull(target);
        if (output.length != target.length) {
            throw new IllegalArgumentException("Output and target arrays must have the same length");
        }
    }

}
