package science.nn.loss;

public class CrossEntropyLoss extends Loss{

    public CrossEntropyLoss(double[] output, double[] target) {
        super(output, target);
    }

    @Override
    protected double calculate(double[] output, double[] target) {
        double sum = 0;
        for (int i = 0; i < output.length; i++)
            sum += Math.log(output[i]) * Math.log(target[i]);
        return -sum;
    }

    @Override
    protected double gradientAt(double output, double target) {
        return -(target/output) + (1-target)/(1-output);
    }
}
