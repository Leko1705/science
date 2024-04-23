package science.nn.loss;

public class MAELoss extends Loss{

    public MAELoss(double[] output, double[] target) {
        super(output, target);
    }

    @Override
    protected double calculate(double[] output, double[] target) {
        double sum = 0;
        for (int i = 0; i < output.length; i++)
            sum += Math.abs(output[i] - target[i]);
        return sum/getAbsoluteCalculations();
    }

    @Override
    protected double gradientAt(double output, double target) {
        if (output > target)
            return +1;
        else
            return -1;
    }
}
