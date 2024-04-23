package science.nn.loss;

public class MSELoss extends Loss {

    public MSELoss(double[] output, double[] target) {
        super(output, target);
    }

    @Override
    protected double calculate(double[] output, double[] target) {
        double sum = 0;
        for (int i = 0; i < output.length; i++)
            sum += Math.pow(output[i] - target[i], 2);
        return sum/getAbsoluteCalculations();
    }

    @Override
    protected double gradientAt(double output, double target) {
        return output - target;
    }
}
