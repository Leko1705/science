package science.nn.loss;

public class HuberLoss extends Loss{

    private double alpha = 0.5;

    public HuberLoss(double[] output, double[] target) {
        super(output, target);
    }

    public HuberLoss(double[] output, double[] target, double alpha) {
        super(output, target);
        this.alpha = alpha;
    }

    @Override
    protected double calculate(double[] output, double[] target) {
        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            if (Math.abs(output[i] - target[i]) <= alpha)
                sum += Math.pow(output[i] - target[i], 2);
            else
                sum += Math.abs(output[i] - target[i]);
        }
        return sum/getAbsoluteCalculations();
    }

    @Override
    protected double gradientAt(double output, double target) {
        if (Math.abs(output - target) <= alpha){
            if (output > target)
                return +1;
            else
                return -1;
        }
        else
            return output - target;
    }
}
