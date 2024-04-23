package science.nn.functional;

public class Sigmoid implements Function{
    @Override
    public double eval(double x) {
        return 1/(1+Math.exp(-x));
    }

    @Override
    public double gradient(double x) {
        return eval(x) * (1-eval(x));
    }
}
