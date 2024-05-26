package science.nn.functional;

public class Swish implements Function {

    private static final Function sigmoid = new Sigmoid();

    @Override
    public double eval(double x) {
        return x * sigmoid.eval(x);
    }

    @Override
    public double gradient(double x) {
        return eval(x) + sigmoid.eval(x) * (1 - eval(x));
    }
}
