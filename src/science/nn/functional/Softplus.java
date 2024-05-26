package science.nn.functional;

public class Softplus implements Function{

    private static final Function sigmoid = new Sigmoid();

    @Override
    public double eval(double x) {
        return Math.log(1+Math.exp(x));
    }

    @Override
    public double gradient(double x) {
        return sigmoid.eval(x);
    }
}
