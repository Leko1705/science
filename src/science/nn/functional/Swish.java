package science.nn.functional;

public class Swish implements Function {
    @Override
    public double eval(double x) {
        return x * new Sigmoid().eval(x);
    }

    @Override
    public double gradient(double x) {
        return eval(x) + new Sigmoid().eval(x) * (1 - eval(x));
    }
}
