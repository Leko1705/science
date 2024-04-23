package science.nn.functional;

public class Relu implements Function{

    @Override
    public double eval(double x) {
        return Math.max(0, x);
    }

    @Override
    public double gradient(double x) {
        return (x > 0) ? 1 : 0;
    }
}
