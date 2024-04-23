package science.nn.functional;

public class Identity implements Function {
    @Override
    public double eval(double x) {
        return x;
    }

    @Override
    public double gradient(double x) {
        return 1;
    }
}
