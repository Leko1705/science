package science.nn.functional;

public class Phistep implements Function{
    @Override
    public double eval(double x) {
        return x >= 0 ? 1 : 0;
    }

    @Override
    public double gradient(double x) {
        return 0;
    }
}
