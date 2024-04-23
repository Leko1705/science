package science.nn.functional;

public class Tanh implements Function {
    @Override
    public double eval(double x) {
        return (Math.exp(x)-Math.exp(-x))/(Math.exp(x)+Math.exp(-x));
    }

    @Override
    public double gradient(double x) {
        return (4*Math.exp(2*x))/(Math.pow(Math.exp(2*x)+1, 2));
    }
}
