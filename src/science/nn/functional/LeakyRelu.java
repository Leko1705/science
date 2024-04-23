package science.nn.functional;

public class LeakyRelu implements Function{

    private final double a;

    public LeakyRelu(){
        a = 0.01;
    }

    public LeakyRelu(double a){
        this.a = a;
    }
    @Override
    public double eval(double x) {
        return x >= 0 ? x : a*x;
    }

    @Override
    public double gradient(double x) {
        return x >= 0 ? 1 : a;
    }
}
