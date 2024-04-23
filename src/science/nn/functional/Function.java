package science.nn.functional;

public interface Function {

    double eval(double x);

    double gradient(double x);

}
