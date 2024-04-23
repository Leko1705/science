package science.nn.model;

import science.nn.loss.Loss;
import science.nn.optim.Optimizer;

public interface Model extends Generator {

    double[] generate(double[] in);

    void backward(Loss loss);

    void fit(Optimizer optimizer);

    void zeroGradients();

}
