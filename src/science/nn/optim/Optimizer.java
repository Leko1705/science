package science.nn.optim;

import science.nn.layer.Layer;

public interface Optimizer {

    void optimize(Layer layer);
}
