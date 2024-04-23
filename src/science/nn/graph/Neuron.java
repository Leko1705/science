package science.nn.graph;

import java.util.Map;

public interface Neuron {

    double getValue();

    void setValue(double value);

    void setGradient(double gradient);

    double getGradient();

    Map<Neuron, Weight> getPredecessors();

    Map<Neuron, Weight> getSuccessors();

    double squash(double x);

    double derivative(double x);

    double getUnsquashed();

}
