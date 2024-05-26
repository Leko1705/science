package science.nn.layer;

import science.nn.graph.Neuron;

public class Max implements Poller1D, Poller2D {
    @Override
    public double poll(Neuron[][] neurons) {
        double max = Double.NEGATIVE_INFINITY;

        for (Neuron[] rwo : neurons) {
            for (Neuron neuron : rwo) {
                max = Math.max(max, neuron.getValue());
            }
        }

        return max;
    }

    @Override
    public double poll(Neuron[] neurons) {
        double max = Double.NEGATIVE_INFINITY;
        for (Neuron neuron : neurons) {
            max = Math.max(max, neuron.getValue());
        }
        return max;
    }
}
