package science.nn.layer;

import science.nn.graph.Neuron;

public class Min implements Poller1D, Poller2D {
    @Override
    public double poll(Neuron[][] neurons) {
        double min = Double.POSITIVE_INFINITY;

        for (Neuron[] row : neurons) {
            for (Neuron neuron : row) {
                min = Math.min(min, neuron.getValue());
            }
        }

        return min;
    }

    @Override
    public double poll(Neuron[] neurons) {
        double min = Double.POSITIVE_INFINITY;
        for (Neuron neuron : neurons) {
            min = Math.min(min, neuron.getValue());
        }
        return min;
    }

}
