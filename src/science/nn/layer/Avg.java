package science.nn.layer;

import science.nn.graph.Neuron;

public class Avg implements Poller1D, Poller2D {

    @Override
    public double poll(Neuron[][] neurons) {
        double sum = 0;
        int count = 0;

        for (Neuron[] row : neurons) {
            for (Neuron neuron : row) {
                sum += neuron.getValue();
                count++;
            }
        }
        return sum / count;
    }

    @Override
    public double poll(Neuron[] neurons) {
        double sum = 0;
        int count = 0;
        for (Neuron neuron : neurons) {
            sum += neuron.getValue();
            count++;
        }
        return sum / count;
    }
}
